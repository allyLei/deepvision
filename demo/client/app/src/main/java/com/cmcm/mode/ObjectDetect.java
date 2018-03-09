package com.cmcm.mode;

import android.content.Context;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.graphics.ImageFormat;
import android.graphics.Matrix;
import android.graphics.Rect;
import android.graphics.YuvImage;
import android.os.Handler;
import android.os.Message;
import android.renderscript.Allocation;
import android.renderscript.Element;
import android.renderscript.RenderScript;
import android.renderscript.ScriptIntrinsicResize;
import android.renderscript.ScriptIntrinsicYuvToRGB;
import android.renderscript.Type;

import com.apkfuns.logutils.LogUtils;
import com.cmcm.camera.CameraInterface;
import com.cmcm.util.ConfigUtil;
import com.cmcm.util.EventUtil;
import com.cmcm.util.HttpUtil;
import com.cmcm.util.IntenetUtil;

import org.json.JSONObject;

import java.io.ByteArrayOutputStream;
import java.util.concurrent.BlockingQueue;
import java.util.concurrent.LinkedBlockingQueue;
import java.util.concurrent.ThreadPoolExecutor;
import java.util.concurrent.TimeUnit;

/**
 * Created by zhangkai on 2017/8/23.
 */


public class ObjectDetect implements ObjectDetectListener {

    private static final int KEEP_ALIVE_TIME = 10;
    private static final TimeUnit TIME_UNIT = TimeUnit.SECONDS;
    private BlockingQueue<Runnable> workQueue;
    private ThreadPoolExecutor mThreadPool;
    private Context mContext;
    private Handler mHandler;
    private static int numSlow = 0;
    private static final int SLOW_THRESHOLD = 5;
    private static int numFast = 0;
    private static final int FAST_THRESHOLD = 3;
    private static int network;
    private static boolean detectEnabled;

    private static RenderScript rs;
    private static ScriptIntrinsicYuvToRGB yuvToRgbScript;
    private static ScriptIntrinsicResize resizeScript;

    public ObjectDetect(Context context, Handler handler) {
        int corePoolSize = Runtime.getRuntime().availableProcessors();
        int maximumPoolSize = corePoolSize * 2;
        workQueue = new LinkedBlockingQueue<Runnable>(64);
        mThreadPool = new ThreadPoolExecutor(corePoolSize, maximumPoolSize, KEEP_ALIVE_TIME, TIME_UNIT, workQueue);
        mHandler = handler;
        mContext = context;
        rs = RenderScript.create(context);
        yuvToRgbScript = ScriptIntrinsicYuvToRGB.create(rs, Element.RGBA_8888(rs));
        resizeScript = ScriptIntrinsicResize.create(rs);

        networkChanged();

        // WebSocketUtil.getInstance().setObjectListener(this);
        HttpUtil.setObjectListener(this);
    }

    public void stop() {
        // WebSocketUtil.getInstance().stop();
    }

    public void networkChanged() {
        network = IntenetUtil.getNetworkState(mContext);
        ConfigUtil.setNetworkState(network);
        detectEnabled = true;
        if (network != IntenetUtil.NETWORN_WIFI) {
            ConfigUtil.setInterval(0);
            CameraInterface.getInstance().stopObjectDetect();
            CameraInterface.getInstance().startObjectDetect();
            detectEnabled = ConfigUtil.getMobileEnabled();
            Message m = mHandler.obtainMessage();
            m.what = EventUtil.NETWORK_NOT_WIFI;
            m.sendToTarget();
        }
    }

    @Override
    public void onReceived(JSONObject result, long detectTime) {
        int interval = ConfigUtil.getInterval();
        int halfInterval = interval / 2;
        int doubleInterval = interval * 2;

        Message m = mHandler.obtainMessage();
        m.what = EventUtil.UPDATE_OBJECT_RECT;
        if (detectTime < ConfigUtil.getDelayTime()) {
            m.obj = result;
        } else {
            m.obj = null;
        }
        m.arg1 = (int) detectTime;
        m.sendToTarget();

        if (detectTime < doubleInterval) {
            numFast += 1;
            if (numFast >= FAST_THRESHOLD) {
                LogUtils.w("preview interval change: " + halfInterval);
                ConfigUtil.setInterval(halfInterval);
                CameraInterface.getInstance().stopObjectDetect();
                CameraInterface.getInstance().startObjectDetect();
                numFast = 0;
            }
        } else if (detectTime > doubleInterval * 2) {
            numSlow += 1;
            if (numSlow >= SLOW_THRESHOLD) {
                LogUtils.w("preview interval change: " + doubleInterval);
                ConfigUtil.setInterval(doubleInterval);
                CameraInterface.getInstance().stopObjectDetect();
                CameraInterface.getInstance().startObjectDetect();
                numSlow = 0;
            }
        } else {
            numSlow = 0;
            numFast = 0;
        }
    }

    public synchronized void post(final byte[] data, final int width, final int height) {
        mThreadPool.execute(new Runnable() {
            @Override
            public void run() {
                try {
                    process(data, width, height);
                } catch (Exception e) {
                    LogUtils.e(e);
                }
            }
        });
    }


    public static byte[] scale(Bitmap bmp, int width, int height, int quality) {
        Bitmap.Config config = bmp.getConfig();
        Bitmap newBmp = Bitmap.createBitmap(width, height, config);
        Allocation ain = Allocation.createFromBitmap(rs, bmp);
        Type type = Type.createXY(rs, ain.getElement(), width, height);
        Allocation aout = Allocation.createTyped(rs, type);
        resizeScript.setInput(ain);
        resizeScript.forEach_bicubic(aout);
        aout.copyTo(newBmp);
        ByteArrayOutputStream stream = new ByteArrayOutputStream();
        newBmp.compress(Bitmap.CompressFormat.JPEG, quality, stream);
        return stream.toByteArray();
    }

    public static byte[] yuvToRgb(byte[] data, int width, int height) {
        Type.Builder yuvType = new Type.Builder(rs, Element.U8(rs)).setX(data.length);
        Allocation in = Allocation.createTyped(rs, yuvType.create(), Allocation.USAGE_SCRIPT);

        Type.Builder rgbaType = new Type.Builder(rs, Element.RGBA_8888(rs)).setX(width).setY(height);
        Allocation out = Allocation.createTyped(rs, rgbaType.create(), Allocation.USAGE_SCRIPT);

        int scaleWidth = ConfigUtil.getScaleWidth();
        int scaleHeight = ConfigUtil.getScaleHeight();
        Type.Builder resizeType = new Type.Builder(rs, Element.RGBA_8888(rs)).setX(scaleWidth).setY(scaleHeight);
        Allocation aout = Allocation.createTyped(rs, resizeType.create(), Allocation.USAGE_SCRIPT);

        in.copyFrom(data);
        yuvToRgbScript.setInput(in);
        yuvToRgbScript.forEach(out);

        resizeScript.setInput(out);
        resizeScript.forEach_bicubic(aout);

        Bitmap bmp = Bitmap.createBitmap(scaleWidth, scaleHeight, Bitmap.Config.ARGB_8888);
        aout.copyTo(bmp);

        ByteArrayOutputStream stream = new ByteArrayOutputStream();
        bmp.compress(Bitmap.CompressFormat.JPEG, ConfigUtil.getJpegQuality(), stream);
        return stream.toByteArray();
    }


    public static byte[] yuvToJpeg(byte[] data, int width, int height) {
        ByteArrayOutputStream stream = new ByteArrayOutputStream();
        YuvImage im = new YuvImage(data, ImageFormat.NV21, width, height, null);
        im.compressToJpeg(new Rect(0, 0, width, height), ConfigUtil.getJpegQuality(), stream);
        Bitmap bmp = BitmapFactory.decodeByteArray(stream.toByteArray(), 0, stream.size());
        float scaleWidth = ((float) ConfigUtil.getScaleWidth()) / bmp.getWidth();
        float scaleHeight = ((float) ConfigUtil.getScaleHeight()) / bmp.getHeight();
        Matrix matrix = new Matrix();
        matrix.postScale(scaleWidth, scaleHeight);
        Bitmap newBmp = Bitmap.createBitmap(bmp, 0, 0, ConfigUtil.getScaleWidth(), ConfigUtil.getScaleHeight(), matrix, true);
        ByteArrayOutputStream baos = new ByteArrayOutputStream();
        newBmp.compress(Bitmap.CompressFormat.JPEG, 100, baos);
        return baos.toByteArray();
    }

    private void process(byte[] data, int width, int height) {
        if (data == null || data.length == 0) return;

        long start_ts = System.currentTimeMillis();
        byte[] outBytes = yuvToRgb(data, width, height);
        long compress_ts = System.currentTimeMillis();

        LogUtils.v("compress time: " + (compress_ts - start_ts));

        if (detectEnabled && IntenetUtil.getNetworkState(mContext) == network) {
            HttpUtil.startDetect(outBytes, start_ts, width, height);
        } else {
            networkChanged();
        }

    }
}