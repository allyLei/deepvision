package com.cmcm.camera;

import android.content.res.Configuration;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.graphics.ImageFormat;
import android.hardware.Camera;
import android.hardware.Camera.AutoFocusCallback;
import android.hardware.Camera.PictureCallback;
import android.hardware.Camera.PreviewCallback;
import android.hardware.Camera.ShutterCallback;
import android.hardware.Camera.Size;
import android.view.SurfaceView;

import com.apkfuns.logutils.LogUtils;
import com.cmcm.mode.ObjectDetect;
import com.cmcm.util.CamParaUtil;
import com.cmcm.util.ConfigUtil;
import com.cmcm.util.FileUtil;
import com.cmcm.util.ImageUtil;

import java.io.IOException;
import java.util.List;
import java.util.Timer;
import java.util.TimerTask;

public class CameraInterface {
    private Camera mCamera;
    private Camera.Parameters mParams;
    private boolean isPreviewing = false;
    private int mCameraId = -1;
    private static CameraInterface mCameraInterface;
    private static Timer timer = null;
    private ObjectDetect objectDetect = null;
    private static int previewWidth;
    private static int previewHeight;

    public void setObjectDetectListener(ObjectDetect objectDetect) {
        this.objectDetect = objectDetect;
    }

    public ObjectDetect getObjectDetectListener(){
        return objectDetect;
    }


    public interface CamOpenOverCallback {
        void cameraHasOpened();
    }

    private CameraInterface() {
    }

    public static synchronized CameraInterface getInstance() {
        if (mCameraInterface == null) {
            mCameraInterface = new CameraInterface();
        }
        return mCameraInterface;
    }


    /**
     * 打开Camera
     *
     * @param callback
     */
    public void doOpenCamera(CamOpenOverCallback callback, int cameraId) {
        LogUtils.i("Camera open....");
        doStopCamera();
        mCamera = Camera.open(cameraId);
        mCameraId = cameraId;
        if (callback != null) {
            callback.cameraHasOpened();
        }
    }

    /**
     * 停止预览，释放Camera
     */
    public void doStopCamera() {
        if (null != mCamera) {
            mCamera.setPreviewCallback(null);
            mCamera.setOneShotPreviewCallback(null);
            if (isPreviewing) mCamera.stopPreview();
            isPreviewing = false;
            mCamera.release();
            mCamera = null;
        }
    }

    /**
     * 开启预览
     *
     * @param surfaceView
     * @param previewRate
     */
    public void doStartPreview(SurfaceView surfaceView, float previewRate) {
        LogUtils.i("doStartPreview...");
        if (isPreviewing) {
            mCamera.stopPreview();
        }
        if (mCamera != null) {
            mParams = mCamera.getParameters();
            mParams.setPictureFormat(ImageFormat.JPEG);
            mParams.setPreviewFormat(ImageFormat.NV21);

            LogUtils.v("previewRate: " + previewRate);
            Size pictureSize = CamParaUtil.getInstance().getOptimalSize(
                    mParams.getSupportedPictureSizes(), previewRate, ConfigUtil.getMinHeight());
            mParams.setPictureSize(pictureSize.width, pictureSize.height);

            Size previewSize = CamParaUtil.getInstance().getOptimalSize(
                    mParams.getSupportedPreviewSizes(), previewRate, ConfigUtil.getMinHeight());
            mParams.setPreviewSize(previewSize.width, previewSize.height);

            List<String> focusModes = mParams.getSupportedFocusModes();
            if (focusModes.contains("continuous-video")) {
                mParams.setFocusMode(Camera.Parameters.FOCUS_MODE_CONTINUOUS_VIDEO);
            }

            CamParaUtil.getInstance().printSupportPreviewSize(mParams);
            LogUtils.i(mParams.getSupportedPreviewFormats());
            LogUtils.i(mParams.getSupportedFocusModes());

            // 如果是竖屏
            if (surfaceView.getContext().getResources().getConfiguration().orientation != Configuration.ORIENTATION_LANDSCAPE) {
                LogUtils.v("set display orientation 90");
                mCamera.setDisplayOrientation(90);
            } else {
                mCamera.setDisplayOrientation(0);
            }

            // mParams.set("iso", "200");

            mCamera.setParameters(mParams);

            try {
                mCamera.setPreviewDisplay(surfaceView.getHolder());
                mCamera.startPreview();
            } catch (IOException e) {
                LogUtils.e(e);
            }

            isPreviewing = true;

            mParams = mCamera.getParameters();
            previewWidth = mParams.getPreviewSize().width;
            previewHeight = mParams.getPreviewSize().height;

            LogUtils.i("最终设置:PreviewSize--With = " + previewWidth
                    + "Height = " + previewHeight);
            LogUtils.i("最终设置:PictureSize--With = " + mParams.getPictureSize().width
                    + "Height = " + mParams.getPictureSize().height);
        }
    }

    public void startObjectDetect() {
        TimerTask timerTask = new TimerTask() {
            @Override
            public void run() {
                if (null != mCamera && isPreviewing) {
                    //   mCamera.autoFocus(mAutoFocusCallback);
                    mCamera.setOneShotPreviewCallback(mPreviewCallback);
                }
            }
        };
        timer = new Timer();
        LogUtils.w("interval: " + ConfigUtil.getInterval());
        timer.schedule(timerTask, 0, ConfigUtil.getInterval());
    }

    public void stopObjectDetect(){
        LogUtils.w("stop Object Detect");
        if(timer != null) {
            timer.cancel();
            timer.purge();
            timer = null;
        }
    }

    public Camera.Parameters getCameraParams() {
        if (mCamera != null) {
            mParams = mCamera.getParameters();
            return mParams;
        }
        return null;
    }

    public Camera getCameraDevice() {
        return mCamera;
    }

    public int getCameraId() {
        return mCameraId;
    }


    // 自动对焦回调
    AutoFocusCallback mAutoFocusCallback = new AutoFocusCallback() {

        public void onAutoFocus(boolean success, Camera camera) {
            // TODO Auto-generated method stub
            if (success)//success表示对焦成功
            {
                LogUtils.i("myAutoFocusCallback: succeed");
                mCamera.setOneShotPreviewCallback(mPreviewCallback);
            } else {
                //未对焦成功，如果聚焦失败就再次启动聚焦
                LogUtils.i("myAutoFocusCallback: failed");
                mCamera.autoFocus(mAutoFocusCallback);
            }
        }
    };


    //预览回调
    PreviewCallback mPreviewCallback = new PreviewCallback() {
        @Override
        public void onPreviewFrame(final byte[] data, Camera camera) {
            if (null != data && null != objectDetect) {
                objectDetect.post(data, previewWidth, previewHeight);
            }
        }
    };


    public void doTakePicture() {
        if (isPreviewing && (mCamera != null)) {
            mCamera.takePicture(mShutterCallback, null, mJpegPictureCallback);
        }
    }

    //快门按下的回调，在这里我们可以设置类似播放“咔嚓”声之类的操作。默认的就是咔嚓。
    ShutterCallback mShutterCallback = new ShutterCallback() {
        public void onShutter() {
            // TODO Auto-generated method stub
            LogUtils.i("myShutterCallback:onShutter...");
        }
    };

    // 拍摄的未压缩原数据的回调,可以为null
    PictureCallback mRawCallback = new PictureCallback() {

        public void onPictureTaken(byte[] data, Camera camera) {
            // TODO Auto-generated method stub
            LogUtils.i("myRawCallback:onPictureTaken...");
        }
    };

    //对jpeg图像数据的回调,最重要的一个回调
    PictureCallback mJpegPictureCallback = new PictureCallback() {
        public void onPictureTaken(byte[] data, Camera camera) {
            // TODO Auto-generated method stub
            LogUtils.i("myJpegCallback:onPictureTaken...");
            Bitmap b = null;
            if (null != data) {
                b = BitmapFactory.decodeByteArray(data, 0, data.length);//data是字节数据，将其解析成位图
                mCamera.stopPreview();
                isPreviewing = false;

                //设置FOCUS_MODE_CONTINUOUS_VIDEO)之后，myParam.set("rotation", 90)失效。
                //图片竟然不能旋转了，故这里要旋转下
                Bitmap rotateBitmap = ImageUtil.getRotateBitmap(b, 90.0f);
                FileUtil.saveBitmap(rotateBitmap);

                //再次进入预览
                mCamera.startPreview();
                isPreviewing = true;
            }

        }
    };

}
