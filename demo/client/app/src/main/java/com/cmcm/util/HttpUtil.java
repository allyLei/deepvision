package com.cmcm.util;

import com.apkfuns.logutils.LogUtils;
import com.cmcm.mode.ObjectDetectListener;

import org.json.JSONObject;

import java.io.ByteArrayInputStream;
import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.util.concurrent.TimeUnit;
import java.util.zip.GZIPInputStream;
import java.util.zip.GZIPOutputStream;

import okhttp3.Call;
import okhttp3.Callback;
import okhttp3.MediaType;
import okhttp3.OkHttpClient;
import okhttp3.Request;
import okhttp3.RequestBody;
import okhttp3.Response;


/**
 * Created by zhangkai on 2017/8/9.
 */


public class HttpUtil {
    private static ObjectDetectListener listener = null;
    private static final MediaType CONTENT_TYPE = MediaType.parse("image/jpeg");
    private static OkHttpClient client = new OkHttpClient.Builder()
            .connectTimeout(5, TimeUnit.SECONDS)
            .readTimeout(5, TimeUnit.SECONDS)
            .build();


    public static void setObjectListener(ObjectDetectListener l) {
        listener = l;
    }

    public static byte[] compress(byte[] data) throws IOException {
        if (null == data || data.length <= 0) {
            return data;
        }
        // 创建一个新的 byte 数组输出流
        ByteArrayOutputStream out = new ByteArrayOutputStream();
        // 使用默认缓冲区大小创建新的输出流
        GZIPOutputStream gzip = new GZIPOutputStream(out);
        // 将 b.length 个字节写入此输出流
        gzip.write(data);
        gzip.close();
        // 使用指定的 charsetName，通过解码字节将缓冲区内容转换为字符串
        return out.toByteArray();
    }

    public static byte[] decompress(byte[] data) throws IOException {
        ByteArrayOutputStream out = new ByteArrayOutputStream();
        ByteArrayInputStream in = new ByteArrayInputStream(data);
        GZIPInputStream gzip = new GZIPInputStream(in);
        byte[] buffer = new byte[1024];
        int n = 0;
        while ((n = gzip.read(buffer, 0, buffer.length)) > 0) {
            out.write(buffer, 0, n);
        }
        gzip.close();
        in.close();
        out.flush();
        return out.toByteArray();
    }

    public static void startDetect(byte[] data, final long start_ts, final int width, final int height) {
        final String url = ConfigUtil.getHttpUrl() + "&width=" + width + "&height=" + height + "&rotate=true";
        RequestBody body = RequestBody.create(CONTENT_TYPE, data);
        Request request = new Request.Builder().url(url).post(body).build();

        Call call = client.newCall(request);

        call.enqueue(new Callback() {
            @Override
            public void onFailure(Call call, IOException e) {
                LogUtils.e("request failed: " + url);
            }

            @Override
            public void onResponse(Call call, Response response) {
                try {
                    byte[] data = decompress(response.body().bytes());
                    JSONObject result = new JSONObject(new String(data));
                    if (result.optInt("err") == 0) {
                        long ts = System.currentTimeMillis();
                        long detectTime = ts - start_ts;
                        if (listener != null) {
                            listener.onReceived(result, detectTime);
                        }
                    }
                } catch (Exception e) {
                    LogUtils.e(e);
                }
            }
        });
    }
}
