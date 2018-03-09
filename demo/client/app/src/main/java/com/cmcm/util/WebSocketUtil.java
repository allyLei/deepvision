package com.cmcm.util;

import android.support.v4.util.ArrayMap;

import com.apkfuns.logutils.LogUtils;
import com.cmcm.mode.ObjectDetectListener;

import org.json.JSONObject;

import java.security.NoSuchAlgorithmException;

import okhttp3.OkHttpClient;
import okhttp3.Request;
import okhttp3.Response;
import okhttp3.WebSocket;
import okhttp3.WebSocketListener;
import okio.ByteString;

/**
 * Created by zhangkai on 2017/8/21.
 */

public final class WebSocketUtil extends WebSocketListener {
    private static final int NORMAL_CLOSURE_STATUS = 1000;
    private static WebSocket mWebSocket = null;
    private static WebSocketUtil mWebSocketUtil = null;
    private static ArrayMap<String, Long> md5Map = new ArrayMap<String, Long>();
    private ObjectDetectListener listener;

    public void setObjectListener(ObjectDetectListener listener) {
        this.listener = listener;
    }

    private void run() {
        String url = ConfigUtil.getWebSocketUrl();
        LogUtils.i("websocket url: " + url);
        OkHttpClient client = new OkHttpClient.Builder().build();
        Request request = new Request.Builder().url(url).build();
        mWebSocket = client.newWebSocket(request, this);
        client.dispatcher().executorService().shutdown();
    }

    public static synchronized WebSocketUtil getInstance() {
        if (mWebSocketUtil == null) {
            mWebSocketUtil = new WebSocketUtil();
            mWebSocketUtil.run();
        }
        return mWebSocketUtil;
    }

    @Override
    public void onOpen(WebSocket webSocket, Response response) {
        mWebSocket = webSocket;
    }

    @Override
    public void onMessage(WebSocket webSocket, String text) {
        try {
            JSONObject result = new JSONObject(text);
            if (result.optInt("err") == 0) {
                long ts = (long) System.currentTimeMillis();
                String md5 = result.getString("md5");
                long detectTime = ts - md5Map.get(md5);
                md5Map.remove(md5);
                LogUtils.v("md5: " + md5 + " detect time: " + detectTime);
                if(listener != null)  listener.onReceived(result, detectTime);
            }
        } catch (Exception e) {
            LogUtils.e(e);
        }
    }

    @Override
    public void onMessage(WebSocket webSocket, ByteString bytes) {
        LogUtils.v("Receiving bytes : " + bytes.hex());
    }

    @Override
    public void onClosed(WebSocket webSocket, int code, String reason) {
        LogUtils.w("Closed : " + code + ", reason: " + reason);
        mWebSocketUtil = null;
    }

    @Override
    public void onFailure(WebSocket webSocket, Throwable t, Response response) {
        LogUtils.v("Error : " + t.getMessage());
        mWebSocket.close(NORMAL_CLOSURE_STATUS, null);
        mWebSocketUtil = null;
    }

    public void startDetect(byte[] data, final long start_ts) {
        try {
            String md5 = MD5Util.getMD5(data);
            md5Map.put(md5, start_ts);
            mWebSocket.send(ByteString.of(data));
        } catch (NoSuchAlgorithmException e) {
            LogUtils.e(e);
        }
    }

    public void stop() {
        if(mWebSocketUtil != null) {
            mWebSocket.close(NORMAL_CLOSURE_STATUS, null);
            mWebSocketUtil = null;
        }
    }
}
