package com.cmcm.util;

import android.support.v4.util.ArrayMap;

import com.apkfuns.logutils.LogUtils;

import org.json.JSONObject;

import java.io.IOException;
import java.util.Timer;
import java.util.TimerTask;
import java.util.concurrent.TimeUnit;

import okhttp3.Call;
import okhttp3.Callback;
import okhttp3.OkHttpClient;
import okhttp3.Request;
import okhttp3.Response;

/**
 * Created by zhangkai on 2017/8/22.
 */

public class ConfigUtil {
    private static ArrayMap<String, String> webSocketUrlMap = new ArrayMap<String, String>();
    private static ArrayMap<String, String> httpUrlMap = new ArrayMap<String, String>();
    private static OkHttpClient client = new OkHttpClient.Builder().connectTimeout(1, TimeUnit.SECONDS).build();
    private static Timer timer = null;

    private static String model = "seg512";
    private static int interval = 100;
    private static boolean intranet = true;
    private static int networkState = IntenetUtil.NETWORN_WIFI;
    private static final String INTRANET_URL = "http://10.60.242.201:5000";
    private static final String EXTRANET_URL = "http://ai.ishield.cn";
    private static String configUrl = null;
    private static String httpUrl = null;

    private static int scaleWidth = 300;
    private static int scaleHeight = 300;
    private static int minHeight = 1080;
    private static int jpegQuality = 90;
    private static int maxInterval = 3200;
    private static int configInterval = 5000;
    private static int delayTime = 30000;
    private static boolean mobileEnabled = true;


    static {
        webSocketUrlMap.put("det300", "ws://10.60.242.201:5000/websocket");

        httpUrlMap.put("det300", "/detect?model=det300");
        httpUrlMap.put("det600", "/detect?model=det600");
        httpUrlMap.put("mask600", "/detect?model=mask600");
        httpUrlMap.put("seg512", "/detect?model=seg512");
        setModel(model);
    }

    public static void setModel(String m) {
        model = m;
        httpUrl = null;
        if(model.equals("det300") || model.equals("mask300")){
            scaleWidth = 300;
            scaleHeight = 300;
        }else if(model.equals("det600") || model.equals("mask600")){
            scaleWidth = 600;
            scaleHeight = 600;
        }else{
            scaleWidth = 768;
            scaleHeight = 480;
        }
    }

    public static int getScaleWidth(){
        return scaleWidth;
    }
    public static int getScaleHeight(){
        return scaleHeight;
    }

    public static void setNetworkState(int i){
        networkState = i;
    }

    public static int getInterval() {
        return interval;
    }
    public static void setInterval(int i) {
        if(networkState != IntenetUtil.NETWORN_WIFI){
            interval = maxInterval;
        }else {
            interval = (i < maxInterval ? i : maxInterval);
        }
    }

    public static int getMinHeight() {
        return minHeight;
    }
    public static int getJpegQuality() {
        return jpegQuality;
    }
    public static int getDelayTime(){ return delayTime; }
    public static boolean getMobileEnabled(){ return mobileEnabled; }

    public static void setIntranet(boolean t) {
        intranet = t;
        httpUrl = null;
    }

    public static String getWebSocketUrl() {
        return webSocketUrlMap.get(model);
    }
    public static String getHttpUrl() {
        if (httpUrl == null) {
            if (intranet) httpUrl = INTRANET_URL + httpUrlMap.get(model);
            else httpUrl = EXTRANET_URL + httpUrlMap.get(model);
        }
        return httpUrl;
    }

    public static String getConfigUrl() {
        if (configUrl == null) {
            if (intranet) configUrl = INTRANET_URL + "/image";
            else configUrl = EXTRANET_URL + "/image";
        }
        return configUrl;
    }


    public static void getConfig() {
        final String url = getConfigUrl();
        Request request = new Request.Builder().url(url).build();
        Call call = client.newCall(request);

        call.enqueue(new Callback() {
            @Override
            public void onFailure(Call call, IOException e) {
                LogUtils.e("request failed: " + url);
                if(intranet){
                    setIntranet(false);
                    getConfig();
                }
            }

            @Override
            public void onResponse(Call call, Response response) {
                try {
                    JSONObject config = new JSONObject(response.body().string());
                    minHeight = config.optInt("min_height", 600);
                    minHeight = minHeight > 600 ? minHeight : 600;

                    jpegQuality = config.optInt("jpeg_quality", 30);
                    maxInterval = config.optInt("max_interval", 3200);
                    delayTime = config.optInt("delay_time", 3000);
                    mobileEnabled = config.optBoolean("mobile_enabled", true);
                } catch (Exception e) {
                    LogUtils.e(e);
                }
            }
        });
    }

    public static void updateConfig() {
        TimerTask timerTask = new TimerTask() {
            @Override
            public void run() {
                getConfig();
            }
        };
        timer = new Timer();
        timer.schedule(timerTask, 0, configInterval);
    }
}
