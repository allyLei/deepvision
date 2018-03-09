package com.cmcm.app;

import android.app.Application;

import com.apkfuns.logutils.LogLevel;
import com.apkfuns.logutils.LogUtils;
import com.cmcm.activity.CameraActivity.MainHandler;

import org.json.JSONObject;
import org.lzh.framework.updatepluginlib.UpdateConfig;
import org.lzh.framework.updatepluginlib.model.Update;
import org.lzh.framework.updatepluginlib.model.UpdateParser;

/**
 * Created by zhangkai on 2017/8/8.
 */

public class App extends Application {

    private MainHandler handler = null;

    public void setHandler(MainHandler handler){
        this.handler = handler;
    }

    public MainHandler getHandler(){
        return handler;
    }


    @Override
    public void onCreate() {
        super.onCreate();
        /*
        if (LeakCanary.isInAnalyzerProcess(this)) {
        }
        LeakCanary.install(this);

        ConfigUtil.updateConfig();
        */

        LogUtils.getLogConfig()
                .configLevel(LogLevel.TYPE_VERBOSE)
                .configTagPrefix("AILab")
                .configFormatTag("%d{HH:mm:ss:SSS} %c");

        UpdateConfig.getConfig()
                .url("http://update.ai.ishield.cn/image")
                .jsonParser(new UpdateParser() {
                    @Override
                    public Update parse(String response) {
                        try {
                            JSONObject object = new JSONObject(response);
                            Update update = new Update();
                            // 此apk包的更新时间
                            update.setUpdateTime(System.currentTimeMillis());
                            // 此apk包的下载地址
                            update.setUpdateUrl(object.optString("update_url"));
                            // 此apk包的版本号
                            update.setVersionCode(object.optInt("update_version_code"));
                            // 此apk包的版本名称
                            update.setVersionName(object.optString("update_version_name"));
                            // 此apk包的更新内容
                            update.setUpdateContent(object.optString("update_content"));
                            // 此apk包是否为强制更新
                            update.setForced(object.optBoolean("force", false));
                            // 是否显示忽略此次版本更新按钮
                            update.setIgnore(object.optBoolean("ignore_able", false));
                            return update;
                        } catch (Exception e) {
                            LogUtils.e(e);
                        }
                        return null;
                    }
                });
    }
}
