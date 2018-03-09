package com.cmcm.mode;

import org.json.JSONObject;

/**
 * Created by zhangkai on 2017/8/24.
 */

public interface ObjectDetectListener {
    public void onReceived(JSONObject result, long detectTime);
}
