package com.cmcm.activity;

import android.app.Activity;
import android.graphics.Point;
import android.hardware.Camera;
import android.hardware.Camera.Face;
import android.os.Bundle;
import android.os.Handler;
import android.os.Message;
import android.view.KeyEvent;
import android.view.Menu;
import android.view.View;
import android.view.View.OnClickListener;
import android.view.ViewGroup.LayoutParams;
import android.view.WindowManager;
import android.widget.ImageButton;
import android.widget.Toast;

import com.apkfuns.logutils.LogUtils;
import com.cmcm.camera.CameraInterface;
import com.cmcm.camera.CameraSurfaceView;
import com.cmcm.mode.FaceDetect;
import com.cmcm.mode.ObjectDetect;
import com.cmcm.playcamera.R;
import com.cmcm.ui.FaceView;
import com.cmcm.ui.ObjectView;
import com.cmcm.util.ConfigUtil;
import com.cmcm.util.DisplayUtil;
import com.cmcm.util.EventUtil;
import com.nex3z.togglebuttongroup.SingleSelectToggleGroup;

import org.json.JSONObject;
import org.lzh.framework.updatepluginlib.UpdateBuilder;

public class CameraActivity extends Activity {
    CameraSurfaceView surfaceView = null;
    ImageButton shutterBtn;
    ImageButton switchCameraBtn;
    FaceView faceView;
    ObjectView objectView;
    float previewRate = -1f;
    private MainHandler mMainHandler = null;
    private FaceDetect faceDetect = null;
    private ObjectDetect objectDetect = null;
    SingleSelectToggleGroup btnGroup = null;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);

        setContentView(R.layout.activity_camera);
        initUI();
        initViewParams();
        mMainHandler = new MainHandler();
        UpdateBuilder.create().check();

        getWindow().addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON);
    }

    @Override
    protected void onStart() {
        super.onStart();
        mMainHandler.sendEmptyMessageDelayed(EventUtil.CAMERA_HAS_STARTED_PREVIEW, 1000);
    }

    @Override
    protected void onStop() {
        super.onStop();
        LogUtils.w("onStop called");
        stopObjectDetect();
        finish();
        System.exit(0);
    }

    @Override
    public boolean onKeyDown(int keyCode, KeyEvent event) {
        LogUtils.w("key code: " + keyCode);
        //捕获返回键按下的事件
        if(keyCode == KeyEvent.KEYCODE_BACK || keyCode == KeyEvent.KEYCODE_HOME){
            onStop();
            return true;
        }
        return false;
    }


    @Override
    public boolean onCreateOptionsMenu(Menu menu) {
        // Inflate the menu; this adds items to the action bar if it is present.
        getMenuInflater().inflate(R.menu.camera, menu);
        return true;
    }

    private void initUI() {
        surfaceView = (CameraSurfaceView) findViewById(R.id.camera_surfaceview);
        shutterBtn = (ImageButton) findViewById(R.id.btn_shutter);
        switchCameraBtn = (ImageButton) findViewById(R.id.btn_switch_camera);
        btnGroup = (SingleSelectToggleGroup) findViewById(R.id.group_choices);
        faceView = (FaceView) findViewById(R.id.face_view);
        objectView = (ObjectView) findViewById(R.id.object_view);

        shutterBtn.setOnClickListener(new BtnListeners());
        switchCameraBtn.setOnClickListener(new BtnListeners());

        btnGroup.setOnCheckedChangeListener(new SingleSelectToggleGroup.OnCheckedChangeListener() {
            @Override
            public void onCheckedChanged(SingleSelectToggleGroup group, int checkedId) {
                switch (checkedId){
                    case R.id.det300:
                        ConfigUtil.setModel("det300");
                        break;
                    case R.id.det600:
                        ConfigUtil.setModel("det600");
                        break;
                    case R.id.mask600:
                        ConfigUtil.setModel("mask600");
                        break;
                    case R.id.seg512:
                        ConfigUtil.setModel("seg512");
                        break;
                }
            }
        });
    }

    private void initViewParams() {
        LayoutParams params = surfaceView.getLayoutParams();
        Point p = DisplayUtil.getScreenMetrics(this);
        params.width = p.x;
        params.height = p.y;
        previewRate = DisplayUtil.getScreenRate(this); //默认全屏的比例预览
        surfaceView.setLayoutParams(params);

        LogUtils.i("surface view width = " + params.width + "; height = " + params.height);
    }

    private class BtnListeners implements OnClickListener {

        @Override
        public void onClick(View v) {
            switch (v.getId()) {
                case R.id.btn_shutter:
                    takePicture();
                    break;
                case R.id.btn_switch_camera:
                    switchCamera();
                    break;
                default:
                    break;
            }
        }
    }





    private void takePicture() {
        CameraInterface.getInstance().doTakePicture();
        mMainHandler.sendEmptyMessageDelayed(EventUtil.CAMERA_HAS_STARTED_PREVIEW, 1000);
    }

    private void switchCamera() {
        stopObjectDetect();
        int newId = (CameraInterface.getInstance().getCameraId() + 1) % 2;
        CameraInterface.getInstance().doStopCamera();
        CameraInterface.getInstance().doOpenCamera(null, newId);
        CameraInterface.getInstance().doStartPreview(surfaceView, previewRate);
        mMainHandler.sendEmptyMessageDelayed(EventUtil.CAMERA_HAS_STARTED_PREVIEW, 1000);
    }

    private void startObjectDetect() {
        LogUtils.i("new object detect in activity");
        objectDetect = new ObjectDetect(getApplicationContext(), mMainHandler);
        CameraInterface.getInstance().setObjectDetectListener(objectDetect);
        CameraInterface.getInstance().startObjectDetect();
    }

    private void stopObjectDetect() {
        CameraInterface.getInstance().setObjectDetectListener(null);
        CameraInterface.getInstance().stopObjectDetect();
        objectView.clearRects();
    }

    private void startFaceDetect() {
        faceDetect = new FaceDetect(getApplicationContext(), mMainHandler);
        Camera.Parameters params = CameraInterface.getInstance().getCameraParams();
        if (params.getMaxNumDetectedFaces() > 0) {
            if (faceView != null) {
                faceView.clearRects();
                faceView.setVisibility(View.VISIBLE);
            }
            CameraInterface.getInstance().getCameraDevice().setFaceDetectionListener(faceDetect);
            CameraInterface.getInstance().getCameraDevice().startFaceDetection();
        }
    }

    private void stopFaceDetect() {
        Camera.Parameters params = CameraInterface.getInstance().getCameraParams();
        if (params.getMaxNumDetectedFaces() > 0) {
            CameraInterface.getInstance().getCameraDevice().setFaceDetectionListener(null);
            CameraInterface.getInstance().getCameraDevice().stopFaceDetection();
            faceView.clearRects();
        }
    }

    public final class MainHandler extends Handler {

        @Override
        public void handleMessage(Message msg) {
            switch (msg.what) {
                case EventUtil.UPDATE_FACE_RECT:
                    Face[] faces = (Face[]) msg.obj;
                    faceView.setRects(faces);
                    break;
                case EventUtil.UPDATE_OBJECT_RECT:
                    JSONObject result = (JSONObject) msg.obj;
                    int detectTime = (int) msg.arg1;
                    objectView.setRects(result, detectTime);
                    break;
                case EventUtil.CAMERA_HAS_STARTED_PREVIEW:
                    startObjectDetect();
                    break;
                case EventUtil.NETWORK_NOT_WIFI:
                    String message = "当前非WIFI网络，已停止检测";
                    if (ConfigUtil.getMobileEnabled()) {
                        message = "当前非WIFI网络，检测帧率调整为最大值";
                    }
                    Toast.makeText(getApplicationContext(), message, Toast.LENGTH_LONG).show();
                    break;
            }
            super.handleMessage(msg);
        }
    }

}
