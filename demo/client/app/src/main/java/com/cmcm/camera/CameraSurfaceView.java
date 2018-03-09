package com.cmcm.camera;

import android.content.Context;
import android.graphics.PixelFormat;
import android.util.AttributeSet;
import android.view.SurfaceHolder;
import android.view.SurfaceView;

import com.apkfuns.logutils.LogUtils;
import com.cmcm.util.DisplayUtil;

public class CameraSurfaceView extends SurfaceView implements SurfaceHolder.Callback {
    Context mContext;
    SurfaceHolder mSurfaceHolder;

    public CameraSurfaceView(Context context, AttributeSet attrs) {
        super(context, attrs);
        mContext = context;
        mSurfaceHolder = getHolder();
        mSurfaceHolder.setFormat(PixelFormat.TRANSPARENT);  //translucent半透明 transparent透明
        //	mSurfaceHolder.setType(SurfaceHolder.SURFACE_TYPE_PUSH_BUFFERS);
        mSurfaceHolder.addCallback(this);
    }

    @Override
    public void surfaceCreated(SurfaceHolder holder) {
        // TODO Auto-generated method stub
        LogUtils.i("surfaceCreated...");

        //    Canvas canvas = holder.lockCanvas();
        //   canvas.scale(1,-1, getWidth()/2,getHeight()/2);
        CameraInterface.getInstance().doOpenCamera(null, android.hardware.Camera.CameraInfo.CAMERA_FACING_BACK);
                // Camera.CameraInfo.CAMERA_FACING_BACK);
        //   holder.unlockCanvasAndPost(canvas);
    }

    @Override
    public void surfaceChanged(SurfaceHolder holder, int format, int width, int height) {
        // TODO Auto-generated method stub
        LogUtils.i("surfaceChanged...");
        float previewRate = DisplayUtil.getScreenRate(mContext);
        CameraInterface.getInstance().doStartPreview(this, previewRate);
    }

    @Override
    public void surfaceDestroyed(SurfaceHolder holder) {
        // TODO Auto-generated method stub
        LogUtils.i("surfaceDestroyed...");
        CameraInterface.getInstance().doStopCamera();

    }

}
