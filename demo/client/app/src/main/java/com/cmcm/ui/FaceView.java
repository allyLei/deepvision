package com.cmcm.ui;

import android.content.Context;
import android.graphics.Canvas;
import android.graphics.Color;
import android.graphics.Matrix;
import android.graphics.Paint;
import android.graphics.Paint.Style;
import android.graphics.Rect;
import android.graphics.RectF;
import android.graphics.drawable.Drawable;
import android.hardware.Camera.CameraInfo;
import android.hardware.Camera.Face;
import android.support.v7.widget.AppCompatImageView;
import android.util.AttributeSet;

import com.apkfuns.logutils.LogUtils;
import com.cmcm.camera.CameraInterface;
import com.cmcm.playcamera.R;

public class FaceView extends AppCompatImageView {
    private Context mContext;
    private Paint mLinePaint;
    private Rect[] mRects;
    private Matrix mMatrix = new Matrix();
    private RectF mRect = new RectF();
    private Drawable mFaceIndicator = null;

    public FaceView(Context context, AttributeSet attrs) {
        super(context, attrs);
        initPaint();
        mContext = context;
        mFaceIndicator = getResources().getDrawable(R.drawable.ic_face_find_2);
    }

    public void setRects(Face[] faces) {
        mRects = new Rect[faces.length];
        for (int i = 0; i < faces.length; i++) {
            mRects[i] = faces[i].rect;
        }
        invalidate();
    }

    public void clearRects() {
        mRects = null;
        invalidate();
    }

    public static void faceMatrix(Matrix matrix, boolean mirror, int displayOrientation,
                                  int viewWidth, int viewHeight) {
        // Need mirror for front camera.
        matrix.setScale(mirror ? -1 : 1, 1);
        // This is the value for android.hardware.Camera.setDisplayOrientation.
        matrix.postRotate(displayOrientation);
        // Camera driver coordinates range from (-1000, -1000) to (1000, 1000).
        // UI coordinates range from (0, 0) to (width, height).
        matrix.postScale(viewWidth / 2000f, viewHeight / 2000f);
        matrix.postTranslate(viewWidth / 2f, viewHeight / 2f);
    }

    @Override
    protected void onDraw(Canvas canvas) {
        // TODO Auto-generated method stub
        if (mRects == null || mRects.length < 1) {
            return;
        }

        boolean isMirror = false;
        int Id = CameraInterface.getInstance().getCameraId();
        if (Id == CameraInfo.CAMERA_FACING_BACK) {
            isMirror = false; //后置Camera无需mirror
        } else if (Id == CameraInfo.CAMERA_FACING_FRONT) {
            isMirror = true;  //前置Camera需要mirror
        }

        faceMatrix(mMatrix, isMirror, 90, getWidth(), getHeight());

        int width = canvas.getWidth();
        int height = canvas.getHeight();

        LogUtils.v("canvas width: " + width + " height: " + height);

        canvas.save();
//		mMatrix.postRotate(0); //Matrix.postRotate默认是顺时针
//		canvas.rotate(-0);   //Canvas.rotate()默认是逆时针

        for (int i = 0; i < mRects.length; i++) {
            mRect.set(mRects[i]);
            mMatrix.mapRect(mRect);
//            mFaceIndicator.setBounds(Math.round(mRect.left), Math.round(mRect.top),
 //                   Math.round(mRect.right), Math.round(mRect.bottom));
 //           mFaceIndicator.draw(canvas);
			canvas.drawRect(mRect, mLinePaint);
        }
        canvas.restore();
        super.onDraw(canvas);
    }

    private void initPaint() {
        mLinePaint = new Paint(Paint.ANTI_ALIAS_FLAG);
//      mLinePaint.setColor(Color.rgb(98, 212, 68));
        mLinePaint.setColor(Color.RED);
        mLinePaint.setStyle(Style.STROKE);    //空心矩形
//        mLinePaint.setStyle(Style.FILL);       //实心矩形
        mLinePaint.setStrokeWidth(5f);
        mLinePaint.setAlpha(180);
    }
}
