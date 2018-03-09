package com.cmcm.ui;

import android.content.Context;
import android.graphics.Bitmap;
import android.graphics.Canvas;
import android.graphics.Color;
import android.graphics.Paint;
import android.graphics.Paint.Style;
import android.graphics.Point;
import android.graphics.Rect;
import android.graphics.RectF;
import android.hardware.Camera;
import android.support.v7.widget.AppCompatImageView;
import android.util.AttributeSet;

import com.cmcm.camera.CameraInterface;
import com.cmcm.util.ConfigUtil;

import org.json.JSONArray;
import org.json.JSONObject;

import java.text.DecimalFormat;

import static com.cmcm.util.DisplayUtil.getScreenMetrics;

public class ObjectView extends AppCompatImageView {
    private Paint mLinePaint;
    private Paint mTextPaint;
    private Paint mMaskPaint;
    private int textSize = 40;
    private JSONObject result;
    private int detectTime;
    private Bitmap bmp1 = Bitmap.createBitmap(320, 512, Bitmap.Config.ARGB_8888);
    private int[] pixel = new int[320 * 512];
    Rect src = new Rect(0, 0, 320, 512);
    RectF dst;
    RectF dst2;

    public ObjectView(Context context, AttributeSet attrs) {
        super(context, attrs);
        initPaint();
        Point p = getScreenMetrics(context);
        dst = new RectF(0, 0, p.x, p.y);
        dst2 = new RectF(0, 0, p.x / 4, p.y / 4);
    }

    public void setRects(JSONObject result, int detectTime) {
        this.result = result;
        this.detectTime = detectTime;
        invalidate();
    }

    public void clearRects() {
        result = null;
        invalidate();
    }

    public static byte[] hexStrToByteArray(String str) {
        if (str == null) {
            return null;
        }
        if (str.length() == 0) {
            return new byte[0];
        }
        byte[] byteArray = new byte[str.length() / 2];
        for (int i = 0; i < byteArray.length; i++) {
            String subStr = str.substring(2 * i, 2 * i + 2);
            byteArray[i] = ((byte) Integer.parseInt(subStr, 16));
        }
        return byteArray;
    }

    private void drawMask(Canvas canvas, JSONObject segmentation) {
        int screenWidth = getWidth();
        int screenHeight = getHeight();

        JSONArray mask = segmentation.optJSONArray("mask");
        JSONObject colorMap = segmentation.optJSONObject("color");
        int width = segmentation.optInt("width");
        int height = segmentation.optInt("height");

        for (int i = 0; i < mask.length(); i++) {
            if (mask.optInt(i) != 0) {
                JSONArray color = colorMap.optJSONArray(String.valueOf(mask.optInt(i)));
                int maskColor = 0xb0000000 | (color.optInt(0) << 16) | (color.optInt(1) << 8) | color.optInt(2);
                pixel[i] = maskColor;
            }else{
                pixel[i] = 0;
            }
        }

        bmp1.setPixels(pixel, 0, width, 0, 0, width, height);
        canvas.drawBitmap(bmp1, src, dst,null);
        canvas.drawBitmap(bmp1, src, dst2,null);

        /*
        Bitmap bmp = Bitmap.createBitmap(width, height, Bitmap.Config.ARGB_8888);
        bmp.setPixels(pixel, 0, width, 0, 0, width, height);
        Bitmap newBmp = Bitmap.createScaledBitmap(bmp, screenWidth, screenHeight, true);
        bmp.recycle();

        Bitmap bmp2 = Bitmap.createBitmap(width, height, Bitmap.Config.ARGB_8888);
        for (int i = 0; i < mask.length(); i++) {
            if (mask.optInt(i) != 0) {
                JSONArray color = colorMap.optJSONArray(String.valueOf(mask.optInt(i)));
                int maskColor = 0xff000000 | (color.optInt(0) << 16) | (color.optInt(1) << 8) | color.optInt(2);
                pixel[i] = maskColor;
            }
        }
        bmp2.setPixels(pixel, 0, width, 0, 0, width, height);
        Bitmap newBmp2 = Bitmap.createScaledBitmap(bmp2, screenWidth / 4, screenHeight / 4, true);
        bmp2.recycle();

        canvas.drawBitmap(newBmp, 0, 0, mMaskPaint);
        canvas.drawBitmap(newBmp2, 0, 0, mMaskPaint);

        newBmp.recycle();
        newBmp2.recycle();
        */
    }

    private void drawObjects(Canvas canvas, JSONArray objects) {
        int screenWidth = getWidth();
        int screenHeight = getHeight();
        boolean isMirror = false;
        if (CameraInterface.getInstance().getCameraId() == Camera.CameraInfo.CAMERA_FACING_FRONT) {
            isMirror = true;
        }
        for (int i = 0; i < objects.length(); i++) {
            JSONObject object = objects.optJSONObject(i);
            JSONArray color = object.optJSONArray("color");
            mLinePaint.setARGB(255, color.optInt(0), color.optInt(1), color.optInt(2));
            mTextPaint.setARGB(255, color.optInt(0), color.optInt(1), color.optInt(2));
            mMaskPaint.setARGB(150, color.optInt(0), color.optInt(1), color.optInt(2));

            JSONArray rect = object.optJSONArray("rect");
            float x = (float) (rect.optDouble(0) * screenWidth);
            float y = (float) (rect.optDouble(1) * screenHeight);
            float xw = (float) (rect.optDouble(2) * screenWidth);
            float yh = (float) (rect.optDouble(3) * screenHeight);
            if (isMirror) {
                float t = screenWidth - xw;
                xw = screenWidth - x;
                x = t;
            }
            canvas.drawRect(new RectF(x, y, xw, yh), mLinePaint);

            String name = object.optString("name");
            canvas.drawText(name, x + textSize, (y - textSize < textSize ? textSize : y - textSize), mTextPaint);

            JSONArray mask = object.optJSONArray("mask");
            int maskColor = 0xf0000000 | (color.optInt(0) << 16) | (color.optInt(1) << 8) | color.optInt(2);
            if (mask != null) {
                int width = object.optInt("width");
                int height = object.optInt("height");
                int size = width * height;
                int[] pixel = new int[size];
                for (int j = 0; j < mask.length(); j++) {
                    int m = mask.optInt(j);
                    String bits = Integer.toBinaryString(m);
                    int l = 8 - bits.length();
                    for (int k = 0; k < 8; k++) {
                        int index = j * 8 + k;
                        if (index < size) {
                            if (k < l) {
                                pixel[index] = 0;
                            } else if (bits.charAt(k - l) == '1') {
                                pixel[index] = maskColor;
                            } else {
                                pixel[index] = 0;
                            }
                        }
                    }
                }
                Bitmap bmp = Bitmap.createBitmap(width, height, Bitmap.Config.ARGB_8888);
                bmp.setPixels(pixel, 0, width, 0, 0, width, height);
                Rect src = new Rect(0, 0, bmp.getWidth(), bmp.getHeight());
                RectF dst = new RectF(x, y, xw, yh);
                canvas.drawBitmap(bmp, src, dst, mMaskPaint);
                bmp.recycle();
            }
        }
    }

    private void drawPerson(Canvas canvas, JSONObject person) {
        int screenWidth = getWidth();
        mTextPaint.setColor(Color.BLUE);
        DecimalFormat df = new DecimalFormat("0.00");
        canvas.drawText("height: " + df.format(person.optDouble("height")), screenWidth - 400, textSize + 80, mTextPaint);
        canvas.drawText("distance: " + df.format(person.optDouble("distance")), screenWidth - 400, textSize + 140, mTextPaint);
        canvas.drawText("fat: " + df.format(person.optDouble("shape")), screenWidth - 400, textSize + 200, mTextPaint);
        if (person.optString("category") != null && person.optString("category") != "null") {
            canvas.drawText("category: " + person.optString("category"), screenWidth - 400, textSize + 260, mTextPaint);
        }
        if (person.optJSONArray("attrs") != null) {
            JSONArray attrs = person.optJSONArray("attrs");
            for (int i = 0; i < attrs.length(); i++) {
                canvas.drawText("attr: " + attrs.optString(i), screenWidth - 400, textSize + 320 + i * 60, mTextPaint);
            }
        }
    }

    @Override
    protected void onDraw(Canvas canvas) {
        int screenWidth = getWidth();
        //   canvas.scale(1,-1, getWidth()/2,getHeight()/2);
        canvas.save();

        mTextPaint.setColor(Color.BLUE);
        canvas.drawText(ConfigUtil.getInterval() + "ms", screenWidth - 400, textSize + 20, mTextPaint);
        canvas.drawText(detectTime + "ms", screenWidth - 200, textSize + 20, mTextPaint);

        if (result != null) {
            JSONObject segmentation = result.optJSONObject("segmentation");
            if (segmentation != null) {
                drawMask(canvas, segmentation);
            }
            JSONArray objects = result.optJSONArray("objects");
            if (objects != null) {
                drawObjects(canvas, objects);
            }
            JSONObject person = result.optJSONObject("person_info");
            if (person != null) {
                // drawPerson(canvas, person);
            }
        }
        canvas.restore();
        super.onDraw(canvas);
    }

    private void initPaint() {
        mLinePaint = new Paint(Paint.ANTI_ALIAS_FLAG);
        mLinePaint.setColor(Color.RED);
        mLinePaint.setStyle(Style.STROKE);
        mLinePaint.setStrokeWidth(8);
        // mLinePaint.setAlpha(180);

        mTextPaint = new Paint(Paint.ANTI_ALIAS_FLAG);
        mTextPaint.setStyle(Style.FILL);
        mTextPaint.setStrokeWidth(8);
        mTextPaint.setTextSize(textSize);
        // mTextPaint.setAlpha(180);

        mMaskPaint = new Paint(Paint.ANTI_ALIAS_FLAG);
        mMaskPaint.setStyle(Style.FILL);


    }
}
