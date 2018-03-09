package com.cmcm.util;

import android.hardware.Camera;
import android.hardware.Camera.Size;

import com.apkfuns.logutils.LogUtils;

import java.util.Collections;
import java.util.Comparator;
import java.util.List;

public class CamParaUtil {
    private CameraSizeComparator sizeComparator = new CameraSizeComparator();
    private static CamParaUtil myCamPara = null;

    private CamParaUtil() {

    }

    public static CamParaUtil getInstance() {
        if (myCamPara == null) {
            myCamPara = new CamParaUtil();
            return myCamPara;
        } else {
            return myCamPara;
        }
    }

    public Size getOptimalSize(List<Camera.Size> list, float th, int minHeight) {
        Collections.sort(list, sizeComparator);

        float minDiff = 1.0f;
        Size result = null;
        for (Size s : list) {
            if (s.height >= minHeight) {
                float diff = diffRate(s, th);
                if (diff < minDiff) {
                    LogUtils.v("width: " + s.width + " height: " + s.height + " diff: " + diff + " rate: " + th);
                    minDiff = diff;
                    result = s;
                }
            }
        }
        return result;
    }

    /* 和横竖屏有关 */
    public float diffRate(Size s, float rate) {
        float r = (float) (s.height) / (float) (s.width);
        return Math.abs(r - rate);
    }

    public class CameraSizeComparator implements Comparator<Camera.Size> {
        public int compare(Size lhs, Size rhs) {
            // TODO Auto-generated method stub
            if (lhs.width == rhs.width) {
                return 0;
            } else if (lhs.width > rhs.width) {
                return -1;
            } else {
                return 1;
            }
        }

    }

    /**
     * 打印支持的previewSizes
     *
     * @param params
     */
    public void printSupportPreviewSize(Camera.Parameters params) {
        List<Size> previewSizes = params.getSupportedPreviewSizes();
        for(int i=0; i< previewSizes.size(); i++){
			Size size = previewSizes.get(i);
			LogUtils.i("previewSizes:width = "+size.width+" height = "+size.height);
		}

    }

    /**
     * 打印支持的pictureSizes
     *
     * @param params
     */
    public void printSupportPictureSize(Camera.Parameters params) {
        List<Size> pictureSizes = params.getSupportedPictureSizes();
        for(int i=0; i< pictureSizes.size(); i++){
			Size size = pictureSizes.get(i);
			LogUtils.i("pictureSizes:width = "+ size.width +" height = " + size.height);
		}
    }

    /**
     * 打印支持的聚焦模式
     *
     * @param params
     */
    public void printSupportFocusMode(Camera.Parameters params) {
        List<String> focusModes = params.getSupportedFocusModes();
        LogUtils.v(focusModes);
        /*
		for(String mode : focusModes){
			LogUtils.i("focusModes--" + mode);
		}
		*/
    }
}
