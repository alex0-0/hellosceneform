package edu.umb.cs.imageprocessinglib;

import android.graphics.Bitmap;
import edu.umb.cs.imageprocessinglib.feature.FeatureDetector;
import edu.umb.cs.imageprocessinglib.feature.FeatureMatcher;
import edu.umb.cs.imageprocessinglib.model.ImageFeature;

import org.opencv.android.Utils;
import org.opencv.core.Mat;
import org.opencv.core.MatOfDMatch;
import org.opencv.core.MatOfKeyPoint;

public class ImageProcessor {
    static String TAG = "IMAGE_PROCESSOR";

    /*
    Extract image feature points
     */
    static public ImageFeature extractDistinctFeatures(Mat img) {
        MatOfKeyPoint kps = new MatOfKeyPoint();
        Mat des = new Mat();
        FeatureDetector.getInstance().extractDistinctFeatures(img, kps, des);
        return new ImageFeature(kps, des);
    }

    /*
    Extract image feature points
     */
    static public ImageFeature extractFeatures(Mat img) {
        MatOfKeyPoint kps = new MatOfKeyPoint();
        Mat des = new Mat();
        FeatureDetector.getInstance().extractFeatures(img, kps, des);
        return new ImageFeature(kps, des);
    }

    static public ImageFeature extractFeatures(Bitmap bitmap) {
        Mat img = new Mat();
        Utils.bitmapToMat(bitmap, img);
        return extractFeatures(img);
    }
    /*
    Extract image feature points with ORB detector, the bound of the number of feature points is num
     */
    static public ImageFeature extractORBFeatures(Mat img, int num) {
        MatOfKeyPoint kps = new MatOfKeyPoint();
        Mat des = new Mat();
        FeatureDetector fd = new FeatureDetector(num);
        fd.extractORBFeatures(img, kps, des);
        return new ImageFeature(kps, des);
    }

    /*
    Match two images
     */
    static public MatOfDMatch matcheImages(ImageFeature qIF, ImageFeature tIF) {
        return FeatureMatcher.getInstance().matchFeature(qIF.getDescriptors(), tIF.getDescriptors(), qIF.getObjectKeypoints(), tIF.getObjectKeypoints());
    }

    static public MatOfDMatch matcheImages(Mat queryImg, Mat temImg) {
        MatOfKeyPoint kps = new MatOfKeyPoint();
        ImageFeature qIF = extractFeatures(queryImg);
        ImageFeature tIF = extractFeatures(temImg);
        return matcheImages(qIF, tIF);
    }

    static public MatOfDMatch matcheImages(Bitmap queryImg, Bitmap temImg) {
        ImageFeature qIF = extractFeatures(queryImg);
        ImageFeature tIF = extractFeatures(temImg);
        return matcheImages(qIF, tIF);
    }
}
