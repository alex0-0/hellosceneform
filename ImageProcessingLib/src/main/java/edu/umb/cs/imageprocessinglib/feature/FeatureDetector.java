package edu.umb.cs.imageprocessinglib.feature;

import edu.umb.cs.imageprocessinglib.util.ImageUtil;

import org.opencv.core.DMatch;
import org.opencv.core.KeyPoint;
import org.opencv.core.Mat;
import org.opencv.core.MatOfDMatch;
import org.opencv.core.MatOfKeyPoint;
import org.opencv.core.MatOfPoint2f;
import org.opencv.core.Point;
import org.opencv.core.Size;
import org.opencv.features2d.FastFeatureDetector;
import org.opencv.features2d.ORB;
import org.opencv.imgproc.Imgproc;
import org.opencv.utils.Converters;
import org.opencv.xfeatures2d.SURF;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

/**
 * Created by alex on 1/27/18.
 */
public class FeatureDetector {
    private static final int        kMaxFeatures = 200;

    private FastFeatureDetector     FAST;
    private SURF surf;
    private ORB orb;

    private static final FeatureDetector ourInstance = new FeatureDetector();

    public static FeatureDetector getInstance() {
        return ourInstance;
    }

    //init ORB detector with specific limit on feature number,
    // this constructor is only used for ORB detector, since no SURF detector is initialized
    public FeatureDetector(int feautureNum) {
        orb = ORB.create(feautureNum, 1.2f, 8, 15, 0, 2, ORB.HARRIS_SCORE, 31, 20);
    }

    private FeatureDetector() {
        FAST = FastFeatureDetector.create();
        surf = SURF.create();
        surf.setHessianThreshold(400);
        orb = ORB.create(500, 1.2f, 8, 15, 0, 2, ORB.HARRIS_SCORE, 31, 20);
    }

    public void extractORBFeatures(Mat img, MatOfKeyPoint keyPoints, Mat descriptors) {
        orb.detectAndCompute(img, new Mat(), keyPoints, descriptors);

    }

    public void extractSurfFeatures(Mat img, MatOfKeyPoint keyPoints, Mat descriptors) {
        surf.detectAndCompute(img, new Mat(), keyPoints, descriptors);

    }
    public void extractFeatures(Mat img, MatOfKeyPoint keyPoints, Mat descriptors) {
        extractORBFeatures(img, keyPoints, descriptors);
    }

    private static int kDistinctThreshold    =   3;      //threshold deciding whether a feature point is robust to distortion

    public boolean extractDistinctFeatures(Mat img, MatOfKeyPoint keyPoints, Mat descriptors) {
        ArrayList<Mat> distortedImages = distortImage(img);
        ArrayList<MatOfKeyPoint> ListOfKeyPoints = new ArrayList<>();
        ArrayList<Mat> ListOfDescriptors = new ArrayList<>();
        MatOfKeyPoint kp = new MatOfKeyPoint();
        Mat des = new Mat();

        //calculate original image's key points and descriptors
        extractFeatures(img, kp, des);

        //record the number of images to which the key point get matched
        ArrayList<Integer> counter = new ArrayList<>(Collections.nCopies((int)kp.total(), 0));

        //calculate key points and descriptors of distorted images
        for (int i = 0; i < distortedImages.size(); i++) {
            MatOfKeyPoint k = new MatOfKeyPoint();
            Mat d = new Mat();
            extractFeatures(distortedImages.get(i), k, d);
            ListOfKeyPoints.add(k);
            ListOfDescriptors.add(d);
        }

        //compare key points of original image to distorted images'
        for (int i = 0; i < distortedImages.size(); i++) {
            MatOfDMatch m = FeatureMatcher.getInstance().matchFeature(ListOfDescriptors.get(i), des, ListOfKeyPoints.get(i), kp);

            //record the times that key point of original image is detected in distorted image
            List<DMatch> matches = m.toList();
            for (int d = 0; d < matches.size(); d++) {
                int index = matches.get(d).trainIdx;
                int count = counter.get(index);
                count++;
                counter.set(index, count);
            }
        }

        ArrayList<KeyPoint> rKeyPoints = new ArrayList<>();     //store key points that will be return
        List<KeyPoint> tKeyPoints = kp.toList();
        for (int i = 0; i < kp.total(); i++) {
            if (counter.get(i) > kDistinctThreshold) {
                rKeyPoints.add(tKeyPoints.get(i));
            }
        }
        keyPoints.fromList(rKeyPoints);
        surf.compute(img, keyPoints, descriptors);

        //release resources before return
        for (int i = 0; i < distortedImages.size(); i++) {
            distortedImages.get(i).release();
            ListOfDescriptors.get(i).release();
            ListOfKeyPoints.get(i).release();
        }
        kp.release();
        des.release();

        return true;
    }

    /**
     * Get a group of distorted images by applying transformation on original image
     * For now only scale and rotation is applying on the image
     * @param image
     * @return          a group of distorted images
     */
    public ArrayList<Mat> distortImage(Mat image) {
        ArrayList<Mat> r = new ArrayList<>();
        r.addAll(scaleImage(image));
        r.addAll(rotateImage(image));
        r.addAll(changeImagePerspective(image));
        r.addAll(affineImage(image));

        return r;
    }

    private static final float kStepScale = 0.1f;        //the difference between scales of generating distorted images
    private static final int kNumOfScales = 6;    //the number of different scale distorted images

    /**
     * Scale original image to generate a group of distorted image
     * @param image     original image
     * @return          a list containing scaled images
     */
    private ArrayList<Mat> scaleImage(Mat image) {
        ArrayList<Mat> r = new ArrayList<>();
        for (int i = 1; i <= kNumOfScales /2; i++) {
            r.add(ImageUtil.scaleImage(image, (1 + i * kStepScale)));
            r.add(ImageUtil.scaleImage(image, (1 - i * kStepScale)));
        }

        return r;
    }

    private static final float kStepAngle = 5.0f;        //the step difference between angles of generating distorted images, in degree.
    private static final int kNumOfRotations = 6;    //the number of different rotated distorted images

    /**
     * Rotate original image to generate a group of distorted image
     * @param image     original image
     * @return          a list containing rotated images
     */
    private ArrayList<Mat> rotateImage(Mat image) {
        ArrayList<Mat> r = new ArrayList<>();
        for (int i = 1; i <= kNumOfRotations /2; i++) {
            r.add(ImageUtil.rotateImage(image, -kStepAngle * i));
            r.add(ImageUtil.rotateImage(image, kStepAngle * i));
        }

        return r;
    }

    static double kStepPerspective = 0.1;
    static int kNumOfPerspectives = 4;

    /**
     * Change original image's view perspective to generate a group of distorted image
     * @param image     original image
     * @return          a list containing rotated images
     */
    private ArrayList<Mat> changeImagePerspective(Mat image) {
        ArrayList<Mat> r = new ArrayList<>();
        List<Point> target = new ArrayList<Point>();

        //TODO: these points can be optimized
        target.add(new Point(0, 0));
        target.add(new Point(image.cols(), 0));
        target.add(new Point(image.cols(), image.rows()));
        target.add(new Point(0, image.rows()));

        for (int i = 0; i < kNumOfPerspectives/2; i++) {
            List<Point> corners = new ArrayList<Point>();
//            corners.add(new Point(image.cols()/5, image.rows()/5));
//            corners.add(new Point(image.cols(), image.rows()/5));
//            corners.add(new Point(image.cols()*3/4, image.rows()*3/4));
//            corners.add(new Point(image.cols()/5, image.rows()*3/4));
            //TODO: these points can be optimized
            corners.add(new Point(0, i * kStepPerspective * image.rows()));
            corners.add(new Point(image.cols(), i * kStepPerspective * image.rows()));
            corners.add(new Point(image.cols() * (1 - kStepPerspective * i), image.rows() * (1 - kStepPerspective * i)));
            corners.add(new Point(image.cols() * i * kStepPerspective, image.rows() * (1 - kStepPerspective * i)));

            Mat cornersMat = Converters.vector_Point2f_to_Mat(corners);
            Mat targetMat = Converters.vector_Point2f_to_Mat(target);
            Mat trans = Imgproc.getPerspectiveTransform(cornersMat, targetMat);

            Mat proj = new Mat();
            Imgproc.warpPerspective(image, proj, trans, new Size(image.cols(), image.rows()));

            Mat revertProj = new Mat();
            trans.release();
            trans = Imgproc.getPerspectiveTransform(targetMat, cornersMat);
            Imgproc.warpPerspective(image, revertProj, trans, new Size(image.cols(), image.rows()));

            r.add(proj);
            r.add(revertProj);

            //release resources
            trans.release();
        }

        return r;
    }

    static int kStepAffine = 5;
    static int kNumOfAffines = 4;

    /**
     * Affine original image to generate a group of distorted image
     * @param image     original image
     * @return          a list containing rotated images
     */
    private ArrayList<Mat> affineImage(Mat image) {
        ArrayList<Mat> r = new ArrayList<>();
        List<Point> original = new ArrayList<>();

        //TODO: this is just a random number given without specific reason, can be optimized if possible
        original.add(new Point(10, 10));
        original.add(new Point(200,50));
        original.add(new Point(50, 200));

        MatOfPoint2f originalMat = new MatOfPoint2f();
        originalMat.fromList(original);

        for (int i = 0; i < kNumOfAffines/2; i++) {
            List<Point> targetA = new ArrayList<>();
            targetA.add(new Point(50 + i * kStepAffine, 100 + i * kStepAffine));
            targetA.add(new Point(200 + i * kStepAffine, 50 + i * kStepAffine));
            targetA.add(new Point(100 + i * kStepAffine, 250 + i * kStepAffine));

            MatOfPoint2f targetMatA = new MatOfPoint2f();
            targetMatA.fromList(targetA);

            //calculate the affine transformation matrix,
            //refer to https://stackoverflow.com/questions/22954239/given-three-points-compute-affine-transformation
            Mat affineTransformA = Imgproc.getAffineTransform(originalMat, targetMatA);
            Mat affineTransformB = Imgproc.getAffineTransform(targetMatA, originalMat);

            Mat affineA = new Mat();
            Mat affineB = new Mat();
            Imgproc.warpAffine(image, affineA, affineTransformA, new Size(image.cols(), image.rows()));
            Imgproc.warpAffine(image, affineB, affineTransformB, new Size(image.cols(), image.rows()));
            r.add(affineA);
            r.add(affineB);

            //release resources
            affineTransformA.release();
            affineTransformB.release();
        }

        originalMat.release();

        return r;
    }
}


