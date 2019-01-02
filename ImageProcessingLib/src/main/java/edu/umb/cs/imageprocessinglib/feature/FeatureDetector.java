package edu.umb.cs.imageprocessinglib.feature;

import edu.umb.cs.imageprocessinglib.model.DescriptorType;
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

import java.util.*;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

/**
 * Created by alex on 1/27/18.
 */

public class FeatureDetector {
    private static final int        kMaxFeatures = 500;

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
        orb = ORB.create(kMaxFeatures, 1.2f, 8, 15, 0, 2, ORB.HARRIS_SCORE, 31, 20);
    }

    public void extractORBFeatures(Mat img, MatOfKeyPoint keyPoints, Mat descriptors) {
        orb.detectAndCompute(img, new Mat(), keyPoints, descriptors);

    }

    public void extractSurfFeatures(Mat img, MatOfKeyPoint keyPoints, Mat descriptors) {
        surf.detectAndCompute(img, new Mat(), keyPoints, descriptors);

    }

    public void extractFeatures(Mat img, MatOfKeyPoint keyPoints, Mat descriptors, DescriptorType type) {
        switch (type) {
            case SURF:
                extractSurfFeatures(img, keyPoints, descriptors);
            case ORB:
            default:
                extractORBFeatures(img, keyPoints, descriptors);
        }
    }

    /**
     * extract feature points of images as well as its distorted images
     * @return  a list of list of integers, root list has the same size as original image features
     *          each sub-list, supposing its index is A_index, corresponds to each feature, supposing it's A_feature, in original image
     *          the integer in A_index sub-list stands for the index of distorted image which can find A_feature.
     */
    public ArrayList<ArrayList<Integer>> trackFeatures(
            Mat img,
            List<Mat> distortedImages,
            MatOfKeyPoint oriKPs,
            Mat oriDes,
            List<MatOfKeyPoint> distortedKPs,
            List<Mat> distortedDes,
            List<MatOfDMatch> distortedMatches,
            DescriptorType type) {

        //calculate original image's key points and descriptors
        extractFeatures(img, oriKPs, oriDes, type);

        //record the index of images to which the key point get matched
        ArrayList<ArrayList<Integer>> tracker = new ArrayList<>();
        for (int i = 0; i < oriDes.rows(); i++)
            tracker.add(new ArrayList<Integer>());

        //calculate key points and descriptors of distorted images
        for (int i = 0; i < distortedImages.size(); i++) {
            MatOfKeyPoint k = new MatOfKeyPoint();
            Mat d = new Mat();
            extractFeatures(distortedImages.get(i), k, d, type);
            distortedKPs.add(k);
            distortedDes.add(d);
        }

        //match key points of original image to distorted images'
        for (int i = 0; i < distortedImages.size(); i++) {
            MatOfDMatch m = FeatureMatcher.getInstance().matchFeature(distortedDes.get(i), oriDes, distortedKPs.get(i), oriKPs, type);
//            ArrayList<Integer> c = new ArrayList<>();

            //record the times that key point of original image is detected in distorted image
            List<DMatch> matches = m.toList();
            for (int d = 0; d < matches.size(); d++) {
                int index = matches.get(d).trainIdx;
                tracker.get(index).add(i);
            }
            distortedMatches.add(m);
        }
        return tracker;
    }

    /**
     * @param img
     * @param keyPoints
     * @param descriptors
     * @param type              the descriptor type
     * @param filterThreshold   remove feature points which has count lower than threshold
     * @param num               the number limit for returning key points
     * @return boolean indicates whether the method is done without problem
     */
    public boolean sortedRobustFeatures(Mat img, List<Mat> distortedImages, MatOfKeyPoint keyPoints, Mat descriptors, DescriptorType type, int filterThreshold, int num) {
        ArrayList<MatOfKeyPoint> listOfKeyPoints = new ArrayList<>();
        ArrayList<Mat> listOfDescriptors = new ArrayList<>();
        MatOfKeyPoint kp = new MatOfKeyPoint();
        Mat des = new Mat();
        ArrayList<MatOfDMatch> listOfMatches = new ArrayList<>();

        List<ArrayList<Integer>> tracker = trackFeatures(img, distortedImages, kp, des, listOfKeyPoints, listOfDescriptors, listOfMatches, type);

        List<KeyPoint> rKeyPoints = new ArrayList<>();     //store key points that will be return
        List<KeyPoint> tKeyPoints = kp.toList();

        //create a list containing keypoints list and counter list
        List<List<Object>> merged =
                IntStream.range(0, tracker.size())
                        .mapToObj(i -> Arrays.asList((Object) tKeyPoints.get(i), tracker.get(i).size()))
                        .collect(Collectors.toList());

        //descending order sort by counter
        merged.sort(new Comparator<List<Object>>() {
            @Override
            public int compare(List<Object> o1, List<Object> o2) {
                if ((Integer) o1.get(1) > (Integer)o2.get(1))
                    return -1;
                else if ((Integer) o1.get(1) < (Integer)o2.get(1))
                    return 1;
                else return 0;
            }
        });

        //remove feature points which appeared less than filterThreshold
        for (int i = 0; i < merged.size(); i++) {
            if ((Integer)merged.get(i).get(1) > filterThreshold) {
                rKeyPoints.add(tKeyPoints.get(i));
            }
        }
        if (rKeyPoints.size() > num)
            rKeyPoints = rKeyPoints.subList(0, num);

        keyPoints.fromList(rKeyPoints);
        if (type == DescriptorType.SURF)
            surf.compute(img, keyPoints, descriptors);
        if (type == DescriptorType.ORB)
            orb.compute(img, keyPoints, descriptors);

        //release resources before return
        for (int i = 0; i < distortedImages.size(); i++) {
            distortedImages.get(i).release();
            listOfDescriptors.get(i).release();
            listOfKeyPoints.get(i).release();
        }
        kp.release();
        des.release();

        return true;
    }

    private static int kRobustThreshold =   3;      //threshold deciding whether a feature point is robust to distortion

    public boolean extractRobustFeatures(Mat img, MatOfKeyPoint keyPoints, Mat descriptors, DescriptorType type, int num) {
        return sortedRobustFeatures(img, distortImage(img), keyPoints, descriptors, type, kRobustThreshold, num);
    }

    public boolean extractRobustFeatures(Mat img, MatOfKeyPoint keyPoints, Mat descriptors, DescriptorType type) {
        return sortedRobustFeatures(img, distortImage(img), keyPoints, descriptors, type, kRobustThreshold, kMaxFeatures);
    }

    public boolean extractRobustFeatures(Mat img, List<Mat> distortedImg, MatOfKeyPoint keyPoints, Mat descriptors, DescriptorType type, int num) {
        return sortedRobustFeatures(img, distortedImg, keyPoints, descriptors, type, kRobustThreshold, num);
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
            List<Point> corners = new ArrayList<>();
//            corners.add(new Point(image.cols()/5, image.rows()/5));
//            corners.add(new Point(image.cols(), image.rows()/5));
//            corners.add(new Point(image.cols()*3/4, image.rows()*3/4));
//            corners.add(new Point(image.cols()/5, image.rows()*3/4));
            //TODO: these points can be optimized
            corners.add(new Point(0, i * kStepPerspective * image.rows()));
            corners.add(new Point(image.cols(), i * kStepPerspective * image.rows()));
            corners.add(new Point(image.cols() * (1 - kStepPerspective * i), image.rows() * (1 - kStepPerspective * i)));
            corners.add(new Point(image.cols() * i * kStepPerspective, image.rows() * (1 - kStepPerspective * i)));

            r.add(ImageUtil.changeImagePerspective(image, corners, target));
            r.add(ImageUtil.changeImagePerspective(image, target, corners));

//            Mat cornersMat = Converters.vector_Point2f_to_Mat(corners);
//            Mat targetMat = Converters.vector_Point2f_to_Mat(target);
//            Mat trans = Imgproc.getPerspectiveTransform(cornersMat, targetMat);
//
//            Mat proj = new Mat();
//            Imgproc.warpPerspective(image, proj, trans, new Size(image.cols(), image.rows()));
//
//            Mat revertProj = new Mat();
//            trans.release();
//            trans = Imgproc.getPerspectiveTransform(targetMat, cornersMat);
//            Imgproc.warpPerspective(image, revertProj, trans, new Size(image.cols(), image.rows()));
//
//            r.add(proj);
//            r.add(revertProj);
//
//            //release resources
//            trans.release();
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

            r.add(ImageUtil.affineImage(image, original, targetA));
            r.add(ImageUtil.affineImage(image, targetA, original));

            //calculate the affine transformation matrix,
            //refer to https://stackoverflow.com/questions/22954239/given-three-points-compute-affine-transformation
//            Mat affineTransformA = Imgproc.getAffineTransform(originalMat, targetMatA);
//            Mat affineTransformB = Imgproc.getAffineTransform(targetMatA, originalMat);
//
//            Mat affineA = new Mat();
//            Mat affineB = new Mat();
//            Imgproc.warpAffine(image, affineA, affineTransformA, new Size(image.cols(), image.rows()));
//            Imgproc.warpAffine(image, affineB, affineTransformB, new Size(image.cols(), image.rows()));
//            r.add(affineA);
//            r.add(affineB);
//
//            //release resources
//            affineTransformA.release();
//            affineTransformB.release();
        }

        originalMat.release();

        return r;
    }


}
