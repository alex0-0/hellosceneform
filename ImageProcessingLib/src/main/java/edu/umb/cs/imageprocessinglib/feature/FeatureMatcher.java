package edu.umb.cs.imageprocessinglib.feature;

import org.opencv.calib3d.Calib3d;
import org.opencv.core.DMatch;
import org.opencv.core.KeyPoint;
import org.opencv.core.Mat;
import org.opencv.core.MatOfDMatch;
import org.opencv.core.MatOfKeyPoint;
import org.opencv.core.MatOfPoint2f;
import org.opencv.core.Point;
import org.opencv.features2d.BFMatcher;
import org.opencv.features2d.DescriptorMatcher;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashSet;
import java.util.Iterator;
import java.util.LinkedList;
import java.util.List;

/**
 * Created by alex on 1/28/18.
 */

public class FeatureMatcher {
    public enum DescriptorType {
        ORB,
        SURF
    }
    //    static private final double kConfidence = 0.99;
//    static private final double kDistance = 3.0;
    static private final double kRatio = 0.7;
    static double kTolerableDifference = 0.1;  //an custom number to determine whether two matches have spacial relation
    static String TAG = "Feature Matcher";

    private DescriptorMatcher matcher;
    private DescriptorType descriptorType;

    private static final FeatureMatcher ourInstance = new FeatureMatcher(DescriptorType.ORB);

    public static FeatureMatcher getInstance() {
        return ourInstance;
    }

    public FeatureMatcher(DescriptorType type) {
        descriptorType = type;
        matcher = createMatcher(type);
    }

    //Default matcher is ORB
    private BFMatcher createMatcher(DescriptorType type) {
        BFMatcher m;
        switch (type) {
            case SURF:
                m = BFMatcher.create(DescriptorMatcher.BRUTEFORCE_SL2, false);
                break;
            case ORB:
            default:
                m = BFMatcher.create(DescriptorMatcher.BRUTEFORCE_HAMMING, false);
                break;
        }
        return m;
    }

    public MatOfDMatch BFMatchFeature(Mat queryDescriptor, Mat templateDescriptor) {
        return BFMatchFeature(queryDescriptor, templateDescriptor, descriptorType);
    }

    /**
     *
     * @param queryDescriptor
     * @param templateDescriptor
     * @param dType                 Descriptor type
     * @return                      matched key points
     */
    public MatOfDMatch BFMatchFeature(Mat queryDescriptor, Mat templateDescriptor, DescriptorType dType) {
        DescriptorMatcher m = matcher;
        if (dType != descriptorType) {
            m = createMatcher(dType);
        }
        MatOfDMatch matches = new MatOfDMatch();
        m.match(queryDescriptor, templateDescriptor, matches);
        return matches;
    }

    public MatOfDMatch matchFeature(Mat queryDescriptor, Mat templateDescriptor, MatOfKeyPoint queryKeyPoints, MatOfKeyPoint templateKeyPoints) {
        return matchFeature(queryDescriptor, templateDescriptor, queryKeyPoints, templateKeyPoints, descriptorType);
    }
    /**
     *
     * @param queryDescriptor
     * @param templateDescriptor
     * @param queryKeyPoints            query key points
     * @param templateKeyPoints            template key points
     * @return                      matched key points
     */
    public MatOfDMatch matchFeature(Mat queryDescriptor,
                                    Mat templateDescriptor,
                                    MatOfKeyPoint queryKeyPoints,
                                    MatOfKeyPoint templateKeyPoints,
                                    DescriptorType dType) {
//        MatOfDMatch matches = new MatOfDMatch();
//        matcher.match(queryDescriptor, templateDescriptor, matches);       //k(final parameter) set to 1 will do crosscheck
//        return matches;
        DescriptorMatcher m = matcher;

        if (dType != descriptorType) {
            m = createMatcher(dType);
        }

        ArrayList<MatOfDMatch> matches1 = new ArrayList<>();
        ArrayList<MatOfDMatch>  matches2 = new ArrayList<>();

        m.knnMatch(queryDescriptor, templateDescriptor, matches1, 2);       //k(final parameter) set to 1 will do crosscheck
        m.knnMatch(templateDescriptor, queryDescriptor, matches2, 2);

        ratioTest(matches1);
        ratioTest(matches2);

        MatOfDMatch symMatches = symmetryTest(matches1, matches2);
        MatOfDMatch ransacMatches = new MatOfDMatch();

        //release resources
        for (int i = 0; i < matches1.size(); i++) {
            matches1.get(i).release();
        }
        for (int i = 0; i < matches2.size(); i++) {
            matches2.get(i).release();
        }

        if (symMatches.total() > 20) {
            ransacTest(symMatches, queryKeyPoints, templateKeyPoints, ransacMatches);
            //release resources
            symMatches.release();
            return ransacMatches;
        }
        return symMatches;
    }

    //if the two best matches are relatively close in distance,
//then there exists a possibility that we make an error if we select one or the other.
    private int ratioTest(ArrayList<MatOfDMatch> matches) {
        ArrayList<MatOfDMatch> updatedMatches = new ArrayList<>();
        int removed=0;
        // for all matches
        for (int i = 0; i < matches.size(); i++) {
            MatOfDMatch matchIterator = matches.get(i);
            // if 2 NN has been identified
            if (matchIterator.total() > 1) {
                DMatch[] match = matchIterator.toArray();
                // check distance ratio
                if (match[0].distance/match[1].distance <= kRatio) {
                    updatedMatches.add(matchIterator);
                    continue;
                }
            }
            // does not have 2 neighbours or distance ratio is higher than threshold
            removed++;
        }

        //assign filtered value to matches
        matches = updatedMatches;

        return removed;
    }

    private MatOfDMatch symmetryTest(ArrayList<MatOfDMatch> matches1, ArrayList<MatOfDMatch> matches2) {

        ArrayList<DMatch> symMatchList = new ArrayList<>();

        for (int i = 0; i < matches1.size(); i++) {
            MatOfDMatch matchIterator1 = matches1.get(i);
            if (matchIterator1.total() < 2) {
                continue;
            }
            DMatch match1 = matchIterator1.toArray()[0];

            for (int d = 0; d < matches2.size(); d++) {
                MatOfDMatch matchIterator2 = matches2.get(d);
                if (matchIterator2.total() < 2)
                    continue;
                DMatch match2 = matchIterator2.toArray()[0];
                // Match symmetry test
                if (match1.queryIdx == match2.trainIdx && match2.queryIdx == match1.trainIdx) {
                    // add symmetrical match
                    symMatchList.add(match1);
                    break; // next match in image 1 -> image 2
                }
            }
        }

        MatOfDMatch symMatches = new MatOfDMatch();
        symMatches.fromList(symMatchList);
        return symMatches;
    }

    //refer to: https://en.wikipedia.org/wiki/Random_sample_consensus
// Identify good matches using RANSAC
    private void ransacTest(MatOfDMatch matches,
                            MatOfKeyPoint keypoints1,
                            MatOfKeyPoint keypoints2,
                            MatOfDMatch outMatches)
    {
        // get keypoint coordinates of good matches to find homography and remove outliers using ransac
        List<Point> pts1 = new ArrayList<>();
        List<Point> pts2 = new ArrayList<>();
        LinkedList<DMatch> good_matches = new LinkedList<>(Arrays.asList(matches.toArray()));
        for(int i = 0; i<good_matches.size(); i++){
            pts1.add(keypoints1.toList().get(good_matches.get(i).queryIdx).pt);
            pts2.add(keypoints2.toList().get(good_matches.get(i).trainIdx).pt);
        }

        // convertion of data types - there is maybe a more beautiful way
        Mat outputMask = new Mat();
        MatOfPoint2f pts1Mat = new MatOfPoint2f();
        pts1Mat.fromList(pts1);
        MatOfPoint2f pts2Mat = new MatOfPoint2f();
        pts2Mat.fromList(pts2);

        // Find homography - here just used to perform match filtering with RANSAC, but could be used to e.g. stitch images
        // the smaller the allowed reprojection error (here 15), the more matches are filtered
        Mat Homog = Calib3d.findHomography(pts1Mat, pts2Mat, Calib3d.RANSAC, 15, outputMask, 2000, 0.995);

        // outputMask contains zeros and ones indicating which matches are filtered
        LinkedList<DMatch> better_matches = new LinkedList<DMatch>();
        for (int i = 0; i < good_matches.size(); i++) {
            if (outputMask.get(i, 0)[0] != 0.0) {
                better_matches.add(good_matches.get(i));
            }
        }
        outMatches.fromList(better_matches);
        //release resource
        outputMask.release();
        pts1Mat.release();
        pts2Mat.release();
        Homog.release();
    }

    /**
     * By clustering matched key points in query image and template image respectively,
     * if a spacial pattern of key points existed in both template and query image,
     * return a bonus by ((the number of points in that pattern)^2 / (the number of key points in template image))
     * @param matches  matched points
     * @param qKPs      query key points
     * @param tKPs      template key points
     * @return  bonus confidence coming from clustering matched key points
     **/
    public double bonusConfidenceFromClusteringMatchedPoints(MatOfDMatch matches, MatOfKeyPoint qKPs, MatOfKeyPoint tKPs) {
        double bonus = 0;
        bonus = greedyClustering(matches, qKPs, tKPs);
        return bonus;
    }

    /**
     * From the first element of matches to the end, cluster as much points to present point as possible
     * @param matches  matched points
     * @param qKPs      query key points
     * @param tKPs      template key points
     * @return          bonus points
     */
    double greedyClustering(MatOfDMatch matches, MatOfKeyPoint qKPs, MatOfKeyPoint tKPs) {
        double bonus = 0;
        KeyPoint q[] = qKPs.toArray();
        KeyPoint t[] = tKPs.toArray();
        LinkedList<DMatch> matchesList = new LinkedList<>(matches.toList());
        Iterator<DMatch> i1 = matchesList.iterator();
        while (i1.hasNext()) {
            HashSet<DMatch> cluster = new HashSet<>();
            cluster.add(i1.next());
            i1.remove();

            //iterate from present point to the end of the list to check if any point matches have similar spacial relation
            if (i1.hasNext()) {
                Iterator<DMatch> i2 = matchesList.iterator();
                while (i2.hasNext()) {
                    DMatch m = i2.next();
                    if (hasSpatialRelation(cluster, m, q, t)) {
                        cluster.add(m);
                        i2.remove();
                    }
                }
                int size = cluster.size();
                bonus += (size > 1)? ((float)size * size / tKPs.total()) : 0;
                //since the matches may be changed
                i1 = matchesList.iterator();
            }
        }
        return bonus;
    }

    //any match in set has relation with query match, return true
    //qKPs is query key points, tKPs is template key points
    boolean hasSpatialRelation(HashSet<DMatch> matches, DMatch queryMatch, KeyPoint[] qKPs, KeyPoint[] tKPs) {
        Point qPoint = qKPs[queryMatch.queryIdx].pt;
        Point tPoint = tKPs[queryMatch.trainIdx].pt;

        Iterator<DMatch> i = matches.iterator();
        while (i.hasNext()) {
            DMatch m = i.next();
            Point mQPoint = qKPs[m.queryIdx].pt;
            Point mTPoint = tKPs[m.trainIdx].pt;

            //compare in template image and query image the ratio of vertical differences and horizontal difference
            double ratioInQueryImage = (qPoint.y - mQPoint.y) / (qPoint.x - mQPoint.x);
            double ratioInTemplateImage = (tPoint.y - mTPoint.y) / (tPoint.x - mTPoint.x);

            if (Math.abs(ratioInQueryImage - ratioInTemplateImage) > kTolerableDifference)
                return false;
        }
        //if the match looks good to existed points in cluster
        return true;
    }

}
