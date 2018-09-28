/*
 * Copyright 2018 Google LLC. All Rights Reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
package com.google.ar.sceneform.samples.hellosceneform;

import android.app.Activity;
import android.app.ActivityManager;
import android.content.Context;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.graphics.Canvas;
import android.graphics.Color;
import android.graphics.ImageFormat;
import android.graphics.Matrix;
import android.graphics.Paint;
import android.graphics.Point;
import android.graphics.Rect;
import android.graphics.RectF;
import android.graphics.Typeface;
import android.graphics.YuvImage;
import android.media.Image;
import android.os.Build;
import android.os.Build.VERSION_CODES;
import android.os.Bundle;
import android.os.Handler;
import android.os.HandlerThread;
import android.os.SystemClock;
import android.support.v7.app.AppCompatActivity;
import android.util.DisplayMetrics;
import android.util.Log;
import android.util.Size;
import android.util.TypedValue;
import android.view.Display;
import android.view.Gravity;
import android.view.MotionEvent;
import android.view.PixelCopy;
import android.view.Surface;
import android.view.View;
import android.view.ViewGroup;
import android.widget.ImageView;
import android.widget.LinearLayout;
import android.widget.Toast;
import com.google.ar.core.Anchor;
import com.google.ar.core.Camera;
import com.google.ar.core.Config;
import com.google.ar.core.Frame;
import com.google.ar.core.HitResult;
import com.google.ar.core.Plane;
import com.google.ar.core.Pose;
import com.google.ar.core.Session;
import com.google.ar.core.exceptions.CameraNotAvailableException;
import com.google.ar.sceneform.AnchorNode;
import com.google.ar.sceneform.ArSceneView;
import com.google.ar.sceneform.HitTestResult;
import com.google.ar.sceneform.math.Vector3;
import com.google.ar.sceneform.rendering.ModelRenderable;
import com.google.ar.sceneform.rendering.PlaneRenderer;
import com.google.ar.sceneform.rendering.Renderer;
import com.google.ar.sceneform.samples.hellosceneform.env.BorderedText;
import com.google.ar.sceneform.samples.hellosceneform.env.ImageUtils;
import com.google.ar.sceneform.samples.hellosceneform.env.Logger;
import com.google.ar.sceneform.samples.hellosceneform.tracking.MultiBoxTracker;
import com.google.ar.sceneform.ux.ArFragment;
import com.google.ar.sceneform.ux.TransformableNode;

import org.opencv.android.BaseLoaderCallback;
import org.opencv.android.LoaderCallbackInterface;
import org.opencv.android.OpenCVLoader;
import org.opencv.android.Utils;
import org.opencv.core.Mat;

import java.io.IOException;
import java.nio.ByteBuffer;
import java.util.LinkedList;
import java.util.List;

/**
 * This is an example activity that uses the Sceneform UX package to make common AR tasks easier.
 */
public class HelloSceneformActivity extends AppCompatActivity {
    private  static final String TAG = "RECTANGLE_DEBUG";
//    private static final String TAG = HelloSceneformActivity.class.getSimpleName();
    private static final double MIN_OPENGL_VERSION = 3.1;

    private MyArFragment arFragment;
    private ModelRenderable andyRenderable;

    private Session arSession;//added by bo

    private float last_chk_time=0;
    private boolean opencvLoaded=false;
    private Classifier classifier;

    private OverlayView trackingOverlay;
    /*** from tensorflow sample code***/
    private Handler handler;
    private long timestamp = 0; //it's actually a counter
    private Bitmap cropCopyBitmap = null;
    private Bitmap croppedBitmap = null;
    private Bitmap rgbFrameBitmap=null;

    private HandlerThread handlerThread;
    private byte[][] yuvBytes = new byte[3][];
    private int[] rgbBytes = null;
    private int yRowStride;

    protected int previewWidth = 0;
    protected int previewHeight = 0;
    private ImageView imgView;

    private static final Logger LOGGER = new Logger();

    // Configuration values for the prepackaged multibox model.
    private static final int MB_INPUT_SIZE = 224;
    private static final int MB_IMAGE_MEAN = 128;
    private static final float MB_IMAGE_STD = 128;
    private static final String MB_INPUT_NAME = "ResizeBilinear";
    private static final String MB_OUTPUT_LOCATIONS_NAME = "output_locations/Reshape";
    private static final String MB_OUTPUT_SCORES_NAME = "output_scores/Reshape";
    private static final String MB_MODEL_FILE = "file:///android_asset/multibox_model.pb";
    private static final String MB_LOCATION_FILE = "file:///android_asset/multibox_location_priors.txt";

    private static final int TF_OD_API_INPUT_SIZE = 300;
    private static final String TF_OD_API_MODEL_FILE = "file:///android_asset/ssd_mobilenet_v1_android_export.pb";
    private static final String TF_OD_API_LABELS_FILE = "file:///android_asset/coco_labels_list.txt";

    // Configuration values for tiny-yolo-voc. Note that the graph is not included with TensorFlow and
    // must be manually placed in the assets/ directory by the user.
    // Graphs and models downloaded from http://pjreddie.com/darknet/yolo/ may be converted e.g. via
    // DarkFlow (https://github.com/thtrieu/darkflow). Sample command:
    // ./flow --model cfg/tiny-yolo-voc.cfg --load bin/tiny-yolo-voc.weights --savepb --verbalise
    private static final String YOLO_MODEL_FILE = "file:///android_asset/graph-tiny-yolo-voc.pb";
    private static final int YOLO_INPUT_SIZE = 416;
    private static final String YOLO_INPUT_NAME = "input";
    private static final String YOLO_OUTPUT_NAMES = "output";
    private static final int YOLO_BLOCK_SIZE = 32;

    // Which detection model to use: by default uses Tensorflow Object Detection API frozen
    // checkpoints.  Optionally use legacy Multibox (trained using an older version of the API)
    // or YOLO.
    private enum DetectorMode {
        TF_OD_API, MULTIBOX, YOLO;
    }
    private static final DetectorMode MODE = DetectorMode.TF_OD_API;

    // Minimum detection confidence to track a detection.
    private static final float MINIMUM_CONFIDENCE_TF_OD_API = 0.6f;
    private static final float MINIMUM_CONFIDENCE_MULTIBOX = 0.1f;
    private static final float MINIMUM_CONFIDENCE_YOLO = 0.25f;

    private static final boolean MAINTAIN_ASPECT = MODE == DetectorMode.YOLO;

    private static final Size DESIRED_PREVIEW_SIZE = new Size(640, 480);

    private static final boolean SAVE_PREVIEW_BITMAP = false;
    private static final float TEXT_SIZE_DIP = 10;

    private Integer sensorOrientation;
    private Matrix frameToCropTransform;
    private Matrix cropToFrameTransform;
    private Matrix frameToDisplayTransform;
    private int rotation=90;

    private MultiBoxTracker tracker;

    private byte[] luminanceCopy;

    private Classifier detector;
    private BorderedText borderedText;

    static Boolean once_token = false;

//    private final PlaneRenderer planeRenderer = new PlaneRenderer(new Renderer(new ));

    @Override
    @SuppressWarnings({"AndroidApiChecker", "FutureReturnValueIgnored"})
    // CompletableFuture requires api level 24
    // FutureReturnValueIgnored is not valid
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);

        if (!checkIsSupportedDeviceOrFinish(this)) {
            return;
        }

        setContentView(R.layout.activity_ux);
        arFragment = (MyArFragment) getSupportFragmentManager().findFragmentById(R.id.ux_fragment);
        arFragment.getPlaneDiscoveryController().hide();
        arFragment.getPlaneDiscoveryController().setInstructionView(null);
        imgView = findViewById(R.id.imgview);


        //ImageView imgview=findViewById(R.id.imgview);

        //imgview.setImageResource(R.drawable.ic_launcher);

        arFragment.setActivity(this);//add by bo
        try {
            Session session = new Session(this);
            Config config = new Config(session);
            config.setFocusMode(Config.FocusMode.AUTO);
            session.configure(config);
        } catch (Exception e) {

        }

//        arSession = arFragment.getArSceneView().getSession();
//        Config config = arSession.getConfig();
//        config.setFocusMode(Config.FocusMode.AUTO);
//        arSession.configure(config);
        arFragment.setOnFrameListener((frameTime, frame) -> {
            float curTime=frameTime.getStartSeconds();
            Bitmap bitmap=null;//Bitmap.createBitmap(previewWidth, previewHeight, Bitmap.Config.ARGB_8888);
            Image img=null;
            //if(curTime-last_chk_time<2) return;
            if(frame==null) {Log.d("myTag","frame is null"); return;}
            else Log.d("myTag","frame is not null");
            try {
                Log.d("myTag","before acquire camera image");
                img = frame.acquireCameraImage(); //catch the image from camera
                String msg = img.getFormat()+":"+Integer.toString(img.getWidth())+","+Integer.toString(img.getHeight());
                Log.d("myTag", msg);
                //setImage(img);

                //TODO: should the luminanceCopy get scaled just as image does
                //TODO: is the conversion done in correct way?
                luminanceCopy = MyUtils.imageToByte(img); //convert image to byte[]
                bitmap=imageToBitmap(img);

                img.close();
                //if(bitmap!=null) setImage(bitmap);
                //else return;
            }catch(Exception e){System.out.println("myTag"+e); return;}

            if(detector==null) initTF(bitmap);

            processImage(bitmap);

        });



        // When you build a Renderable, Sceneform loads its resources in the background while returning
        // a CompletableFuture. Call thenAccept(), handle(), or check isDone() before calling get().
        ModelRenderable.builder()
                .setSource(this, R.raw.andy)
                .build()
                .thenAccept(renderable -> andyRenderable = renderable)
                .exceptionally(
                        throwable -> {
                            Toast toast =
                                    Toast.makeText(this, "Unable to load andy renderable", Toast.LENGTH_LONG);
                            toast.setGravity(Gravity.CENTER, 0, 0);
                            toast.show();
                            return null;
                        });

        arFragment.setOnTapArPlaneListener(
                (HitResult hitResult, Plane plane, MotionEvent motionEvent) -> {
                    if (andyRenderable == null) {
                        return;
                    }

                    Log.d("myTag","panel listener");

                    float x = motionEvent.getRawX();
                    float y = motionEvent.getRawY();
                    float x1 = motionEvent.getX();
                    float y1 = motionEvent.getY();
                    float x2 = motionEvent.getXPrecision();
                    float y2 = motionEvent.getYPrecision();

                    // Create the Anchor.
                    Anchor anchor = hitResult.createAnchor();
                    AnchorNode anchorNode = new AnchorNode(anchor);
                    anchorNode.setParent(arFragment.getArSceneView().getScene());
                    float[] xs = anchor.getPose().getXAxis();
                    float[] ys = anchor.getPose().getYAxis();
                    float[] zs = anchor.getPose().getZAxis();
                    Vector3 localPosition = anchorNode.getLocalPosition();
                    Vector3 worldPosition = anchorNode.getWorldPosition();

                    // Create the transformable andy and add it to the anchor.
                    TransformableNode andy = new TransformableNode(arFragment.getTransformationSystem());
                    andy.setParent(anchorNode);
                    andy.setRenderable(andyRenderable);
                    andy.select();

//                    float[] mAnchorMatrix = new float[16];
//                    anchor.getPose().toMatrix(mAnchorMatrix, 0);
//                    objectRenderer.updateModelMatrix(mAnchorMatrix, 1);
//                    objectRenderer.draw(cameraView, cameraPerspective, lightIntensity);
//
//                    float[] centerVertexOf3dObject = {0f, 0f, 0f, 1};
//                    float[] vertexResult = new float[4];
//                    Matrix.multiplyMV(vertexResult, 0,
//                            objectRenderer.getModelViewProjectionMatrix(), 0,
//                            centerVertexOf3dObject, 0);
//// circle hit test
//                    float radius = (viewWidth / 2) * (cubeHitAreaRadius/vertexResult[3]);
//                    float dx = event.getX() - (viewWidth / 2) * (1 + vertexResult[0]/vertexResult[3]);
//                    float dy = event.getY() - (viewHeight / 2) * (1 - vertexResult[1]/vertexResult[3]);
//                    double distance = Math.sqrt(dx * dx + dy * dy);
                });

        //added by bo

    }

    void initTF(Bitmap bitmap){
        previewWidth = bitmap.getWidth();
        previewHeight = bitmap.getHeight();

        //last_chk_time=curTime;
        /*** from detector activity in tensorflow sample code***/
        final float textSizePx =
                TypedValue.applyDimension(
                        TypedValue.COMPLEX_UNIT_DIP, TEXT_SIZE_DIP, getResources().getDisplayMetrics());
        borderedText = new BorderedText(textSizePx);
        borderedText.setTypeface(Typeface.MONOSPACE);

        //if(tracker==null)
        tracker = new MultiBoxTracker(this);


        int cropSize = TF_OD_API_INPUT_SIZE;
        if (MODE == DetectorMode.YOLO) {
            detector =
                    TensorFlowYoloDetector.create(
                            getAssets(),
                            YOLO_MODEL_FILE,
                            YOLO_INPUT_SIZE,
                            YOLO_INPUT_NAME,
                            YOLO_OUTPUT_NAMES,
                            YOLO_BLOCK_SIZE);
            cropSize = YOLO_INPUT_SIZE;
        } else if (MODE == DetectorMode.MULTIBOX) {
            detector =
                    TensorFlowMultiBoxDetector.create(
                            getAssets(),
                            MB_MODEL_FILE,
                            MB_LOCATION_FILE,
                            MB_IMAGE_MEAN,
                            MB_IMAGE_STD,
                            MB_INPUT_NAME,
                            MB_OUTPUT_LOCATIONS_NAME,
                            MB_OUTPUT_SCORES_NAME);
            cropSize = MB_INPUT_SIZE;
        } else {
            try {
                detector = TensorFlowObjectDetectionAPIModel.create(
                        getAssets(), TF_OD_API_MODEL_FILE, TF_OD_API_LABELS_FILE, TF_OD_API_INPUT_SIZE);
                cropSize = TF_OD_API_INPUT_SIZE;
            } catch (final IOException e) {
                LOGGER.e("Exception initializing classifier!", e);
                Toast toast =
                        Toast.makeText(
                                getApplicationContext(), "Classifier could not be initialized", Toast.LENGTH_SHORT);
                toast.show();
                finish();
            }
        }




        sensorOrientation = rotation - getScreenOrientation();
        /*
        LOGGER.i("Camera orientation relative to screen canvas: %d", sensorOrientation);

        LOGGER.i("Initializing at size %dx%d", previewWidth, previewHeight);
        */
        rgbFrameBitmap = Bitmap.createBitmap(previewWidth, previewHeight, Bitmap.Config.ARGB_8888);

        croppedBitmap = Bitmap.createBitmap(cropSize, cropSize, Bitmap.Config.ARGB_8888);

        frameToCropTransform =
                ImageUtils.getTransformationMatrix(
                        previewWidth, previewHeight,
                        cropSize, cropSize,
                        sensorOrientation, MAINTAIN_ASPECT);
        cropToFrameTransform = new Matrix();
        frameToCropTransform.invert(cropToFrameTransform);

        /**/
        float h = arFragment.getView().getHeight();//1944
        float w = arFragment.getView().getWidth();//1080
        trackingOverlay = (OverlayView) findViewById(R.id.tracking_overlay);
        float dpHeight = 1005;//trackingOverlay.getHeight() / 2;
        float dpWidth = 540;//trackingOverlay.getWidth() / 2;
        //75 is the height of image view
//        frameToDisplayTransform =
//                ImageUtils.getTransformationMatrix(
//                        previewWidth, previewHeight,
//                        (int)dpHeight, (int)dpWidth,
//                        0, MAINTAIN_ASPECT);
        //track object in AR fragment view
        trackingOverlay.addCallback(
                new OverlayView.DrawCallback() {
                    @Override
                    public void drawCallback(final Canvas canvas) {
                        tracker.draw(canvas);
//                        tracker.drawDebug(canvas);
                    }
                });
    }

    void processImage(Bitmap bitmap){
        if(bitmap==null) return;

        //byte[] originalLuminance = getLuminance();

        ++timestamp;
        final long currTimestamp = timestamp;

        if (luminanceCopy == null) {
            //luminanceCopy = new byte[originalLuminance.length];
            Log.d("myTag","luminanceCopy is null");
            return;
        }

        tracker.onFrame(
                previewWidth,
                previewHeight,
                previewWidth, //stride is the same as previewWidth
                sensorOrientation,
                luminanceCopy,
                timestamp);
//        trackingOverlay.postInvalidate();
        //System.arraycopy(originalLuminance, 0, luminanceCopy, 0, originalLuminance.length);

        final Canvas canvas = new Canvas(croppedBitmap);
        canvas.drawBitmap(bitmap, frameToCropTransform, null);

        setImage(croppedBitmap);
        runInBackground(
                new Runnable() {
                    @Override
                    public void run() {
                        final long startTime = SystemClock.uptimeMillis();

                        Log.d("myTag","before recognizeimage");
                        final List<Classifier.Recognition> results = detector.recognizeImage(croppedBitmap);

                        cropCopyBitmap = Bitmap.createBitmap(croppedBitmap);
//                        final Canvas canvas = new Canvas(cropCopyBitmap);
//                        final Paint paint = new Paint();
//                        paint.setColor(Color.RED);
//                        paint.setStyle(Paint.Style.STROKE);
//                        paint.setStrokeWidth(2.0f);

                        float minimumConfidence = MINIMUM_CONFIDENCE_TF_OD_API;
                        switch (MODE) {
                            case TF_OD_API:
                                minimumConfidence = MINIMUM_CONFIDENCE_TF_OD_API;
                                break;
                            case MULTIBOX:
                                minimumConfidence = MINIMUM_CONFIDENCE_MULTIBOX;
                                break;
                            case YOLO:
                                minimumConfidence = MINIMUM_CONFIDENCE_YOLO;
                                break;
                        }

                        final List<Classifier.Recognition> mappedRecognitions =
                                new LinkedList<Classifier.Recognition>();
                        String str=String.format("results:%d",results.size());

                        DisplayMetrics displayMetrics = new DisplayMetrics();
                        getWindowManager().getDefaultDisplay().getMetrics(displayMetrics);
//                        Display display = getWindowManager().getDefaultDisplay();
//                        Point size = new Point();
//                        display.getSize(size);
                        int dpi = displayMetrics.densityDpi;
                        int height = displayMetrics.heightPixels;//2016
                        int width = displayMetrics.widthPixels;//1080
//                        float density = getResources().getDisplayMetrics().density;

                        if (!once_token) {
                            once_token = true;
////                            trackingOverlay = (OverlayView) findViewById(R.id.tracking_overlay);
//                        float dHeight = trackingOverlay.getHeight();
//                        float dWidth = trackingOverlay.getWidth();
//                        float hhh = imgView.getHeight();
//                            float hhb = croppedBitmap.getHeight();
//                            double dpHeight = (dHeight - hhh)  * 0.5;
////                            double dpHeight = 2016  * 0.5;
////                                    (trackingOverlay.getHeight() - imgView.getHeight()) * (imgView.getHeight()/cropCopyBitmap.getHeight());
//                            double dpWidth = dWidth * 0.5;
////                                    trackingOverlay.getWidth() * (imgView.getHeight()/cropCopyBitmap.getHeight());
//                            //75 is the height of image view
//                            frameToDisplayTransform =
//                                    ImageUtils.getTransformationMatrix(
//                                            previewWidth,previewHeight,
//                                            (int)640,(int)480,
//                                            0, false);

//                            runOnUiThread(new Runnable() {
//                                @Override
//                                public void run() {
//                                    ViewGroup.LayoutParams tp = trackingOverlay.getLayoutParams();
////                                    tp.height = height;
////                                    tp.width = width;
//                                    tp.height = previewWidth * 2;
//                                    tp.width = previewHeight * 2;
//                                    trackingOverlay.setLayoutParams(tp);
//                                    ViewGroup.LayoutParams rp = arFragment.getArSceneView().getLayoutParams();
////                                    rp.height = height;
////                                    rp.width = width;
//                                    rp.height = previewWidth * 2;
//                                    rp.width = previewHeight * 2;
//                                    arFragment.getArSceneView().setLayoutParams(rp);
//                                    ViewGroup.LayoutParams rv = arFragment.getView().getLayoutParams();
////                                    rv.height = height;
////                                    rv.width = width;
//                                    rv.height = previewWidth * 2;
//                                    rv.width = previewHeight * 2;
//                                    arFragment.getView().setLayoutParams(rv);
//                                }
//                            });
                        }

                        org.opencv.core.Rect roi = new org.opencv.core.Rect();

                        Log.d("myTag",str);
                        for (final Classifier.Recognition result : results) {
                            final RectF location = result.getLocation();
                            if (location != null && result.getConfidence() >= minimumConfidence) {

                                roi = new org.opencv.core.Rect((int)location.left, (int)location.top, (int)(location.right - location.left), (int)(location.bottom - location.top));
                                Log.d(TAG, "main: rect before transform " + location.toString());
                                cropToFrameTransform.mapRect(location);
                                Log.d(TAG, "main: rect after transform " + location.toString());
//                                frameToDisplayTransform.mapRect(location);
                                //TODO: the width is a bit narrower than supposed to be, figure out later what is going wrong
//                                float hRatio = height / previewWidth / 2.0f;
//                                float wRatio = width / previewHeight / 2.0f;
//                                RectF loc = new RectF(location.left*hRatio, location.top*wRatio, location.right*hRatio, location.bottom*wRatio);
//                                result.setLocation(loc);
                                result.setLocation(location);
                                mappedRecognitions.add(result);
                            }
                        }

                        if (mappedRecognitions.size() > 0) {
                            Mat mat = new Mat();
                            Utils.bitmapToMat(cropCopyBitmap, mat);
                            Mat cropMat = new Mat(mat, roi);
                            Bitmap tBM = Bitmap.createBitmap(roi.width, roi.height, Bitmap.Config.ARGB_8888);
                            Utils.matToBitmap(cropMat, tBM);
                            mat.release();
                            cropMat.release();
//                            setImage(tBM);
                        }

//                        RectF rectF = new RectF(9, 79, 283, 216);//previewWidth, previewHeight);
//                        cropToFrameTransform.mapRect(rectF);
                        RectF rectF = new RectF(0, 0, previewWidth, previewHeight);
                        Classifier.Recognition result = new Classifier.Recognition("1434", "test", 0.99f, rectF);
//                        mappedRecognitions.add(result);

//                        str=String.format("mapped:%d, cropped image size(%d, %d)",mappedRecognitions.size(), bitmap.getWidth(), bitmap.getHeight());
//                        Log.d("myTag",str);
                        tracker.trackResults(mappedRecognitions, luminanceCopy, currTimestamp);
                        if (mappedRecognitions.size() > 0)
                            trackingOverlay.postInvalidate();
                    }
                });
        bitmap.recycle();
    }

    //added by bo
    public void onPeekTouch (){
        Log.d("myTag","on peek touch");

        if (andyRenderable == null) {
            return;
        }

        Camera camera=arFragment.getArSceneView().getArFrame().getCamera();
        Pose mCameraRelativePose= Pose.makeTranslation(0.0f, 0.0f, -1f);
        arSession = arFragment.getArSceneView().getSession();

        if(mCameraRelativePose==null) Log.d("myTag","pose is null");
        else Log.d("myTag","pose is not null");

        if(arSession==null) Log.d("myTag","arSession is null");
        Pose cPose = camera.getPose().compose(mCameraRelativePose).extractTranslation();
        if(cPose!=null)
            Log.d("myTag","camera pose is not null" + cPose.toString());
        Anchor anchor=arSession.createAnchor(cPose);

        if(anchor==null) Log.d("myTag","anchor is null");
        else Log.d("myTag","anchor is not null");

        Log.d("myTag","step 1");
        //copy&paste
        AnchorNode anchorNode = new AnchorNode(anchor);
        anchorNode.setParent(arFragment.getArSceneView().getScene());

        Log.d("myTag","step 2");

        // Create the transformable andy and add it to the anchor.
        TransformableNode andy = new TransformableNode(arFragment.getTransformationSystem());
        andy.setParent(anchorNode);
        andy.setRenderable(andyRenderable);
        andy.select();

    }


    /**
     * Returns false and displays an error message if Sceneform can not run, true if Sceneform can run
     * on this device.
     *
     * <p>Sceneform requires Android N on the device as well as OpenGL 3.1 capabilities.
     *
     * <p>Finishes the activity if Sceneform can not run
     */
    public static boolean checkIsSupportedDeviceOrFinish(final Activity activity) {
        if (Build.VERSION.SDK_INT < VERSION_CODES.N) {
            Log.e(TAG, "Sceneform requires Android N or later");
            Toast.makeText(activity, "Sceneform requires Android N or later", Toast.LENGTH_LONG).show();
            activity.finish();
            return false;
        }
        String openGlVersionString =
                ((ActivityManager) activity.getSystemService(Context.ACTIVITY_SERVICE))
                        .getDeviceConfigurationInfo()
                        .getGlEsVersion();
        if (Double.parseDouble(openGlVersionString) < MIN_OPENGL_VERSION) {
            Log.e(TAG, "Sceneform requires OpenGL ES 3.1 later");
            Toast.makeText(activity, "Sceneform requires OpenGL ES 3.1 or later", Toast.LENGTH_LONG)
                    .show();
            activity.finish();
            return false;
        }
        return true;
    }

    Bitmap imageToBitmap(Image image){
        String str=String.format("Image, width:%d,height:%d",image.getWidth(),image.getHeight());
        Log.d("myTag",str);
        if(!opencvLoaded) return null;
        Mat mat=  MyUtils.imageToMat(image);
        str=String.format("Mat, width:%d,height:%d",mat.cols(),mat.rows());
        Log.d("myTag",str);
        Bitmap bitmap=Bitmap.createBitmap(mat.cols(),  mat.rows(), Bitmap.Config.ARGB_8888);


        Utils.matToBitmap(mat,bitmap);
        str=String.format("Bitmap, width:%d,height:%d",bitmap.getWidth(),bitmap.getHeight());
        Log.d("myTag",str);

//        Matrix matrix = new Matrix();
//        matrix.postRotate(90);
//        Bitmap rotatedBitmap = Bitmap.createBitmap(bitmap , 0, 0, bitmap.getWidth(), bitmap.getHeight(), matrix, true);

//        str=String.format("RotatedBitmap, width:%d,height:%d",rotatedBitmap.getWidth(),rotatedBitmap.getHeight());
//        Log.d("myTag",str);

        //return rotatedBitmap;
        return bitmap;
    }

    //miniature image view set image
    void setImage(Bitmap image){
        runOnUiThread(new Runnable() {
            @Override
            public void run() {
                imgView.setImageBitmap(image);
            }
        });
    }

    void setImage(Image image){
        Log.d("myTag", Boolean.toString(opencvLoaded));
        if(!opencvLoaded) return;
        Mat mat=  MyUtils.imageToMat(image);
        Log.d("myTag","imageToMat");

        //ImageView imgview=findViewById(R.id.imgview);
//      Bitmap bitmap=Bitmap.createBitmap(image.getWidth(),  image.getHeight(),Bitmap.Config.ARGB_8888);
        Bitmap bitmap=Bitmap.createBitmap(mat.cols(),  mat.rows(), Bitmap.Config.ARGB_8888);

        Log.d("myTag","create a bitmap done");
        Utils.matToBitmap(mat,bitmap);

        Log.d("myTag","mat to bitmap");

        Matrix matrix = new Matrix();
        matrix.postRotate(90);

        Bitmap rotatedBitmap = Bitmap.createBitmap(bitmap , 0, 0, bitmap.getWidth(), bitmap.getHeight(), matrix, true);

        //imgview.setImageBitmap(rotatedBitmap);
        Log.d("myTag","set bitmap image");
    }



    private BaseLoaderCallback mLoaderCallback = new BaseLoaderCallback(this) {
        @Override
        public void onManagerConnected(int status) {
            opencvLoaded=false;
            switch (status) {
                case LoaderCallbackInterface.SUCCESS:
                {
                    Log.i("myTag", "OpenCV loaded successfully");
                    opencvLoaded=true;
                } break;
                default:
                {
                    Log.i("myTag", "OpenCV load default");
                    super.onManagerConnected(status);
                } break;
            }
        }
    };

    @Override
    public void onResume()
    {
        super.onResume();
        OpenCVLoader.initAsync(OpenCVLoader.OPENCV_VERSION_3_3_0, this, mLoaderCallback);
        handlerThread = new HandlerThread("inference");
        handlerThread.start();
        handler = new Handler(handlerThread.getLooper());
        arFragment.onResume();
    }


    public void addCallback(final OverlayView.DrawCallback callback) {
        final OverlayView overlay = (OverlayView) findViewById(R.id.debug_overlay);
        if (overlay != null) {
            overlay.addCallback(callback);
        }
    }

    @Override
    public synchronized void onPause() {

        handlerThread.quitSafely();
        try {
            handlerThread.join();
            handlerThread = null;
            handler = null;
        } catch (final InterruptedException e) {
            LOGGER.e(e, "Exception!");
        }
        arFragment.onPause();

        super.onPause();
    }

    protected synchronized void runInBackground(final Runnable r) {
        if (handler != null) {
            handler.post(r);
        }
    }

    protected int getScreenOrientation() {
        switch (getWindowManager().getDefaultDisplay().getRotation()) {
            case Surface.ROTATION_270:
                return 270;
            case Surface.ROTATION_180:
                return 180;
            case Surface.ROTATION_90:
                return 90;
            default:
                return 0;
        }
    }


    public void requestRender() {
        final OverlayView overlay = (OverlayView) findViewById(R.id.debug_overlay);
        if (overlay != null) {
            overlay.postInvalidate();
        }
    }
}
