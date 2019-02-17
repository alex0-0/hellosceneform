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
import android.app.Fragment;
import android.app.FragmentManager;
import android.content.Context;
import android.graphics.Bitmap;
import android.graphics.Canvas;
import android.graphics.Matrix;
import android.graphics.RectF;
import android.hardware.Sensor;
import android.hardware.SensorEvent;
import android.hardware.SensorEventListener;
import android.hardware.SensorManager;
import android.media.Image;
import android.net.Uri;
import android.os.AsyncTask;
import android.os.Build;
import android.os.Build.VERSION_CODES;
import android.os.Bundle;
import android.os.Handler;
import android.os.HandlerThread;
import android.os.SystemClock;
import android.support.v7.app.AppCompatActivity;
import android.util.Log;
import android.view.Display;
import android.view.Gravity;
import android.view.MotionEvent;
import android.view.Surface;
import android.view.View;
import android.widget.Button;
import android.widget.ImageView;
import android.widget.RadioButton;
import android.widget.RadioGroup;
import android.widget.SeekBar;
import android.widget.Switch;
import android.widget.TextView;
import android.widget.Toast;

import com.google.ar.core.Anchor;
import com.google.ar.core.Camera;
import com.google.ar.core.HitResult;
import com.google.ar.core.Plane;
import com.google.ar.core.Pose;
import com.google.ar.core.Session;
import com.google.ar.sceneform.AnchorNode;
import com.google.ar.sceneform.Node;
import com.google.ar.sceneform.NodeParent;
import com.google.ar.sceneform.math.Vector3;
import com.google.ar.sceneform.rendering.ModelRenderable;
import com.google.ar.sceneform.samples.hellosceneform.env.BorderedText;
import com.google.ar.sceneform.samples.hellosceneform.env.ImageUtils;
import com.google.ar.sceneform.samples.hellosceneform.tracking.MultiBoxTracker;
import com.google.ar.sceneform.ux.TransformableNode;

import org.opencv.android.BaseLoaderCallback;
import org.opencv.android.LoaderCallbackInterface;
import org.opencv.android.OpenCVLoader;
import org.opencv.android.Utils;
import org.opencv.core.KeyPoint;
import org.opencv.core.Mat;
import org.opencv.core.MatOfDMatch;
import org.opencv.core.MatOfKeyPoint;
import org.opencv.core.Rect;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Set;

import edu.umb.cs.imageprocessinglib.ImageProcessor;
import edu.umb.cs.imageprocessinglib.ObjectDetector;
import edu.umb.cs.imageprocessinglib.feature.FeatureStorage;
import edu.umb.cs.imageprocessinglib.model.BoxPosition;
import edu.umb.cs.imageprocessinglib.model.DescriptorType;
import edu.umb.cs.imageprocessinglib.model.ImageFeature;
import edu.umb.cs.imageprocessinglib.model.Recognition;

/**
 * This is an example activity that uses the Sceneform UX package to make common AR tasks easier.
 */
public class HelloSceneformActivity extends AppCompatActivity implements SensorEventListener, SavingFeatureDialog.OnFragmentInteractionListener {
    private  static final String TAG = "RECTANGLE_DEBUG";
    private static final int OWNER_STATE=1, VIEWER_STATE=2;
//    private static final String TAG = HelloSceneformActivity.class.getSimpleName();
    private static final double MIN_OPENGL_VERSION = 3.1;

    //fixed file name for storing metadata of image features and recognitions
    private static final String dataFileName = "data_file";

    private int state=OWNER_STATE;

    private TransformableNode andy;



    //image recognition object as key, value is a list of image features list recognized as this object by TF.
    //Each element is a distortion robust image feature, sorted as left, right, top and bottom
    private Map<String,List<List<ImageFeature>>> rs;

    private MyArFragment arFragment;
    private ModelRenderable andyRenderable;

    private Session arSession;//added by bo

    private float last_chk_time=0;
    private boolean opencvLoaded=false;
//    private Classifier classifier;
    private ObjectDetector objectDetector;

    private OverlayView trackingOverlay;
    /*** from tensorflow sample code***/
    private Handler handler;
    private long timestamp = 0; //it's actually a counter
    private Bitmap cropCopyBitmap = null;
    private Bitmap croppedBitmap = null;
    private Bitmap rgbFrameBitmap=null;
    private Bitmap copyBitmp = null;

    private HandlerThread handlerThread;
    private byte[][] yuvBytes = new byte[3][];
    private int[] rgbBytes = null;
    private int yRowStride;

    protected int previewWidth = 0;
    protected int previewHeight = 0;
    private ImageView imgView;

    private Integer sensorOrientation;
    private Matrix frameToCropTransform;
    private Matrix cropToFrameTransform;
    private Matrix frameToDisplayTransform;
    private int rotation=90;

    private MultiBoxTracker tracker;

    private byte[] luminanceCopy;
    private List<Recognition> recognitions;

//    private Classifier detector;
    private BorderedText borderedText;

    static Boolean onRecord = false;
    static Boolean onRetrieve = false;

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
        OpenCVLoader.initDebug();

        setContentView(R.layout.activity_ux);
        arFragment = (MyArFragment) getSupportFragmentManager().findFragmentById(R.id.ux_fragment);
        arFragment.getPlaneDiscoveryController().hide();
        arFragment.getPlaneDiscoveryController().setInstructionView(null);
        imgView = findViewById(R.id.imgview);

        //orientation sensor manager
        mSensorManager = (SensorManager) getSystemService(SENSOR_SERVICE);
        mRotationVectorSensor = mSensorManager.getDefaultSensor(
                Sensor.TYPE_ROTATION_VECTOR);
        mSensorManager.registerListener(this, mRotationVectorSensor, 10000);

        Display display = this.getWindowManager().getDefaultDisplay();
        int stageWidth = display.getWidth();
        int stageHeight = display.getHeight();
        Log.v("myTag","screen size "+Integer.toString(stageWidth)+","+Integer.toString(stageHeight));

        //ImageView imgview=findViewById(R.id.imgview);

        //imgview.setImageResource(R.drawable.ic_launcher);

        arFragment.setActivity(this);//add by bo
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

                //TODO: is the conversion done in correct way?
                luminanceCopy = MyUtils.imageToByte(img); //convert image to byte[]
                bitmap=MyUtils.imageToBitmap(img);

                //added by bo to scale down the image
//                bitmap=Bitmap.createScaledBitmap(bitmap, 360,640,false);


                img.close();
                //if(bitmap!=null) setImage(bitmap);
                //else return;
            }catch(Exception e){System.out.println("myTag"+e); return;}

//            if(detector==null) initTF(bitmap);
            if(objectDetector==null) initTF(bitmap);

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
        Button recBtn = findViewById(R.id.record);  //record button
        Button rteBtn = findViewById(R.id.retrieve);    //retrieve button
        recBtn.setTag("Place VO");
        recBtn.setOnClickListener(new View.OnClickListener(){
            public void onClick(View view) {
                SeekBar sbar=findViewById(R.id.seekBar);
               Button btn=(Button) view;
                String tag=(String)btn.getTag();
                if(tag.equals("Place VO")) {
                    btn.setText("Confirm");
                    btn.setTag("Confirm");
                    placeAndy();
                    sbar.setProgress(50);
                    sbar.setVisibility(View.VISIBLE);

                }
                else{
                    onRecord = true;
                    btn.setTag("Place VO");
                    btn.setText("Place VO");
                    sbar.setVisibility(View.INVISIBLE);

                }
            }
        });
        rteBtn.setOnClickListener(new View.OnClickListener(){
            public void onClick(View view) {
                AsyncTask.execute(()->{
                    loadData();
                    onRetrieve = true;
                });
            }
        });


        RadioButton rb=findViewById(R.id.rb_owner);
        rb.setChecked(true);
        rteBtn.setEnabled(false);

        RadioGroup radioGroup=findViewById(R.id.rg_role);
        radioGroup.setOnCheckedChangeListener(new RadioGroup.OnCheckedChangeListener() {
            @Override
            public void onCheckedChanged(RadioGroup group, int checkedId) {
                RadioButton rb = (RadioButton) group.findViewById(checkedId);
                String msg= "Switch to "+ rb.getText();
                if (null != rb ) {
                    Toast.makeText(HelloSceneformActivity.this, msg, Toast.LENGTH_SHORT).show();
                }
                if(rb.getId()==R.id.rb_owner){
                    recBtn.setEnabled(true);
                    rteBtn.setEnabled(false);
                    state=OWNER_STATE;
                }else{
                    recBtn.setEnabled(false);
                    rteBtn.setEnabled(true);
                    state=VIEWER_STATE;
                    andy.setParent(null);
                }
            }
        });

        SeekBar sbar=findViewById(R.id.seekBar);
        sbar.setMax(100);
        sbar.setMin(0);
        sbar.setVisibility(View.INVISIBLE);

        sbar.setOnSeekBarChangeListener(new SeekBar.OnSeekBarChangeListener() {
                                            @Override
                                            public void onProgressChanged(SeekBar seekBar, int i, boolean b) {
                                                float dist=(float)((i-50)*0.01+1);
                                                placeAndyWithDist(dist);
                                            }

                                            @Override
                                            public void onStartTrackingTouch(SeekBar seekBar) {

                                            }

                                            @Override
                                            public void onStopTrackingTouch(SeekBar seekBar) {

                                            }
                                        }


        );
    }

    void loadData() {
        if (rs==null) {
            FeatureStorage fs = new FeatureStorage();
            String data = MyUtils.readFromFile(dataFileName, this);
            rs = new HashMap<>();
            String[] recStrs = data.split("\n");
            String[] os = recStrs[0].split(" ");
            String dirPath = getFilesDir().getPath();
            //get orientation data
            refRD = new RotationData(new Float(os[0]),new Float(os[1]),new Float(os[2]));
            for (int i=1; i<recStrs.length; i++) {
                String r = recStrs[i];
                String[] rec = r.split("\t");
                if (rs.get(rec[0]) == null)
                    rs.put(rec[0], new ArrayList<>());
                //restore image features
                String fName = dirPath + "/" + rec[2];
                List<ImageFeature> IFs = new ArrayList<>();
                IFs.add(fs.loadFPfromFile(fName + "_left"));
                IFs.add(fs.loadFPfromFile(fName + "_right"));
                IFs.add(fs.loadFPfromFile(fName + "_top"));
                IFs.add(fs.loadFPfromFile(fName + "_bottom"));
                rs.get(rec[0]).add(IFs);
            }
        }
        runOnUiThread(()->{
            Toast.makeText(getApplicationContext(), "Data loaded", Toast.LENGTH_SHORT).show();
        });
    }

    void initTF(Bitmap bitmap) {
        previewWidth = bitmap.getWidth();
        previewHeight = bitmap.getHeight();
        sensorOrientation = rotation - getScreenOrientation();
        objectDetector = new ObjectDetector();
        objectDetector.init(this);

        //last_chk_time=curTime;
        /*** from detector activity in tensorflow sample code***/
//        final float textSizePx =
//                TypedValue.applyDimension(
//                        TypedValue.COMPLEX_UNIT_DIP, TEXT_SIZE_DIP, getResources().getDisplayMetrics());
//        borderedText = new BorderedText(textSizePx);
//        borderedText.setTypeface(Typeface.MONOSPACE);
//
        //if(tracker==null)
        tracker = new MultiBoxTracker(this);




//        sensorOrientation = rotation - getScreenOrientation();
//        /*
//        LOGGER.i("Camera orientation relative to screen canvas: %d", sensorOrientation);
//
//        LOGGER.i("Initializing at size %dx%d", previewWidth, previewHeight);
//        */
//        rgbFrameBitmap = Bitmap.createBitmap(previewWidth, previewHeight, Bitmap.Config.ARGB_8888);
//
        croppedBitmap = Bitmap.createBitmap(300, 300, Bitmap.Config.ARGB_8888);
        copyBitmp = Bitmap.createBitmap(previewWidth, previewHeight, Bitmap.Config.ARGB_8888);
//
        frameToCropTransform =
                ImageUtils.getTransformationMatrix(
                        previewWidth, previewHeight,
                        300, 300,
                        sensorOrientation, false);
//        frameToCropTransform = new Matrix();
//        frameToCropTransform.postRotate(sensorOrientation);
//        frameToCropTransform =
//                ImageUtils.getTransformationMatrix(
//                        previewWidth, previewHeight,
//                        previewWidth, previewHeight,
//                        sensorOrientation, MAINTAIN_ASPECT);
        cropToFrameTransform = new Matrix();
        frameToCropTransform.invert(cropToFrameTransform);
//
//        /**/
//        float h = arFragment.getView().getHeight();//1944
//        float w = arFragment.getView().getWidth();//1080
        trackingOverlay = (OverlayView) findViewById(R.id.tracking_overlay);
//        float dpHeight = 1005;//trackingOverlay.getHeight() / 2;
//        float dpWidth = 540;//trackingOverlay.getWidth() / 2;
//        //75 is the height of image view
////        frameToDisplayTransform =
////                ImageUtils.getTransformationMatrix(
////                        previewWidth, previewHeight,
////                        (int)dpHeight, (int)dpWidth,
////                        0, MAINTAIN_ASPECT);
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

    void processImage(Bitmap bitmap) {
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
//        final Canvas canvas = new Canvas(rgbFrameBitmap);
        canvas.drawBitmap(bitmap, frameToCropTransform, null);

//        Bitmap copy = Bitmap.createBitmap(bitmap);
//        Bitmap copy = bitmap.copy(bitmap.getConfig(), true);
        final Canvas c = new Canvas(copyBitmp);
//        final Canvas canvas = new Canvas(rgbFrameBitmap);
        c.drawBitmap(bitmap, new Matrix(), null);

//        setImage(croppedBitmap);
        runInBackground(
                new Runnable() {
                    @Override
                    public void run() {
                        final long startTime = SystemClock.uptimeMillis();

                        Log.d("myTag","before recognizeimage");
                        final List<Recognition> results = objectDetector.recognizeImage(croppedBitmap);
//                        final List<Recognition> results = objectDetector.recognizeImage(rgbFrameBitmap);
//
                        org.opencv.core.Rect roi = new org.opencv.core.Rect();
//
//                        Log.d("myTag",str);
                        for (final Recognition result : results) {
                            BoxPosition pos = result.getLocation();
                            final RectF location = new RectF(pos.getLeft(), pos.getTop(), pos.getRight(), pos.getBottom());
//                            roi = new org.opencv.core.Rect((int)location.left, (int)location.top, (int)(location.right - location.left), (int)(location.bottom - location.top));
                            Log.d(TAG, "main: rect before transform " + location.toString());
                            cropToFrameTransform.mapRect(location);
                            roi = new org.opencv.core.Rect((int)location.left, (int)location.top, (int)(location.right - location.left), (int)(location.bottom - location.top));
                            result.setLocation(new BoxPosition(location.left, location.top, location.width(), location.height()));
                        }

                        if (results.size() > 0) {
                            if (onRecord) {
                                record(copyBitmp, results);
                            }
                            else if (onRetrieve) {
                                retrieve(copyBitmp, results);
                            }
//                            Mat mat = new Mat();
//                            Utils.bitmapToMat(copyBitmp, mat);
//                            Mat cropMat = new Mat(mat, roi);
//                            Bitmap tBM = Bitmap.createBitmap(roi.width, roi.height, Bitmap.Config.ARGB_8888);
//                            Utils.matToBitmap(cropMat, tBM);
//                            mat.release();
//                            cropMat.release();
//                            setImage(tBM);
//                            copy.recycle();
                        }
                        else if (onRecord) {
                            runOnUiThread(()->{
                                Toast.makeText(getApplicationContext(), "There is no recognized object in frame", Toast.LENGTH_SHORT).show();
                            });
                            onRecord = false;
                        }
//                        RectF rectF = new RectF(9, 79, 283, 216);//previewWidth, previewHeight;
//                        cropToFrameTransform.mapRect(rectF);
//                        RectF rectF = new RectF(0, 0, previewWidth, previewHeight);
//                        Classifier.Recognition result = new Classifier.Recognition("1434", "test", 0.99f, rectF);
//                        mappedRecognitions.add(result);

//                        str=String.format("mapped:%d, cropped image size(%d, %d)",mappedRecognitions.size(), bitmap.getWidth(), bitmap.getHeight());
//                        Log.d("myTag",str);
//                        tracker.trackResults(mappedRecognitions, luminanceCopy, currTimestamp);
                        tracker.trackResults(results, luminanceCopy, currTimestamp);
//                        if (mappedRecognitions.size() > 0)
                        if (results.size() > 0)
                            trackingOverlay.postInvalidate();
                    }
                });
        bitmap.recycle();
    }

    static int kTemplateFPNum = 100;
    static int kDisThd = 300;
    private void record(Bitmap img, List<Recognition> recognitions) {
        runOnUiThread(()->{
            FragmentManager fm=getFragmentManager();
            SavingFeatureDialog sf=new SavingFeatureDialog();
            sf.show(fm, "sf_dialog");
        });

        setAngle();
        FeatureStorage fs = new FeatureStorage();
        Mat mat = new Mat();
        Utils.bitmapToMat(img, mat);
        StringBuilder data = new StringBuilder();
        data.append(refRD);
        String dirPath = getFilesDir().getPath();
        for (Recognition r : recognitions) {
            BoxPosition location = r.getLocation();
            Rect roi = new Rect(location.getLeftInt(), location.getTopInt(), location.getWidthInt(), location.getHeightInt());
            Mat tMat = new Mat(mat, roi);

            //TODO:figure out if the orientation of the original image influence the final matching
            //At present the image is counter-clock rotated 90 degrees
            List<Mat> leftImgs = ImageProcessor.changeToLeftPerspective(tMat, 5f, 10);
            for (Mat i : leftImgs) {
                Bitmap bitmap=Bitmap.createBitmap(i.cols(),  i.rows(), Bitmap.Config.ARGB_8888);
                Utils.matToBitmap(i,bitmap);
                int k = 0;
            }

            data.append("\n" + r.getTitle() + "\t" + r.getConfidence() + "\t" + r.getUuid());

            fs.saveFPtoFile( dirPath + "/" + r.getUuid() + "_left",
                    ImageProcessor.extractRobustFeatures(tMat, ImageProcessor.changeToLeftPerspective(tMat, 5f, 10),
                            kTemplateFPNum, kDisThd, DescriptorType.ORB, null));
            fs.saveFPtoFile( dirPath + "/" + r.getUuid() + "_right",
                    ImageProcessor.extractRobustFeatures(tMat, ImageProcessor.changeToRightPerspective(tMat, 5f, 10),
                            kTemplateFPNum, kDisThd, DescriptorType.ORB, null));
            fs.saveFPtoFile( dirPath + "/" + r.getUuid() + "_bottom",
                    ImageProcessor.extractRobustFeatures(tMat, ImageProcessor.changeToBottomPerspective(tMat, 5f, 10),
                            kTemplateFPNum, kDisThd, DescriptorType.ORB, null));
            fs.saveFPtoFile( dirPath + "/" + r.getUuid() + "_top",
                    ImageProcessor.extractRobustFeatures(tMat, ImageProcessor.changeToTopPerspective(tMat, 5f, 10),
                            kTemplateFPNum, kDisThd, DescriptorType.ORB, null));
        }
        MyUtils.writeToFile(dataFileName, data.toString(), this);
        runOnUiThread(()->{
            Toast.makeText(getApplicationContext(), "Image features saved", Toast.LENGTH_SHORT).show();
            FragmentManager fm=getFragmentManager();
            SavingFeatureDialog sf=(SavingFeatureDialog)fm.findFragmentByTag("sf_dialog");
            if(sf!=null) sf.dismiss();;

        });
        onRecord = false;

    }

    private void retrieve(Bitmap img, List<Recognition> recognitions) {
        Mat mat = new Mat();
        Utils.bitmapToMat(img, mat);
        Set<String> recs = rs.keySet();
        StringBuilder sb = new StringBuilder();
        //horizontal angle difference, positive stands for right perspective, negative for left perspective
        float hd = (Math.abs(cRD.z - refRD.z)>180)?(360-(cRD.z-refRD.z)) : (cRD.z-refRD.z);
        //vertical angle difference, assume the angle can't exceed 90
        float vd = cRD.x-refRD.x;
        if (Math.abs(hd) > 90 || Math.abs(vd) > 90)
            sb.append("angle difference larger than 90 degree");
        else {
            for (Recognition r : recognitions) {
                if (recs.contains(r.getTitle())) {
                    BoxPosition location = r.getLocation();
                    Rect roi = new Rect(location.getLeftInt(), location.getTopInt(), location.getWidthInt(), location.getHeightInt());
                    Mat qMat = new Mat(mat, roi);
                    ImageFeature qIF = ImageProcessor.extractORBFeatures(qMat, 500);
                    List<List<ImageFeature>> tIFs = rs.get(r.getTitle());
                    for (List<ImageFeature> ts : tIFs) {
                        ImageFeature tIF = constructTemplateFP(ts, hd, vd, kTemplateFPNum);
                        MatOfDMatch matches = ImageProcessor.matchWithRegression(qIF, tIF, 5, 300, 20);
                        sb.append(r.getTitle() + " " + (float) matches.total() / tIF.getSize() + ",");
                    }
                }
            }
        }
        runOnUiThread(()->{
            TextView tv = findViewById(R.id.mratio);
            tv.setText(sb.toString());
        });
    }

    ImageFeature constructTemplateFP(List<ImageFeature> tIFs, float hd, float vd, int tNum) {
        //tIFs is sorted as left, right, top and bottom
        //calculate ratios
        float hr = Math.abs(hd)/(Math.abs(hd) + Math.abs(vd));
        float vr = Math.abs(vd)/(Math.abs(hd) + Math.abs(vd));
        ImageFeature IF1;
        ImageFeature IF2;
        //guarantee the feature point robust on more-changed orientation is returned at first
        if (hr >= vr) {
            int num = (int)(hr * tNum);
            IF1 = (hd>0)?tIFs.get(1) : tIFs.get(0);
            if (IF1.getSize() > num)
                IF1 = IF1.subImageFeature(0, num);
            IF2 = (vd>0)?tIFs.get(3) : tIFs.get(2);
        } else {
            int num = (int)(vr * tNum);
            IF1 = (vd>0)?tIFs.get(3) : tIFs.get(2);
            if (IF1.getSize() > num)
                IF1 = IF1.subImageFeature(0, num);
            IF2 = (hd>0)?tIFs.get(1) : tIFs.get(0);
        }
        if (IF1.getSize() >= tNum) return IF1;
        //the number of FP from the other ImageFeature
        int num = tNum - IF1.getSize();
        List<KeyPoint> kp = new ArrayList<>(IF1.getObjectKeypoints().toList());
        Mat des = new Mat();//new Size(IF1.getDescriptors().cols(),tNum), IF1.getDescriptors().type());
        des.push_back(IF1.getDescriptors());

        List<KeyPoint> kp1 = IF1.getObjectKeypoints().toList();
        List<KeyPoint> kp2 = IF2.getObjectKeypoints().toList();
        for (int i=0; i < kp2.size(); i++) {
            KeyPoint k = kp2.get(i);
            boolean newFP = true;
            for (KeyPoint k1 : kp1) {
                if (k1.pt.x != k.pt.x || k1.pt.y != k.pt.y)
                    continue;
                else {
                    newFP = false;
                    break;
                }
            }
            if (newFP) {
                kp.add(k);
                Mat tMat = IF2.getDescriptors().row(i);
                des.push_back(tMat);
            }
            if (kp.size() >= num)
                break;
        }
        MatOfKeyPoint tKP = new MatOfKeyPoint();
        tKP.fromList(kp);
        return new ImageFeature(tKP, des, IF1.getDescriptorType());
    }

    //added by bo
    public void onPeekTouch (){
        Log.d("myTag","on peek touch");

        return;
/*
        if (andyRenderable == null) {
//        if (true) {
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
*/
    }


    void placeAndy(){
        if (andyRenderable == null) {
//        if (true) {
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
        if(andy==null) andy = new TransformableNode(arFragment.getTransformationSystem());
        placeAndy(anchorNode);
    }

    void placeAndyWithDist(float dist){
        if (andyRenderable == null) {
//        if (true) {
            return;
        }

        Camera camera=arFragment.getArSceneView().getArFrame().getCamera();
        Pose mCameraRelativePose= Pose.makeTranslation(0.0f, 0.0f, -dist);
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

        //copy&paste
        AnchorNode anchorNode = new AnchorNode(anchor);
        anchorNode.setParent(arFragment.getArSceneView().getScene());

        if(andy==null) andy = new TransformableNode(arFragment.getTransformationSystem());
        placeAndy(anchorNode);
    }

    void placeAndy(AnchorNode an){
        andy.setParent(an);
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
        OpenCVLoader.initAsync(OpenCVLoader.OPENCV_VERSION, this, mLoaderCallback);
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
//            LOGGER.e(e, "Exception!");
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


    //Codes below this line is for orientation monitor
    private RotationData refRD =null;
    private RotationData cRD=null;
    private SensorManager mSensorManager;
    private Sensor mRotationVectorSensor;
    private boolean checkAngle=false, firstValue=false;

    public void onAccuracyChanged(Sensor sensor, int accuracy) {
    }
    public void onSensorChanged(SensorEvent event) {
        if (firstValue) {
            refRD =new RotationData(event.values);
            firstValue=false;
        } else {
            cRD=new RotationData(event.values);
            displayData(cRD);
        }
    }

    void displayData(RotationData temp){
        TextView textView=findViewById(R.id.cangle);
        if (refRD != null)
            textView.setText(refRD.toString()+", "+temp.toString());
    }

    void setAngle(){
        refRD = cRD;    //just in case cRD is re-assigned value
        checkAngle=true;
        firstValue=true;
    }

    @Override
    public void onFragmentInteraction(Uri uri) {

    }

    private class RotationData{
        private float x,y,z,cos;
        private static final int FROM_RADS_TO_DEGS = -57;

        RotationData(float x, float y, float z){
            this.x = x;
            this.y = y; //
            this.z = z;
        }

        RotationData(float[] values){
//            x=values[0];
//            y=values[1];
//            z=values[2];
//            cos=values[3];
            float[] rotationMatrix = new float[9];
            SensorManager.getRotationMatrixFromVector(rotationMatrix, values);
            int worldAxisX = SensorManager.AXIS_X;
            int worldAxisZ = SensorManager.AXIS_Z;
            float[] adjustedRotationMatrix = new float[9];
            SensorManager.remapCoordinateSystem(rotationMatrix, worldAxisX, worldAxisZ, adjustedRotationMatrix);
            float[] orientation = new float[3];
            SensorManager.getOrientation(adjustedRotationMatrix, orientation);
            float pitch = orientation[1] * FROM_RADS_TO_DEGS;
            float roll = orientation[2] * FROM_RADS_TO_DEGS;
            float azimuth = orientation[0] * FROM_RADS_TO_DEGS;
            x = pitch;  //top,bottom perspective. -90~90.
            y = roll;   //rotation on same vertical plane. -180~180
            z = azimuth;    //left,right perspective. -180~180
        }

        public String toString(){
            //return Float.toString(x)+" "+Float.toString(y)+" "+Float.toString(z);

//            return String.format("%.02f %.02f %.02f",Math.asin(x)/Math.PI*180,Math.asin(y)/Math.PI*180,Math.asin(z)/Math.PI*180);
            return String.format("%.02f %.02f %.02f", x, y, z);
            //return Double.toString(Math.asin(x))+" "+Double.toString(Math.asin(y))+" "+Double.toString(Math.asin(z));
        }
    }
}
