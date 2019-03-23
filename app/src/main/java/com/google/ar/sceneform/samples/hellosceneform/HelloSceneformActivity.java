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
import android.graphics.Point;
import android.graphics.RectF;
import android.hardware.Sensor;
import android.hardware.SensorEvent;
import android.hardware.SensorEventListener;
import android.hardware.SensorManager;
import android.hardware.camera2.CameraAccessException;
import android.hardware.camera2.CameraCharacteristics;
import android.hardware.camera2.CameraManager;
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
import android.util.DisplayMetrics;
import android.util.Log;
import android.util.SizeF;
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
import com.google.ar.sceneform.math.Vector3;
import com.google.ar.sceneform.rendering.ModelRenderable;
import com.google.ar.sceneform.samples.hellosceneform.env.BorderedText;
import com.google.ar.sceneform.samples.hellosceneform.env.ImageUtils;
import com.google.ar.sceneform.samples.hellosceneform.env.Size;
import com.google.ar.sceneform.samples.hellosceneform.tracking.MultiBoxTracker;
import com.google.ar.sceneform.ux.TransformableNode;

import org.opencv.android.BaseLoaderCallback;
import org.opencv.android.LoaderCallbackInterface;
import org.opencv.android.OpenCVLoader;
import org.opencv.android.Utils;
import org.opencv.core.DMatch;
import org.opencv.core.KeyPoint;
import org.opencv.core.Mat;
import org.opencv.core.MatOfDMatch;
import org.opencv.core.MatOfKeyPoint;
import org.opencv.core.Rect;

import java.util.ArrayList;
import java.util.Arrays;
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
    private  static final String TAG = "MAIN_DEBUG";
    private static final int OWNER_STATE=1, VIEWER_STATE=2;
//    private static final String TAG = HelloSceneformActivity.class.getSimpleName();
    private static final double MIN_OPENGL_VERSION = 3.1;

    //fixed file name for storing metadata of image features and recognitions
    private static final String dataFileName = "data_file";

    private int state=OWNER_STATE;

    private TransformableNode andy;

    private float v_viewangle=60, h_viewangle=48;

    private float VO_dist=0, VO_dist_for_viewer=0;


    //image recognition object as key, value is a list of image features list recognized as this object by TF.
    //Each element is a distortion robust image feature, sorted as left, right, top and bottom
    private Map<String,List<List<ImageFeature>>> rs;
    private Map<String,List<BoxPosition>> bs; //store position
    Size imgSize;

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

        //calculate filed of views
        setFOV();

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
        rteBtn.setTag("Retrieve");
        recBtn.setOnClickListener(new View.OnClickListener(){
            public void onClick(View view) {
                SeekBar sbar=findViewById(R.id.seekBar);
                Button btn=(Button) view;
                String tag=(String)btn.getTag();
                if(tag.equals("Place VO")) {
                    placeAndy();
                    runOnUiThread(()-> {
                        btn.setText("Confirm");
                        btn.setTag("Confirm");

                        sbar.setProgress(50);
                        sbar.setVisibility(View.VISIBLE);

                    });

                }
                else{
                    onRecord = true;
                    runOnUiThread(()-> {
                        btn.setTag("Place VO");
                        btn.setText("Place VO");
                        sbar.setVisibility(View.INVISIBLE);
                    });


                }
            }
        });
        rteBtn.setOnClickListener(new View.OnClickListener(){
            public void onClick(View view) {
                AsyncTask.execute(()->{
                    Button btn=(Button) view;
                    String tag=(String)btn.getTag();
                    if(tag.equals("Retrieve")) {

                        loadData();
                        onRetrieve = true;
                        runOnUiThread(()-> {
                            btn.setText("Clear");
                            btn.setTag("Clear");
                            //btn.setEnabled(false);
                        });
                    }
                    else{
                        onRetrieve=false;
                        runOnUiThread(()-> {
                            btn.setTag("Retrieve");
                            btn.setText("Retrieve");
                            andy.setParent(null);
                        });

                    }

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
                    onRetrieve=false;
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
                                                VO_dist=dist;
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

    //FOV (rectilinear) =  2 * arctan (frame size/(focal length * 2))
    void setFOV() {
        //suppose there is only one camera
        int camNum = 0;
        CameraManager manager = (CameraManager) getSystemService(Context.CAMERA_SERVICE);
        try {
            String[] cameraIds = manager.getCameraIdList();
            for (String id : cameraIds) {
//            if (cameraIds.length > camNum) {
//                CameraCharacteristics characteristics = manager.getCameraCharacteristics(cameraIds[camNum]);
                CameraCharacteristics characteristics = manager.getCameraCharacteristics(id);
//                size = character.get(CameraCharacteristics.SENSOR_INFO_PHYSICAL_SIZE);
                int cOrientation = characteristics.get(CameraCharacteristics.LENS_FACING);
                if (cOrientation == CameraCharacteristics.LENS_FACING_BACK) {
                    float[] maxFocus = characteristics.get(CameraCharacteristics.LENS_INFO_AVAILABLE_FOCAL_LENGTHS);
                    SizeF size = characteristics.get(CameraCharacteristics.SENSOR_INFO_PHYSICAL_SIZE);
                    float w = size.getWidth();
                    float h = size.getHeight();
                    h_viewangle = (float) (2 * Math.atan(w / (maxFocus[0] * 2)))/(float)(2*Math.PI)*360;
                    v_viewangle = (float) (2 * Math.atan(h / (maxFocus[0] * 2)))/(float)(2*Math.PI)*360;
                }
            }
        }
        catch (CameraAccessException e)
        {
            Log.e(TAG, e.getMessage(), e);
        }
    }

    void loadData() {
        if (rs==null) {
            FeatureStorage fs = new FeatureStorage();
            String data = MyUtils.readFromFile(dataFileName, this);
            rs = new HashMap<>();
            bs = new HashMap<>();
            String[] recStrs = data.split("\n");
            String[] os = recStrs[0].split(" ");
            String dirPath = getFilesDir().getPath();
            //get orientation data
            refRD = new RotationData(new Float(os[0]),new Float(os[1]),new Float(os[2]));
            imgSize = new Size(new Integer(os[3]), new Integer(os[4]));
            VO_dist_for_viewer= Float.parseFloat(os[5]);
            for (int i=1; i<recStrs.length; i++) {
//                data.append("\n" + r.getTitle() + "\t" + r.getConfidence() + "\t" + r.getUuid() //recognition
//                        + "\t" + location.getTop() + "\t" + location.getLeft() + "\t" + location.getBottom() + "\t" + location.getRight()); //location
                String r = recStrs[i];
                String[] rec = r.split("\t");
                if (rs.get(rec[0]) == null)
                    rs.put(rec[0], new ArrayList<>());
                if (bs.get(rec[0]) == null)
                    bs.put(rec[0], new ArrayList<>());
                //restore image features
                String fName = dirPath + "/" + rec[2];
                List<ImageFeature> IFs = new ArrayList<>();
                IFs.add(fs.loadFPfromFile(fName + "_left"));
                IFs.add(fs.loadFPfromFile(fName + "_right"));
                IFs.add(fs.loadFPfromFile(fName + "_top"));
                IFs.add(fs.loadFPfromFile(fName + "_bottom"));
                IFs.add(fs.loadFPfromFile(fName + "_scale_up"));
                IFs.add(fs.loadFPfromFile(fName + "_scale_down"));
                rs.get(rec[0]).add(IFs);
                bs.get(rec[0]).add(new BoxPosition(new Float(rec[4]), new Float(rec[3]),
                        new Float(rec[6])-new Float(rec[4]),new Float(rec[5])-new Float(rec[3])));
                //bs.put(rec[0],new BoxPosition(new Float(rec[4]), new Float(rec[3]),
                //        new Float(rec[6])-new Float(rec[4]),new Float(rec[5])-new Float(rec[3])));
            }
        }
        runOnUiThread(()->{
            Toast.makeText(getApplicationContext(), "Data loaded", Toast.LENGTH_SHORT).show();
            Button btn= findViewById(R.id.retrieve);
            btn.setEnabled(true);
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

    long timeStamp = 0;
    static private long kInterval = 500;

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

        //stop the background thread when program is halted
        if (System.currentTimeMillis() - timeStamp > kInterval && handler != null) {
            timeStamp = System.currentTimeMillis();
        } else return;

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

                        Log.d("myTag","before recognizeimage");
                        final long startTime = System.currentTimeMillis();
                        final List<Recognition> results = objectDetector.recognizeImage(croppedBitmap);
//                        final List<Recognition> results = objectDetector.recognizeImage(rgbFrameBitmap);
                        long endTime = System.currentTimeMillis();
                        Log.d(TAG, "Recognition number:\t" + results.size() + " time:\t"+(endTime-startTime));
//
                        org.opencv.core.Rect roi = new org.opencv.core.Rect();
//
//                        Log.d("myTag",str);
                        for (final Recognition result : results) {
                            BoxPosition pos = result.getLocation();
                            final RectF location = new RectF(pos.getLeft(), pos.getTop(), pos.getRight(), pos.getBottom());
//                            roi = new org.opencv.core.Rect((int)location.left, (int)location.top, (int)(location.right - location.left), (int)(location.bottom - location.top));
//                            Log.d(TAG, "main: rect before transform " + location.toString());
                            cropToFrameTransform.mapRect(location);
                            roi = new org.opencv.core.Rect((int)location.left, (int)location.top, (int)(location.right - location.left), (int)(location.bottom - location.top));
                            result.setLocation(new BoxPosition(location.left, location.top, location.width(), location.height()));
                        }

                        if (results.size() > 0) {
                            Log.d("myTag","result size >0:"+Integer.toString(results.size()));

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
    static int kDisThd = 400;
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
        data.append(" " + img.getWidth() + " " + img.getHeight()+" " + VO_dist);
        String dirPath = getFilesDir().getPath();
        for (Recognition r : recognitions) {
            BoxPosition location = r.getLocation();
            Rect roi = new Rect(location.getLeftInt(), location.getTopInt(), location.getWidthInt(), location.getHeightInt());
            Mat tMat = new Mat(mat, roi);

            //TODO:figure out if the orientation of the original image influence the final matching
            //At present the image is counter-clock rotated 90 degrees
//            List<Mat> leftImgs = ImageProcessor.changeToLeftPerspective(tMat, 5f, 10);
//            for (Mat i : leftImgs) {
//                Bitmap bitmap=Bitmap.createBitmap(i.cols(),  i.rows(), Bitmap.Config.ARGB_8888);
//                Utils.matToBitmap(i,bitmap);
//                int k = 0;
//            }

            data.append("\n" + r.getTitle() + "\t" + r.getConfidence() + "\t" + r.getUuid() //recognition
                    + "\t" + location.getTop() + "\t" + location.getLeft() + "\t" + location.getBottom() + "\t" + location.getRight()); //location

            long startTime, endTime;
            startTime = System.currentTimeMillis();
            ImageFeature tIF = ImageProcessor.extractFeatures(tMat);
            endTime = System.currentTimeMillis();
            Log.d("ar_timer", String.format("extract tIF time:%d", endTime-startTime));


//            startTime = System.currentTimeMillis();
//            ImageFeature i1 = ImageProcessor.extractRobustFeatures(tIF, ImageProcessor.changeToLeftPerspective(tMat, 5f, 10),
//                            (int)(1.2*kTemplateFPNum), kDisThd, DescriptorType.ORB, null);
//            endTime = System.currentTimeMillis();
//            Log.d("ar_timer", String.format("first extraction time:%d", endTime-startTime));
            startTime = System.currentTimeMillis();
            fs.saveFPtoFile( dirPath + "/" + r.getUuid() + "_left", //i1);
                    ImageProcessor.extractRobustFeatures(tIF, ImageProcessor.changeToLeftPerspective(tMat, 5f, 10),
                            (int)(1.2*kTemplateFPNum), kDisThd, DescriptorType.ORB, null));
            endTime = System.currentTimeMillis();
            Log.d("ar_timer", String.format("left saving time:%d", endTime-startTime));
//            Log.d("ar_timer", String.format("first saving time:%d", endTime-startTime));
            startTime = System.currentTimeMillis();

            fs.saveFPtoFile( dirPath + "/" + r.getUuid() + "_right",// i2);
                    ImageProcessor.extractRobustFeatures(tIF, ImageProcessor.changeToRightPerspective(tMat, 5f, 10),
                            (int)(1.2*kTemplateFPNum), kDisThd, DescriptorType.ORB, null));
            endTime = System.currentTimeMillis();
            Log.d("ar_timer", String.format("right saving time:%d", endTime-startTime));
            startTime = System.currentTimeMillis();

            fs.saveFPtoFile( dirPath + "/" + r.getUuid() + "_bottom",//i3);
                    ImageProcessor.extractRobustFeatures(tIF, ImageProcessor.changeToBottomPerspective(tMat, 5f, 10),
                            (int)(1.2*kTemplateFPNum), kDisThd, DescriptorType.ORB, null));
            endTime = System.currentTimeMillis();
            Log.d("ar_timer", String.format("bottom saving time:%d", endTime-startTime));
            startTime = System.currentTimeMillis();

            fs.saveFPtoFile( dirPath + "/" + r.getUuid() + "_top",//i4);
                    ImageProcessor.extractRobustFeatures(tIF, ImageProcessor.changeToTopPerspective(tMat, 5f, 10),
                            (int)(1.2*kTemplateFPNum), kDisThd, DescriptorType.ORB, null));
            endTime = System.currentTimeMillis();
            Log.d("ar_timer", String.format("top saving time:%d", endTime-startTime));
            startTime = System.currentTimeMillis();

            fs.saveFPtoFile( dirPath + "/" + r.getUuid() + "_scale_up", //i5);
                    ImageProcessor.extractRobustFeatures(tIF, ImageProcessor.scaleImage(tMat, 0.05f, 10),
                            (int)(1.2*kTemplateFPNum), kDisThd, DescriptorType.ORB, null));
            endTime = System.currentTimeMillis();
            Log.d("ar_timer", String.format("scale up saving time:%d", endTime-startTime));
            startTime = System.currentTimeMillis();

            fs.saveFPtoFile( dirPath + "/" + r.getUuid() + "_scale_down",//i6);
                    ImageProcessor.extractRobustFeatures(tIF, ImageProcessor.scaleImage(tMat, -0.05f, 10),
                            (int)(1.2*kTemplateFPNum), kDisThd, DescriptorType.ORB, null));
            endTime = System.currentTimeMillis();
            Log.d("ar_timer", String.format("scale down saving time:%d", endTime-startTime));
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

        Log.d("match strings","enter retrieve method");

        double mr_th=0.15; //matching ratio threshold
        boolean match=false;
        Mat mat = new Mat();
        Utils.bitmapToMat(img, mat);
        Set<String> recs = rs.keySet();
        StringBuilder sb = new StringBuilder();
        //horizontal angle difference, positive stands for right perspective, negative for left perspective
        float hd = (Math.abs(cRD.z - refRD.z)>180)?(360-(cRD.z-refRD.z)) : (cRD.z-refRD.z);
        //vertical angle difference, assume the angle can't exceed 90
        float vd = cRD.x-refRD.x;
        float vo_x=0;
        float vo_y=0;
        float scale=0;

        if (Math.abs(hd) > 90 || Math.abs(vd) > 90)
            sb.append("angle difference larger than 90 degree");
        else {
            int count_r=0;
            Log.d("match strings","recognized "+Integer.toString(recognitions.size())+" items");

            MatOfDMatch m = null;
            ImageFeature qmIF = null;
            ImageFeature tmIF = null;
            for (Recognition r : recognitions) {
                double mr=0; //temporarily save the matching ratio
                if (recs.contains(r.getTitle())) {
                    BoxPosition location = r.getLocation();
                    Rect roi = new Rect(location.getLeftInt(), location.getTopInt(), location.getWidthInt(), location.getHeightInt());
                    Mat qMat = new Mat(mat, roi);
                    ImageFeature qIF = ImageProcessor.extractORBFeatures(qMat, 500);
                    List<List<ImageFeature>> tIFs = rs.get(r.getTitle());
                    List<BoxPosition> tBPs = bs.get(r.getTitle());
                    int match_idx=-1;
                    for (int i=0; i < tIFs.size(); i++) {
                        List<ImageFeature> ts = tIFs.get(i);
                        BoxPosition bp = tBPs.get(i);
                        //construct template image feature candidates
                        List<ImageFeature> ifs = new ArrayList<>();
                        float area_ratio = bp.getHeight()*bp.getWidth() / (r.getLocation().getWidth()*r.getLocation().getHeight());
                        if (hd > 0)
                            ifs.add(ts.get(1));
                        else ifs.add(ts.get(0));
                        if (vd>0)
                            ifs.add(ts.get(3));
                        else ifs.add(ts.get(2));
                        if (area_ratio > 1)
                            ifs.add(ts.get(5));
                        else ifs.add(ts.get(4));

                        ImageFeature tIF = constructTemplateFP(ifs, new float[]{Math.abs(hd)/45, Math.abs(vd)/45, Math.abs(area_ratio-1)}, kTemplateFPNum);
                        MatOfDMatch matches = ImageProcessor.matchWithRegression(qIF, tIF, 5, 300, 20);

                        double tmr = (double) matches.total() / tIF.getSize();
                        sb.append(r.getTitle() + " " + tmr + ",");
                        if (tmr > mr){
                            mr = tmr;
                            match_idx=tIFs.indexOf(ts);
                            m = matches;
                            qmIF = qIF;
                            tmIF = tIF;
                        }
                    }


                    //derive the position of the VO
                    if (mr > mr_th) {
                        match = true;
                        List<BoxPosition> bpList=bs.get(r.getTitle());
                        BoxPosition bp = bpList.get(match_idx);
                        if (bp == null) return;

                        double tmin_x, tmin_y, tmax_x, tmax_y;
                        double qmin_x, qmin_y, qmax_x, qmax_y;
                        qmax_x = qmax_y = tmax_x = tmax_y = Double.MIN_VALUE;
                        qmin_x = qmin_y = tmin_x = tmin_y = Double.MAX_VALUE;

                        List<KeyPoint> tKP = tmIF.getObjectKeypoints().toList();
                        List<KeyPoint> qKP = qmIF.getObjectKeypoints().toList();
                        //get a rectangle that can bound the matched key point
                        for (DMatch dMatch : m.toList()) {
                            KeyPoint q = qKP.get(dMatch.queryIdx);
                            KeyPoint t = tKP.get(dMatch.trainIdx);
                            if (q.pt.x > qmax_x) qmax_x = q.pt.x;
                            if (q.pt.y > qmax_y) qmax_y = q.pt.y;
                            if (t.pt.x > tmax_x) tmax_x = t.pt.x;
                            if (t.pt.y > tmax_y) tmax_y = t.pt.y;
                            if (q.pt.x < qmin_x) qmin_x = q.pt.x;
                            if (q.pt.y < qmin_y) qmin_y = q.pt.y;
                            if (t.pt.x < tmin_x) tmin_x = t.pt.x;
                            if (t.pt.y < tmin_y) tmin_y = t.pt.y;
                        }
                        float dx = (float)(imgSize.height/2 -(tmax_x+tmin_x)/2);
                        float dy = (float)(imgSize.width/2 -(tmax_y+tmin_y)/2);
                        Log.d("match string",String.format("dx:%.02f,dy:%.02f",dx,dy));
                        float r_scale = (float)((tmax_x-tmin_x) / (qmax_x-qmin_x) + (tmax_y-tmin_y) / (qmax_y-qmin_y)) / 2;
                        Log.d("match string",String.format("Scale:%.02f,%.02f\t%.02f",(tmax_x-tmin_x) / (qmax_x-qmin_x), (tmax_y-tmin_y) / (qmax_y-qmin_y),r_scale));
                        Log.d("match string",String.format("qmin:(%.02f,%.02f)\tqmax:(%.02f,%.02f)\ttmin(%.02f,%.02f)\ttmax(%.02f,%.02f)",
                                qmin_x, qmin_y, qmax_x, qmax_y, tmin_x, tmin_y, tmax_x, tmax_y));
                        float r_center_x = (float)(qmax_x+qmin_x) / 2;
                        float r_center_y = (float)(qmax_y+qmin_y) / 2;



//                        float img_center_y = (float) imgSize.width / 2;
//                        float img_center_x = (float) imgSize.height / 2;
//                        Log.d("match string",String.format("center.x:%.02f,center.y:%.02f",img_center_x,img_center_y));
//
//                        float box_center_x = bp.getLeft() + bp.getWidth() / 2;
//                        float box_center_y = bp.getTop() + bp.getHeight() / 2;
//                        float dx = img_center_x - box_center_x;
//                        float dy = img_center_y - box_center_y;
//                        Log.d("match string",String.format("dx:%.02f,dy:%.02f",dx,dy));
//                        float r_scale = (bp.getWidth() / location.getWidth() + bp.getHeight() / location.getHeight()) / 2;
//                        Log.d("match string",String.format("Scale:%.02f,%.02f\t%.02f",bp.getWidth() / location.getWidth(), bp.getHeight() / location.getHeight(),r_scale));
//
//                        float r_center_x = location.getLeft() + location.getWidth() / 2;
//                        float r_center_y = location.getTop() + location.getHeight() / 2;

                        vo_x = r_center_x + dx * r_scale;
                        vo_y = r_center_y + dy * r_scale;
                        scale = r_scale;
                        //count_r++;
                    }
                }
            }
            /*
            if(match) {
                vo_x = vo_x / count_r;
                vo_y = vo_y / count_r;
                scale= scale / count_r;
            }*/
        }
        Log.d("match strings",sb.toString());
        if(match) {

            float finalScale = scale;
            float finalVo_x = vo_x;
            float finalVo_y = vo_y;
            Log.d("match strings","scale:"+Float.toString(finalScale)+" "+Float.toString(finalVo_x)+" "+Float.toString(finalVo_y));


            runOnUiThread(() -> {
                TextView tv = findViewById(R.id.mratio);
                tv.setText(sb.toString());

//                DisplayMetrics displayMetrics = new DisplayMetrics();
//                getWindowManager().getDefaultDisplay().getMetrics(displayMetrics);
//                float width = displayMetrics.heightPixels;
//                float height = displayMetrics.widthPixels;

                float width=previewHeight;
                float height=previewWidth;
//                float width=imgSize.height;
//                float height=imgSize.width;
                float x,y,z;
                float v_dist_center_x=(float) (width/2/Math.tan(h_viewangle/2/180*Math.PI)); //virtual distance to the center of the cameraview
                float v_dist_center_y=(float) (height/2/Math.tan(v_viewangle/2/180*Math.PI)); //virtual distance to the center of the cameraview
                Log.d("match string",String.format("width:%.02f,height:%.02f",width, height));
                Log.d("match string","dist_center:"+Float.toString(v_dist_center_x)+" "+Float.toString(v_dist_center_y));
                float v_dist=v_dist_center_x;//(v_dist_center_x+v_dist_center_y)/2; //distance in units of pixels
                float v_dist_center= (float)Math.sqrt((finalVo_x -width/2)*(finalVo_x -width/2)+(finalVo_y -height/2)*(finalVo_y -height/2));
                float v_angle=(float)Math.atan(v_dist_center/v_dist);
                float x_angle= (float)Math.atan((finalVo_x -width/2)/v_dist);//x angle of the VO
                float y_angle= (float)Math.atan((finalVo_y -height/2)/v_dist);

                float dist_to_pixel= (float) (VO_dist_for_viewer * finalScale * Math.cos(v_angle) / v_dist);
                z = -v_dist*dist_to_pixel;
                x= (finalVo_x-width/2)*dist_to_pixel;
                y= (finalVo_y-height/2)*dist_to_pixel;
                Log.d("match string",String.format("before placeAndy:%.02f,%.02f,%.02f",x,y,z));
                placeAndy(x, y, z);
                onRetrieve=false;

            });
        }
    }

    static class KPoint{
        double x,y;
        boolean selected =false;
        int[] idx;

        int idx1, idx2;
        KPoint(double x, double y, int len){
            this.x=x;this.y=y;
            idx=new int[len];
            Arrays.fill(idx,-1);
        }

        KPoint(double x, double y){
            this.x=x;this.y=y;
        }
        @Override
        public boolean equals(Object obj) {
            if(!(obj instanceof KPoint)) return false;
            KPoint kobj=(KPoint) obj;
            return ((kobj.getX()==x)&&(kobj.getY()==y));
        }

        public double getX() {
            return x;
        }


        public double getY() {
            return y;
        }

        public void setIdx(int kp_list_idx, int fp_idx) {
            idx[kp_list_idx]=fp_idx;
        }


        public int getIdx(int i) {
            return idx[i];
        }



        public void setSelected(boolean selected){
            this.selected=selected;
        }



        public boolean isSelected(){ return selected;}

        public boolean isInList(int i) {
            return (idx[i]>=0);
        }

    }

    static ImageFeature constructTemplateFP(List<ImageFeature> tIFs, float[] weights, int tNum) {
        //calculate ratios
        //float hr = Math.abs(hd)/(Math.abs(hd) + Math.abs(vd));
        //float vr = Math.abs(vd)/(Math.abs(hd) + Math.abs(vd));
        //ImageFeature IF1=tIFs.get(0); //horizontal
        //ImageFeature IF2=tIFs.get(1); //vertical

        float sum=0;
        for(float f : weights) sum+=f;
        for(int i=0;i<weights.length;i++) weights[i]=weights[i]/sum;

        for (int i=0; i<weights.length; i++)
            if (weights[i]==1f)
                return (tIFs.get(i).getSize()>tNum)? tIFs.get(i).subImageFeature(0, tNum) : tIFs.get(i);

        List<KeyPoint> kp= new ArrayList<>();//(IF1.getObjectKeypoints().toList());
        Mat des = new Mat();//new Size(IF1.getDescriptors().cols(),tNum), IF1.getDescriptors().type());
        //des.push_back(IF1.getDescriptors());

        //List<KeyPoint> kp1 = IF1.getObjectKeypoints().toList();
        //List<KeyPoint> kp2 = IF2.getObjectKeypoints().toList();

        List<List<KeyPoint>> kp_list = new ArrayList();
        for(int i=0;i<tIFs.size();i++){
            kp_list.add((tIFs.get(i).getObjectKeypoints().toList()));
        }


        List<KPoint> distKPs=new ArrayList<>(); //distinct key points

        for(int i=0;i<kp_list.size();i++){
            for(int j=0;j<kp_list.get(i).size();j++){
                KeyPoint k1= kp_list.get(i).get(j);
                KPoint tkp=new KPoint(k1.pt.x, k1.pt.y, kp_list.size());
                tkp.setIdx(i,j);
                int idx=distKPs.indexOf(tkp);
                if(idx<0) {
                    distKPs.add(tkp);
                }else{
                    distKPs.get(idx).setIdx(i,j);
                }
            }
        }

        int[] c_list=new int[kp_list.size()];
        int[] p_list=new int[kp_list.size()];
        int sum_c=0;
        while( kp.size()<tNum){
            KeyPoint k;
            float max_deficit=-1;
            int candidate_idx=-1;
            for(int i=0;i<c_list.length;i++){
                int n_sum= (sum_c==0)? tNum : sum_c;
                //System.out.printf("%d:%.02f,%.02f\n",i,weights[i],(float)c_list[i]/n_sum);

                float deficit=weights[i]-(float)c_list[i]/n_sum;
                //add the feature points of list with largest deficit when it still has candidates
                if(deficit>max_deficit && kp_list.get(i).size() > p_list[i]) {
                    max_deficit=deficit;
                    candidate_idx=i;
                }
            }

            if (candidate_idx==-1) break;//can't add candidate anymore

            k=kp_list.get(candidate_idx).get(p_list[candidate_idx]++);

            KPoint kkp=new KPoint(k.pt.x,k.pt.y);
            int idx=distKPs.indexOf(kkp);
            if(idx<0) System.out.println("sth is wrong, idx<0");
            kkp=distKPs.get(idx);
            if(kkp.isSelected()){
                continue;
            }

            for(int i=0;i<c_list.length;i++){
                if(kkp.isInList(i)){
                    c_list[i]++;
                    sum_c++;
                }
            }
            kkp.setSelected(true);

            kp.add(k);

            Mat tMat=null;
            for(int i=0;i<kp_list.size();i++){
                if(kkp.isInList(i)){
                    int rowidx=kkp.getIdx(i);
                    tMat=tIFs.get(i).getDescriptors().row(rowidx);
                    break;
                    //System.out.printf("kp_idx:%d,row_idx:%d\n",i,rowidx);
                }
            }
            /*if(kkp.isInFirst()){
                tMat = IF1.getDescriptors().row(kkp.getIdx1());
            }else if(kkp.isInSecond()){
                tMat = IF2.getDescriptors().row(kkp.getIdx2());
            }else{System.out.println("sth. is wrong, not in 1 or 2");}*/
            des.push_back(tMat);
            //System.out.println(des.size().toString());
            //System.out.printf("d1:%.02f, d2:%.02f, p1:%d,p2:%d,kp:%d\n",deficit1,deficit2,p1,p2,kp.size());
        }

        MatOfKeyPoint tKP = new MatOfKeyPoint();
        tKP.fromList(kp);
        //System.out.printf("construct FP size: %d, %s\n", kp.size(),des.size().toString());
        return new ImageFeature(tKP, des, tIFs.get(0).getDescriptorType());
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
        arSession = arFragment.getArSceneView().getSession();

        float y=(float) (dist*Math.tan(Math.PI/6));
        Pose mCameraRelativePose= Pose.makeTranslation(0.0f, 0, -dist);
        //Log.d("myTag","dist:"+Float.toString(dist)+", y:"+Float.toString(y));

        Pose cPose = camera.getPose().compose(mCameraRelativePose).extractTranslation();
        Anchor anchor=arSession.createAnchor(cPose);

        //copy&paste
        AnchorNode anchorNode = new AnchorNode(anchor);
        anchorNode.setParent(arFragment.getArSceneView().getScene());

        if(andy==null) andy = new TransformableNode(arFragment.getTransformationSystem());
        placeAndy(anchorNode);
    }

    void placeAndy(float x, float y, float z){
        Camera camera=arFragment.getArSceneView().getArFrame().getCamera();
        Pose mCameraRelativePose= Pose.makeTranslation(x, y, z);
        arSession = arFragment.getArSceneView().getSession();

        Pose cPose = camera.getPose().compose(mCameraRelativePose).extractTranslation();
        Anchor anchor=arSession.createAnchor(cPose);

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
