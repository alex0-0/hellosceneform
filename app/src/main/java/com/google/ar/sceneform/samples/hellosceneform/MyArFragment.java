package com.google.ar.sceneform.samples.hellosceneform;

import android.app.Activity;
import android.graphics.Bitmap;
import android.media.Image;
import android.util.Log;
import android.view.MotionEvent;

import com.google.ar.core.Frame;
import com.google.ar.sceneform.ArSceneView;
import com.google.ar.sceneform.FrameTime;
import com.google.ar.sceneform.HitTestResult;
import com.google.ar.sceneform.Scene;
import com.google.ar.sceneform.samples.hellosceneform.HelloSceneformActivity;
import com.google.ar.sceneform.ux.ArFragment;

public class MyArFragment extends ArFragment {
    private HelloSceneformActivity activity=null;
    private float preTime=0;

    FrameListener listener = null;

    public interface FrameListener {
        void onFrame(FrameTime frameTime, Frame frame);
    }

    public void setOnFrameListener(FrameListener listener) {
        this.listener = listener;
    }

    @Override
    public void onPeekTouch(HitTestResult hitTestResult, MotionEvent motionEvent) {
        super.onPeekTouch(hitTestResult, motionEvent);
        if((hitTestResult.getNode()!=null)&&(activity!=null)) activity.onPeekTouch();
    }

    void setActivity(HelloSceneformActivity a){
        activity=a;
    }

    @Override
    public void onUpdate(FrameTime frameTime){
        /*** add a listener ***/

        super.onUpdate(frameTime);

        Frame arFrame = getArSceneView().getArFrame();
        if (listener != null) {
            listener.onFrame(frameTime, arFrame);
        }


    }
}
