package edu.umb.cs.imageprocessinglib.tensorflow;

import android.graphics.Bitmap;
import java.util.List;
import edu.umb.cs.imageprocessinglib.model.Recognition;

/**
 * Generic interface for interacting with different recognition engines.
 */
public interface Classifier {
    List<Recognition> recognizeImage(Bitmap bitmap);

    void enableStatLogging(final boolean debug);

    String getStatString();

    void close();
}
