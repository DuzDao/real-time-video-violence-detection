import numpy as np
import tensorflow as tf
import cv2 

class BloodDetector:
    def __init__(self, model_path):
        print("Loading model for blood detector...")
        self.model = tf.keras.models.load_model(model_path)

    def process_frame1(self, frame):
        """
        Resize video frame to (224, 224) for detect blood on (video) frames.
        ----------
        `frame`: numpy.ndarray | [A video frame]
        """

        # resize
        frame = cv2.resize(frame, (224, 224))

        # scale
        if np.max(frame) > 1:
            frame = frame / 255.0

        return frame 

    def detect_blood(self, frame):
        frame = self.process_frame1(frame)
        frame = np.expand_dims(frame, axis=0)
        prediction = self.model.predict(frame)[0][0]

        if prediction > 0.9:
            return True
        else:
            return False

