import cv2
import numpy as np
from kafka import KafkaConsumer
from skimage.transform import resize
from tqdm import tqdm

from violence_detector.image_model import ViolenceDetector

class VideoConsumer:
    def __init__(self, config):
        self.consumer = KafkaConsumer(
            config["topic"], 
            bootstrap_servers = [config["bootstrap_servers"]])
        
        # Loaded pretrained weight to get model
        self.model = ViolenceDetector(config).get_model()
        self.all_frames = []
        self.num_frames_to_predict = config["num_frames_to_predict"]

    def process_frame(self, frame):
        """
        Resize video frame to (160, 160, 3)
        ----------
        `frame`: numpy.ndarray | [A video frame]
        """

        # resize
        frame = resize(frame, (160,160,3))

        # scale
        if np.max(frame) > 1:
            frame = frame / 255.0

        return frame       

    def detect_violence(self, threshold=0.9):
        """
        ----------
        num_frames_to_predict: int  | [Number of frames will pass into the model to get prediction].
        threshold: float            | [The accuracy the model must achieve if it considers the video to be violent]
        """
        for msg in tqdm(self.consumer, "Đang duyệt nội dung bạo lực..."):
            buffer = np.frombuffer(msg.value, dtype=np.uint8)
            buffer = cv2.imdecode(buffer, cv2.IMREAD_COLOR)
            buffer = self.process_frame(buffer)
            self.all_frames.append(buffer)

            if len(self.all_frames) % self.num_frames_to_predict == 0:
                _input = np.array(self.all_frames[-self.num_frames_to_predict:])

                # Change from (30, 160, 160, 3) to (1, 30, 160, 160, 3)
                _input = np.expand_dims(_input, axis=0)
                
                prediction = self.model.predict(_input)
                if prediction[0][1] >= threshold:
                    print("PHÁT HIỆN BẠO LỰC ({} %)!".format(round(prediction[0][1], 4) * 100))
                    break
                else:
                    print(prediction[0][1])
