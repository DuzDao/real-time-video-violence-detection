import cv2
import time
import numpy as np
from skimage.transform import resize

from kafka import KafkaProducer

class VideoProducer:
    def __init__(self, config):
        self.bootstrap_servers = config["bootstrap_servers"]
        self.topic = config["topic"]
        self.interval = config["interval"]
        self.producer = KafkaProducer(bootstrap_servers = self.bootstrap_servers)


    def produce(self, frame):
        """
        Encode & produce frame.
        ----------
        `frame`: numpy.ndarray | [A video frame]
        """
        frame = self.process_frame(frame)

        _, buffer = cv2.imencode('.jpg', frame)

        self.producer.send(self.TOPIC, buffer.tobytes())

        time.sleep(self.interval)


    def publish_from_video(self, mp4_path):
        """
        Read video frame, process frame & produce.
        ----------
        `mp4_path`: str     | [Path of mp4 video]
        """
        video = cv2.VideoCapture(mp4_path, cv2.CAP_DSHOW)
        
        while video.isOpened():
            success, frame = video.read()

            if not success:
                print("Read video not success!")
                break
            
            self.produce(frame)
        video.release()

        
    def process_frame(self, frame):
        """
        Resize video frame to (1, 160, 160, 3)
        ----------
        `frame`: numpy.ndarray | [A video frame]
        """

        # resize
        frame = resize(frame, (160,160,3))  
        frame = np.expand_dims(frame, axis=0)

        # scale
        frame = frame / 255.0
        return frame            
