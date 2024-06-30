import cv2
import time
import numpy as np

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
        _, buffer = cv2.imencode('.jpg', frame)
        
        self.producer.send(self.topic, buffer.tobytes())


    def publish_from_video(self, mp4_path):
        """
        Read video frame, process frame & produce.
        ----------
        `mp4_path`: str     | [Path of mp4 video]
        """
        video = cv2.VideoCapture(mp4_path)
        
        frame_cnt = 0

        while video.isOpened():
            success, frame = video.read()
            
            if not success:
                print("Read video not success!")
                break
            else:
                frame_cnt += 1
                print("Frame {} sended!".format(frame_cnt))
                self.produce(frame)

            # Interval    
            time.sleep(self.interval)
        
        video.release()         
