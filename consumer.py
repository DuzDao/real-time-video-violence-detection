import yaml
import argparse

from kafka_utils.video_producer import VideoProducer
from kafka_utils.video_consumer import VideoConsumer

def main(args):
    with open(args.config) as conf_file:
        config = yaml.safe_load(conf_file)
    
    consumer = VideoConsumer(config, args.video_path)
    consumer.detect_hate_speech()
    consumer.detect_violence_and_blood()

if __name__ == "__main__":
    args_parser = argparse.ArgumentParser()
    args_parser.add_argument('--config', dest='config', required=True)
    args_parser.add_argument('--video', dest="video_path")
    args = args_parser.parse_args()
    main(args)
