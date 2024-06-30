import yaml
import argparse
from kafka_utils.video_producer import VideoProducer

def main(args):
    with open(args.config) as conf_file:
        config = yaml.safe_load(conf_file)
    
    producer = VideoProducer(config)
    producer.publish_from_video(args.video_path)

if __name__ == "__main__":
    args_parser = argparse.ArgumentParser()
    args_parser.add_argument('--config', dest='config', required=True)
    args_parser.add_argument('--video', dest='video_path', required=True)

    args = args_parser.parse_args()
    main(args)
