import yaml
import argparse
import datetime

from kafka import KafkaConsumer
from flask import Flask, Response, render_template

args_parser = argparse.ArgumentParser()
args_parser.add_argument('--config', dest='config', required=True)
args = args_parser.parse_args()

with open(args.config) as conf_file:
    config = yaml.safe_load(conf_file)

consumer = KafkaConsumer(
    config["topic"], 
    bootstrap_servers = config["bootstrap_servers"])

# Set the consumer in a Flask App
app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed', methods=['GET'])
def video_feed():
    return Response(
        get_video_stream(), 
        mimetype='multipart/x-mixed-replace; boundary=frame')

def get_video_stream():
    for msg in consumer:
        yield (b'--frame\r\n'
               b'Content-Type: image/jpg\r\n\r\n' + msg.value + b'\r\n\r\n')

if __name__ == "__main__":
    app.run(host='0.0.0.0', debug=True)
