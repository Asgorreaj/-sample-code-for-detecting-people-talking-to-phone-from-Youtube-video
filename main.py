import os
import argparse
from detector import Detector
import yt_dlp

# Model URL for downloading the model
modelURL = "http://download.tensorflow.org/models/object_detection/tf2/20200711/faster_rcnn_resnet50_v1_640x640_coco17_tpu-8.tar.gz"

# Function to download YouTube video
def download_youtube_video(url, output_path='./videos'):
    ydl_opts = {
        'format': 'mp4',
        'outtmpl': os.path.join(output_path, '%(title)s.%(ext)s'),
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info_dict = ydl.extract_info(url, download=True)
        video_title = info_dict.get('title', None)
        file_extension = info_dict.get('ext', 'mp4')
        file_path = os.path.join(output_path, f"{video_title}.{file_extension}")
        return file_path

# Parse command-line arguments
parser = argparse.ArgumentParser(description="Object Detection in YouTube Video")
parser.add_argument("url", type=str, help="YouTube video URL")
args = parser.parse_args()

# Download the YouTube video
videoPath = download_youtube_video(args.url, output_path='./videos')

# Paths to necessary files
classFile = os.path.join(os.path.dirname(__file__), 'coco.names')
threshold = 0.5

# Create a detector object and perform detection on the downloaded video
detector = Detector()
detector.readClasses(classFile)
detector.downloadModel(modelURL)
detector.loadModel()
detector.predictVideo(videoPath, threshold)
