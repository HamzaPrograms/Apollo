import os
import boto3
from dotenv import load_dotenv
import cv2

load_dotenv()
AWS_ACCESS_KEY = os.getenv('AWS_ACCESS_KEY')
AWS_SECRET_KEY = os.getenv('AWS_SECRET_KEY')
AWS_REGION = os.getenv('AWS_REGION')

s3 = boto3.client('s3',
                  aws_access_key_id=AWS_ACCESS_KEY,
                  aws_secret_access_key=AWS_SECRET_KEY,
                  region_name=AWS_REGION)

BUCKET_NAME = 'apollo-deepfake-data'
VIDEO_FOLDER = 'raw/'
LOCAL_VIDEO_FOLDER = 'downloaded_videos'
FRAME_SAVE_FOLDER = 'extracted_frames'

s3 = boto3.client('s3')

os.makedirs(LOCAL_VIDEO_FOLDER, exist_ok=True)
os.makedirs(FRAME_SAVE_FOLDER, exist_ok=True)

response = s3.list_objects_v2(Bucket=BUCKET_NAME, Prefix=VIDEO_FOLDER)

if 'Contents' in response:
    for obj in response['Contents']:
        s3_key = obj['Key'] #full path of S3 file
        
        if s3_key.endswith('.mp4') or s3_key.endswith('.avi'):
            filename = os.path.basename(s3_key)
            local_path = os.path.join(LOCAL_VIDEO_FOLDER, filename)#where to save vid locally
            print(f"Downloading {s3_key}...")
            s3.download_file(BUCKET_NAME, s3_key, local_path)
            print(f"Extracting frames from {filename}...")
            cap = cv2.VideoCapture(local_path)#extract frames
            frame_count = 0

            while True:
                ret, frame = cap.read()#ret = true if saved, frame is read
                if not ret:
                    break

                frame_filename = os.path.join(FRAME_SAVE_FOLDER, f"{filename}_frame_{frame_count}.jpg")
                cv2.imwrite(frame_filename, frame)#save frame to location frame_filename
                frame_count += 1

            cap.release()
            print(f"Saved {frame_count} frames from {filename}")

else:
    print("No video files found in the specified S3 folder.")
