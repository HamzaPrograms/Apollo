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

LOCAL_REAL_FOLDER = os.path.join(LOCAL_VIDEO_FOLDER, 'real')
LOCAL_FAKE_FOLDER = os.path.join(LOCAL_VIDEO_FOLDER, 'fake')
FRAME_REAL_FOLDER = os.path.join(FRAME_SAVE_FOLDER, 'real')
FRAME_FAKE_FOLDER = os.path.join(FRAME_SAVE_FOLDER, 'fake')

os.makedirs(LOCAL_REAL_FOLDER, exist_ok=True)
os.makedirs(LOCAL_FAKE_FOLDER, exist_ok=True)
os.makedirs(FRAME_REAL_FOLDER, exist_ok=True)
os.makedirs(FRAME_FAKE_FOLDER, exist_ok=True)

response = s3.list_objects_v2(Bucket=BUCKET_NAME, Prefix=VIDEO_FOLDER)
# S3 paginator to get all objects
paginator = s3.get_paginator('list_objects_v2')
page_iterator = paginator.paginate(Bucket=BUCKET_NAME, Prefix=VIDEO_FOLDER)

# Process each page of results
for page in page_iterator:
    if 'Contents' in page:
        for obj in page['Contents']:
            s3_key = obj['Key']  # Full S3 key

            # Check for video files
            if s3_key.endswith('.mp4') or s3_key.endswith('.avi'):
                
                # Determine category based on S3 key
                if 'raw/real/' in s3_key:
                    local_path = os.path.join(LOCAL_REAL_FOLDER, os.path.basename(s3_key))
                    frame_save_subfolder = FRAME_REAL_FOLDER
                elif 'raw/fake/' in s3_key:
                    local_path = os.path.join(LOCAL_FAKE_FOLDER, os.path.basename(s3_key))
                    frame_save_subfolder = FRAME_FAKE_FOLDER
                else:
                    print(f"Skipping {s3_key}, unknown category.")
                    continue

                # Download video
                print(f"Downloading {s3_key}...")
                s3.download_file(BUCKET_NAME, s3_key, local_path)

                # Extract frames
                print(f"Extracting frames from {os.path.basename(s3_key)}...")
                cap = cv2.VideoCapture(local_path)
                frame_count = 0

                while True:
                    ret, frame = cap.read()
                    if not ret:
                        break
                    frame_filename = os.path.join(frame_save_subfolder, f"{os.path.basename(s3_key)}_frame_{frame_count}.jpg")
                    cv2.imwrite(frame_filename, frame)
                    frame_count += 1

                cap.release()
                print(f"Saved {frame_count} frames from {os.path.basename(s3_key)}")