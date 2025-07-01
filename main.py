import os
import boto3
from dotenv import load_dotenv
import cv2
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import random
from torch.utils.data import DataLoader
import torch.nn as nn
import torchvision.models as models
import torch.optim as optim

# load_dotenv()
# AWS_ACCESS_KEY = os.getenv('AWS_ACCESS_KEY')
# AWS_SECRET_KEY = os.getenv('AWS_SECRET_KEY')
# AWS_REGION = os.getenv('AWS_REGION')

# s3 = boto3.client('s3',
#                   aws_access_key_id=AWS_ACCESS_KEY,
#                   aws_secret_access_key=AWS_SECRET_KEY,
#                   region_name=AWS_REGION)

# BUCKET_NAME = 'apollo-deepfake-data'
# VIDEO_FOLDER = 'raw/'
# LOCAL_VIDEO_FOLDER = 'downloaded_videos'
# FRAME_SAVE_FOLDER = 'extracted_frames'

# LOCAL_REAL_FOLDER = os.path.join(LOCAL_VIDEO_FOLDER, 'real')
# LOCAL_FAKE_FOLDER = os.path.join(LOCAL_VIDEO_FOLDER, 'fake')
# FRAME_REAL_FOLDER = os.path.join(FRAME_SAVE_FOLDER, 'real')
# FRAME_FAKE_FOLDER = os.path.join(FRAME_SAVE_FOLDER, 'fake')

# os.makedirs(LOCAL_REAL_FOLDER, exist_ok=True)
# os.makedirs(LOCAL_FAKE_FOLDER, exist_ok=True)
# os.makedirs(FRAME_REAL_FOLDER, exist_ok=True)
# os.makedirs(FRAME_FAKE_FOLDER, exist_ok=True)

# response = s3.list_objects_v2(Bucket=BUCKET_NAME, Prefix=VIDEO_FOLDER)
# paginator = s3.get_paginator('list_objects_v2')
# page_iterator = paginator.paginate(Bucket=BUCKET_NAME, Prefix=VIDEO_FOLDER)

# for page in page_iterator:
#     if 'Contents' in page:
#         for obj in page['Contents']:
#             s3_key = obj['Key']  # Full S3 key

#             if s3_key.endswith('.mp4') or s3_key.endswith('.avi'):
                
#                 if 'raw/real/' in s3_key:
#                     local_path = os.path.join(LOCAL_REAL_FOLDER, os.path.basename(s3_key))
#                     frame_save_subfolder = FRAME_REAL_FOLDER
#                 elif 'raw/fake/' in s3_key:
#                     local_path = os.path.join(LOCAL_FAKE_FOLDER, os.path.basename(s3_key))
#                     frame_save_subfolder = FRAME_FAKE_FOLDER
#                 else:
#                     print(f"Skipping {s3_key}, unknown category.")
#                     continue
#                 print(f"Downloading {s3_key}...")
#                 s3.download_file(BUCKET_NAME, s3_key, local_path)
#                 print(f"Extracting frames from {os.path.basename(s3_key)}...")
#                 cap = cv2.VideoCapture(local_path)
#                 frame_count = 0

#                 while True:
#                     ret, frame = cap.read()
#                     if not ret:
#                         break
#                     frame_filename = os.path.join(frame_save_subfolder, f"{os.path.basename(s3_key)}_frame_{frame_count}.jpg")
#                     cv2.imwrite(frame_filename, frame)
#                     frame_count += 1

#                 cap.release()
#                 print(f"Saved {frame_count} frames from {os.path.basename(s3_key)}")

class DeepfakeFrameDataset(Dataset):
    def __init__(self, frame_root, sequence_length=10, transform=None):
        """
        Args:
            frame_root (str): Root folder containing 'real' and 'fake' subfolders.
            sequence_length (int): Number of consecutive frames per sequence.
            transform (callable, optional): Optional transform to be applied on a frame.
        """
        self.sequence_length = sequence_length
        self.transform = transform

        self.samples = []  # List of (video_id, frame_folder_path, label)

        for label, subfolder in enumerate(['real', 'fake']):  # real=0, fake=1
            subfolder_path = os.path.join(frame_root, subfolder)

            video_frames = {}
            for filename in os.listdir(subfolder_path):
                if filename.endswith('.jpg'):
                    video_id = "_".join(filename.split('_')[:2])
                    video_frames.setdefault(video_id, []).append(os.path.join(subfolder_path, filename))

            # Add samples
            for video_id, frame_paths in video_frames.items():
                frame_paths.sort()
                if len(frame_paths) >= sequence_length:
                    self.samples.append((frame_paths, label))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        frame_paths, label = self.samples[idx]

        max_start = len(frame_paths) - self.sequence_length
        start_idx = random.randint(0, max_start)

        selected_frames = frame_paths[start_idx : start_idx + self.sequence_length]

        frames = []
        for frame_path in selected_frames:
            img = Image.open(frame_path).convert('RGB')
            if self.transform:
                img = self.transform(img)
            frames.append(img)

        # Shape: (sequence_length, C, H, W)
        frames = torch.stack(frames) #Stacks frames into a pytorch tensor

        return frames, torch.tensor(label, dtype=torch.float32)


transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),  # Converts images to (C, H, W) tensors with values in [0,1]
])

dataset = DeepfakeFrameDataset(frame_root='extracted_frames', sequence_length=10, transform=transform)
dataloader = DataLoader(dataset, batch_size=4, shuffle=True)


class DeepfakeCNNLSTM(nn.Module):
    def __init__(self, hidden_dim=128, num_layers=1):
        super(DeepfakeCNNLSTM, self).__init__()

        # Pretrained CNN (ResNet18) to extract frame features
        cnn = models.resnet18(pretrained=True)
        cnn.fc = nn.Identity()  # Remove final classification layer, keep feature extractor
        self.cnn = cnn

        self.feature_dim = 512  # ResNet18 outputs 512-dimensional features

        # LSTM to process sequence of frame features
        self.lstm = nn.LSTM(input_size=self.feature_dim, hidden_size=hidden_dim, num_layers=num_layers, batch_first=True)

        # Final classifier
        self.fc = nn.Linear(hidden_dim, 1)  # Binary classification (real/fake)

    def forward(self, frames):
        """
        Args:
            frames: Tensor of shape (batch_size, sequence_length, C, H, W)
        Returns:
            logits: Tensor of shape (batch_size, 1)
        """
        batch_size, seq_len, C, H, W = frames.shape

        # Extract features for each frame
        features = []
        for i in range(seq_len):
            frame = frames[:, i, :, :, :]  # Shape: (batch_size, C, H, W)
            f = self.cnn(frame)  # Shape: (batch_size, feature_dim)
            features.append(f)

        features = torch.stack(features, dim=1)  # Shape: (batch_size, sequence_length, feature_dim)

        # Pass through LSTM
        lstm_out, _ = self.lstm(features)  # Shape: (batch_size, sequence_length, hidden_dim)

        # Use last output for classification
        last_output = lstm_out[:, -1, :]  # Shape: (batch_size, hidden_dim)
        logits = self.fc(last_output)  # Shape: (batch_size, 1)

        return logits.squeeze(1)  # Shape: (batch_size,)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

model = DeepfakeCNNLSTM().to(device)
criterion = nn.BCEWithLogitsLoss()  # Combines sigmoid + binary cross entropy
optimizer = optim.Adam(model.parameters(), lr=1e-4)

num_epochs = 10

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0

    for batch_idx, (frames, labels) in enumerate(dataloader):
        frames = frames.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(frames)  # Shape: (batch_size,)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        if batch_idx % 10 == 0:
            print(f"Processed batch {batch_idx}/{len(dataloader)}")

    avg_loss = running_loss / len(dataloader)
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}")

torch.save(model.state_dict(), 'deepfake_cnn_lstm.pth')
