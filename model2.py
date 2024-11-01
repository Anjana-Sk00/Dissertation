import tensorflow as tf
device_name = tf.test.gpu_device_name()
print(f'Using GPU: {device_name}')

import tensorflow as tf

print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))

!pip install opencv-python-headless
!pip install scikit-learn
!pip install tensorflow
!pip install numpy
!pip install matplotlib

!pip install --upgrade opencv-python-headless
!pip install --upgrade scikit-learn
!pip install --upgrade tensorflow
!pip install --upgrade matplotlib

from google.colab import drive
drive.mount('/content/drive')

import tensorflow as tf

# Check TensorFlow version
print(f"TensorFlow version: {tf.__version__}")

# Check if GPU is available
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

# Ensure TensorFlow is set to run on GPU
device_name = tf.test.gpu_device_name()
if not device_name:
    raise SystemError('GPU device not found')
print('Found GPU at: {}'.format(device_name))

# Enable mixed precision training
from tensorflow.keras import mixed_precision

# Set the policy to mixed_float16
policy = mixed_precision.Policy('mixed_float16')
mixed_precision.set_global_policy(policy)

# Verify the policy
print("Mixed precision policy:", mixed_precision.global_policy())

import os

processed_faces_path = '/content/extracted_frames/'
flow_save_path = '/content/optical_flows/'

os.makedirs(processed_faces_path, exist_ok=True)
os.makedirs(flow_save_path, exist_ok=True)

print("Folders created locally in Colab!")

import zipfile
import os

zipped_files_path = '/content/drive/My Drive/'

unzip_folder = '/content/ravdess_videos/'
os.makedirs(unzip_folder, exist_ok=True)

def unzip_ravdess_files():
    for i in range(1, 25):
        zip_file_path = os.path.join(zipped_files_path, f'ravdess_video_actor_{i:02}.zip')
        with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
            zip_ref.extractall(unzip_folder)
        print(f'Unzipped: {zip_file_path}')

unzip_ravdess_files()

#code for frame extraction
import os
import cv2

emotion_dict = {
    '01': 'neutral',
    '02': 'calm',
    '03': 'happy',
    '04': 'sad',
    '05': 'angry',
    '06': 'fearful',
    '07': 'disgust',
    '08': 'surprised'
}

colab_unzip_path = '/content/ravdess_videos/'  
processed_faces_path = '/content/extracted_frames/' 

os.makedirs(processed_faces_path, exist_ok=True)

def extract_frames_from_video(video_file, actor_id, video_id, emotion_label, frame_rate=2):
    cap = cv2.VideoCapture(video_file)
    frame_count = 0
    saved_frame_count = 0

    actor_folder = os.path.join(processed_faces_path, f"Actor_{actor_id}")
    video_folder = os.path.join(actor_folder, f"video_{video_id}_{emotion_label}")  
    os.makedirs(video_folder, exist_ok=True)  

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break  

        if frame_count % frame_rate == 0:
            frame_filename = f"frame_{saved_frame_count}.jpg"
            frame_path = os.path.join(video_folder, frame_filename)
            if cv2.imwrite(frame_path, frame):
                if saved_frame_count % 100 == 0: 
                    print(f"Saved {frame_filename}")
                saved_frame_count += 1
            else:
                print(f"Failed to save {frame_filename}")

        frame_count += 1 

    cap.release()
    print(f"Extracted {saved_frame_count} frames from {video_file}")

for root, dirs, files in os.walk(colab_unzip_path):
    for file in files:
        if file.endswith('.mp4'):
            video_file_path = os.path.join(root, file)

            try:
                
                actor_id = file.split('-')[6]  
                video_id = file.split('-')[4]  

                emotion_code = file.split('-')[2]
                emotion_label = emotion_dict.get(emotion_code, 'unknown')  

                if emotion_label == 'unknown':
                    raise ValueError(f"Invalid emotion code in {file}")

                print(f"Processing {file} for Actor {actor_id}, Video {video_id}")

                extract_frames_from_video(video_file_path, actor_id, video_id, emotion_label)

            except (IndexError, ValueError, cv2.error) as e:
                print(f"Failed to process {file}. Skipping... Error: {str(e)}")

import os

processed_faces_path = '/content/extracted_frames/'

def count_total_frames(directory):
    total_frames = 0

    for root, dirs, files in os.walk(directory):
       
        jpg_files = [f for f in files if f.endswith('.jpg')]
        total_frames += len(jpg_files)

    return total_frames

total_frames_extracted = count_total_frames(processed_faces_path)

print(f"Total number of frames extracted: {total_frames_extracted}")

import os
import cv2
import numpy as np

def compute_optical_flow(frames):
    optical_flows = []

    for i in range(len(frames) - 1):
        prev_frame = frames[i]
        next_frame = frames[i + 1]

        # Convert frames to grayscale
        prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
        next_gray = cv2.cvtColor(next_frame, cv2.COLOR_BGR2GRAY)

        # Compute optical flow using Farneback method
        flow = cv2.calcOpticalFlowFarneback(prev_gray, next_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)

        # Convert flow to magnitude and angle
        magnitude, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])

        # Normalize magnitude
        magnitude = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX)
        magnitude = np.uint8(magnitude)

        # Append the optical flow magnitude to the list
        optical_flows.append(magnitude)

    return optical_flows

# Function to load frames from a video directory
def load_frames_from_directory(video_folder):
    frames = []
    frame_files = sorted([f for f in os.listdir(video_folder) if f.endswith('.jpg')])

    # Load each frame in order
    for file_name in frame_files:
        frame_path = os.path.join(video_folder, file_name)
        frame = cv2.imread(frame_path)
        if frame is not None:
            frames.append(frame)

    return frames

# Function to compute optical flow for all videos in the dataset
def compute_optical_flow_for_all_videos(extracted_frames_path, save_optical_flow=True):
    for actor_folder in os.listdir(extracted_frames_path):
        actor_path = os.path.join(extracted_frames_path, actor_folder)

        if os.path.isdir(actor_path):  # Check if it's a directory
            for video_folder in os.listdir(actor_path):
                video_path = os.path.join(actor_path, video_folder)

                if os.path.isdir(video_path):
                    print(f"Processing optical flow for {actor_folder}/{video_folder}")

                    # Load frames from the video directory
                    frames = load_frames_from_directory(video_path)

                    # Compute optical flow if there are enough frames
                    if len(frames) >= 2:
                        optical_flows = compute_optical_flow(frames)
                        print(f"Computed {len(optical_flows)} optical flow frames.")

                        # Optionally, save optical flow frames
                        if save_optical_flow:
                            save_optical_flow_frames(optical_flows, actor_folder, video_folder)
                    else:
                        print(f"Not enough frames to compute optical flow for {actor_folder}/{video_folder}")

# Function to save optical flow frames (optional)
def save_optical_flow_frames(optical_flows, actor_folder, video_folder):
    # Directory to save computed optical flow frames
    optical_flow_save_path = f'/content/optical_flow_frames/{actor_folder}/{video_folder}/'
    os.makedirs(optical_flow_save_path, exist_ok=True)

    for i, flow in enumerate(optical_flows):
        save_path = os.path.join(optical_flow_save_path, f'flow_frame_{i}.png')
        cv2.imwrite(save_path, flow)
        print(f"Saved optical flow frame {i} at {save_path}")

# Path to the extracted frames directory
extracted_frames_path = '/content/extracted_frames/'

# Run the process to compute optical flow for all videos
compute_optical_flow_for_all_videos(extracted_frames_path)

import os

# Path to the optical flow frames directory
optical_flow_frames_dir = '/content/optical_flow_frames/'  # Update the path if needed

def count_total_frames_in_directory(directory):
    total_frames = 0

    # Walk through all the directories and subdirectories
    for root, dirs, files in os.walk(directory):
        # Count the number of image files in each directory
        frame_files = [f for f in files if f.endswith(('.jpg', '.png'))]  # Adjust extensions as needed
        total_frames += len(frame_files)

    return total_frames

# Count total frames in the optical flow frames directory
total_frames = count_total_frames_in_directory(optical_flow_frames_dir)
print(f"Total number of frames in optical flow folder: {total_frames}")

import os

# Paths to your frame and optical flow directories
base_frame_dir = '/content/extracted_frames/'  
base_optical_flow_dir = '/content/optical_flow_frames/'  

# Function to synchronize frame and optical flow counts by removing excess files
def synchronize_frame_flow_counts(frame_dir, optical_flow_dir):
    frame_files = sorted([f for f in os.listdir(frame_dir) if f.endswith('.jpg')])
    optical_flow_files = sorted([f for f in os.listdir(optical_flow_dir) if f.endswith('.png')])

    frame_count = len(frame_files)
    optical_flow_count = len(optical_flow_files)

    removed_frame_count = 0
    removed_optical_flow_count = 0

    # Handle excess frames
    if frame_count > optical_flow_count:
        excess_frames = frame_files[optical_flow_count:]  
        removed_frame_count = len(excess_frames)
        for frame in excess_frames:
            os.remove(os.path.join(frame_dir, frame))  

    # Handle excess optical flow frames
    if optical_flow_count > frame_count:
        excess_optical_flows = optical_flow_files[frame_count:]  
        removed_optical_flow_count = len(excess_optical_flows)
        for flow in excess_optical_flows:
            os.remove(os.path.join(optical_flow_dir, flow))  

    # Print how many files were removed
    print(f"Removed {removed_frame_count} frames and {removed_optical_flow_count} optical flow files from {frame_dir} and {optical_flow_dir}")

    # Return updated counts
    return frame_count - removed_frame_count, optical_flow_count - removed_optical_flow_count

# Loop through all actor directories
for actor in os.listdir(base_frame_dir):
    actor_frame_dir = os.path.join(base_frame_dir, actor)
    actor_optical_flow_dir = os.path.join(base_optical_flow_dir, actor)

    if os.path.isdir(actor_frame_dir) and os.path.isdir(actor_optical_flow_dir):
        # Loop through each video directory
        for video in os.listdir(actor_frame_dir):
            video_frame_dir = os.path.join(actor_frame_dir, video)
            video_optical_flow_dir = os.path.join(actor_optical_flow_dir, video)

            if os.path.isdir(video_frame_dir) and os.path.isdir(video_optical_flow_dir):
                frame_count, optical_flow_count = synchronize_frame_flow_counts(video_frame_dir, video_optical_flow_dir)

                # Print results after synchronization
                print(f"{actor}/{video}: {frame_count} frames, {optical_flow_count} optical flow frames (synchronized)")

import os

# Paths to your frame and optical flow directories
base_frame_dir = '/content/extracted_frames/'  # Root directory for frames
base_optical_flow_dir = '/content/optical_flow_frames/'  # Root directory for optical flow frames

# Function to synchronize frame and optical flow counts by removing excess files
def synchronize_frame_flow_counts(frame_dir, optical_flow_dir):
    frame_files = sorted([f for f in os.listdir(frame_dir) if f.endswith('.jpg')])
    optical_flow_files = sorted([f for f in os.listdir(optical_flow_dir) if f.endswith('.png')])

    frame_count = len(frame_files)
    optical_flow_count = len(optical_flow_files)

    removed_frame_count = 0
    removed_optical_flow_count = 0

    # Handle excess frames
    if frame_count > optical_flow_count:
        excess_frames = frame_files[optical_flow_count:]  # Extra frames that don't have matching optical flow
        removed_frame_count = len(excess_frames)
        for frame in excess_frames:
            os.remove(os.path.join(frame_dir, frame))  # Remove excess frames

    # Handle excess optical flow frames
    if optical_flow_count > frame_count:
        excess_optical_flows = optical_flow_files[frame_count:]  # Extra optical flow files without matching frames
        removed_optical_flow_count = len(excess_optical_flows)
        for flow in excess_optical_flows:
            os.remove(os.path.join(optical_flow_dir, flow))  # Remove excess optical flow files

    # Print how many files were removed
    print(f"Removed {removed_frame_count} frames and {removed_optical_flow_count} optical flow files from {frame_dir} and {optical_flow_dir}")

    # Return updated counts
    return frame_count - removed_frame_count, optical_flow_count - removed_optical_flow_count

# Loop through all actor directories
for actor in os.listdir(base_frame_dir):
    actor_frame_dir = os.path.join(base_frame_dir, actor)
    actor_optical_flow_dir = os.path.join(base_optical_flow_dir, actor)

    if os.path.isdir(actor_frame_dir) and os.path.isdir(actor_optical_flow_dir):
        # Loop through each video directory
        for video in os.listdir(actor_frame_dir):
            video_frame_dir = os.path.join(actor_frame_dir, video)
            video_optical_flow_dir = os.path.join(actor_optical_flow_dir, video)

            if os.path.isdir(video_frame_dir) and os.path.isdir(video_optical_flow_dir):
                frame_count, optical_flow_count = synchronize_frame_flow_counts(video_frame_dir, video_optical_flow_dir)

                # Print results after synchronization
                print(f"{actor}/{video}: {frame_count} frames, {optical_flow_count} optical flow frames (synchronized)")

import cv2
import os
from google.colab.patches import cv2_imshow

# Define paths to the directories
frame_dir = '/content/extracted_frames/Actor_01.mp4/video_02_angry'
optical_flow_dir = '/content/optical_flow_frames/Actor_01.mp4/video_02_angry'

# Get the list of frames and optical flow images
frame_files = sorted([f for f in os.listdir(frame_dir) if f.endswith('.jpg')])
optical_flow_files = sorted([f for f in os.listdir(optical_flow_dir) if f.endswith('.png')])

# Ensure lists are of the same length
if len(frame_files) != len(optical_flow_files):
    print("Mismatch in the number of frames and optical flow images.")
else:
    # Loop over the files and display them
    for frame_file, flow_file in zip(frame_files, optical_flow_files):
        # Load the frame and optical flow image
        frame_image_path = os.path.join(frame_dir, frame_file)
        optical_flow_image_path = os.path.join(optical_flow_dir, flow_file)

        frame_image = cv2.imread(frame_image_path)
        optical_flow_image = cv2.imread(optical_flow_image_path)

        # Check if images are loaded properly
        if frame_image is None:
            print(f"Failed to load frame: {frame_file}")
            continue
        if optical_flow_image is None:
            print(f"Failed to load optical flow: {flow_file}")
            continue

        # Convert images from BGR to RGB for display
        frame_image_rgb = cv2.cvtColor(frame_image, cv2.COLOR_BGR2RGB)
        optical_flow_image_rgb = cv2.cvtColor(optical_flow_image, cv2.COLOR_BGR2RGB)

        # Display the images side by side
        print(f"Displaying: {frame_file} and {flow_file}")
        cv2_imshow(frame_image_rgb)
        cv2_imshow(optical_flow_image_rgb)

import cv2
import numpy as np
import os

# Function to load frames and optical flow data for a single video
def load_frame_and_flow_data(frame_dir, flow_dir, img_size=(224, 224)):
    frames = []
    flows = []

    # Sorted frame and flow files (ensuring matching pairs)
    frame_files = sorted([f for f in os.listdir(frame_dir) if f.endswith('.jpg')])
    flow_files = sorted([f for f in os.listdir(flow_dir) if f.endswith('.png')])

    # Load frame images
    for frame_file in frame_files:
        frame_path = os.path.join(frame_dir, frame_file)
        frame = cv2.imread(frame_path)
        if frame is not None:
            frame = cv2.resize(frame, img_size)  # Resize to target size
            frames.append(frame)

    # Load optical flow images
    for flow_file in flow_files:
        flow_path = os.path.join(flow_dir, flow_file)
        flow = cv2.imread(flow_path)
        if flow is not None:
            flow = cv2.resize(flow, img_size)  # Resize to target size
            flows.append(flow)

    # Convert lists to numpy arrays
    frames = np.array(frames)
    flows = np.array(flows)

    # Normalize pixel values
    frames = frames / 255.0
    flows = flows / 255.0

    return frames, flows

# Function to process data in batches to avoid memory overload
def process_data_in_batches(base_frame_dir, base_flow_dir, img_size=(224, 224)):
    # Loop through each actor
    for actor in os.listdir(base_frame_dir):
        actor_frame_dir = os.path.join(base_frame_dir, actor)
        actor_flow_dir = os.path.join(base_flow_dir, actor)

        if os.path.isdir(actor_frame_dir) and os.path.isdir(actor_flow_dir):
            # Loop through each video for this actor
            for video in os.listdir(actor_frame_dir):
                video_frame_dir = os.path.join(actor_frame_dir, video)
                video_flow_dir = os.path.join(actor_flow_dir, video)

                if os.path.isdir(video_frame_dir) and os.path.isdir(video_flow_dir):
                    # Load frames and flows for this video
                    frames, flows = load_frame_and_flow_data(video_frame_dir, video_flow_dir, img_size)

                    # Here you can train your model on this batch or process the data
                    print(f"Processing {actor}/{video}: {frames.shape[0]} frames and {flows.shape[0]} flow images")

                    # After processing, clear the data from memory (optional)
                    del frames, flows

# Paths to directories
base_frame_dir = '/content/extracted_frames'
base_flow_dir = '/content/optical_flow_frames'

# Process data in batches to prevent memory overload
process_data_in_batches(base_frame_dir, base_flow_dir)

import cv2
import numpy as np
import os
import gc  # For explicit garbage collection

# Function to load frames and optical flow data for a single video
def load_frame_and_flow_data(frame_dir, flow_dir, img_size=(224, 224)):
    frames = []
    flows = []

    # Sorted frame and flow files (ensuring matching pairs)
    frame_files = sorted([f for f in os.listdir(frame_dir) if f.endswith('.jpg')])
    flow_files = sorted([f for f in os.listdir(flow_dir) if f.endswith('.png')])

    # Load frame images
    for frame_file in frame_files:
        frame_path = os.path.join(frame_dir, frame_file)
        frame = cv2.imread(frame_path)
        if frame is not None:
            frame = cv2.resize(frame, img_size)  # Resize to target size
            frames.append(frame)

    # Load optical flow images
    for flow_file in flow_files:
        flow_path = os.path.join(flow_dir, flow_file)
        flow = cv2.imread(flow_path)
        if flow is not None:
            flow = cv2.resize(flow, img_size)  # Resize to target size
            flows.append(flow)

    # Convert lists to numpy arrays
    frames = np.array(frames)
    flows = np.array(flows)

    # Normalize pixel values
    frames = frames / 255.0
    flows = flows / 255.0

    return frames, flows

# Function to combine frame and optical flow data
def combine_frame_and_flow_data(frames, flows):
    combined_data = np.concatenate((frames, flows), axis=-1)  # Concatenate along the channel axis
    return combined_data

# Function to process data in batches to avoid memory overload
def process_data_in_batches(base_frame_dir, base_flow_dir, img_size=(224, 224)):
    # Loop through each actor
    for actor in os.listdir(base_frame_dir):
        actor_frame_dir = os.path.join(base_frame_dir, actor)
        actor_flow_dir = os.path.join(base_flow_dir, actor)

        if os.path.isdir(actor_frame_dir) and os.path.isdir(actor_flow_dir):
            # Loop through each video for this actor
            for video in os.listdir(actor_frame_dir):
                video_frame_dir = os.path.join(actor_frame_dir, video)
                video_flow_dir = os.path.join(actor_flow_dir, video)

                if os.path.isdir(video_frame_dir) and os.path.isdir(video_flow_dir):
                    # Load frames and flows for this video
                    frames, flows = load_frame_and_flow_data(video_frame_dir, video_flow_dir, img_size)

                    # Combine frames and flows
                    combined_data = combine_frame_and_flow_data(frames, flows)

                    # Here you can train your model on this batch or process the data
                    print(f"Processing {actor}/{video}: {frames.shape[0]} frames and {flows.shape[0]} flow images (combined shape: {combined_data.shape})")

                    # After processing, clear the data from memory
                    del frames, flows, combined_data
                    gc.collect()  # Explicit garbage collection to free up memory

                    # Optional: Check memory usage (for Colab)
                    os.system('free -h')

# Paths to directories
base_frame_dir = '/content/extracted_frames'
base_flow_dir = '/content/optical_flow_frames'

# Process data in batches to prevent memory overload
process_data_in_batches(base_frame_dir, base_flow_dir)

from collections import defaultdict

# Function to check the balance of the dataset based on emotion labels in video names
def check_dataset_balance(base_frame_dir):
    emotion_counts = defaultdict(int)

    # Loop through actors
    for actor in os.listdir(base_frame_dir):
        actor_frame_dir = os.path.join(base_frame_dir, actor)

        if os.path.isdir(actor_frame_dir):
            # Loop through each video
            for video in os.listdir(actor_frame_dir):
                # Extract emotion from video name (assuming it follows the pattern: video_XX_emotion)
                emotion = video.split('_')[-1]
                emotion_counts[emotion] += 1

    return emotion_counts

# Check dataset balance
emotion_counts = check_dataset_balance(base_frame_dir)

# Display results
print("Dataset balance by emotion:")
for emotion, count in emotion_counts.items():
    print(f"{emotion}: {count} videos")

import cv2
import numpy as np
import os
import gc  # For explicit garbage collection
from sklearn.model_selection import train_test_split
import shutil  # For copying files during the split

# Function to load frames and optical flow data for a single video
def load_frame_and_flow_data(frame_dir, flow_dir, img_size=(224, 224)):
    frames = []
    flows = []

    # Sorted frame and flow files (ensuring matching pairs)
    frame_files = sorted([f for f in os.listdir(frame_dir) if f.endswith('.jpg')])
    flow_files = sorted([f for f in os.listdir(flow_dir) if f.endswith('.png')])

    # Load frame images
    for frame_file in frame_files:
        frame_path = os.path.join(frame_dir, frame_file)
        frame = cv2.imread(frame_path)
        if frame is not None:
            frame = cv2.resize(frame, img_size)  # Resize to target size
            frames.append(frame)

    # Load optical flow images
    for flow_file in flow_files:
        flow_path = os.path.join(flow_dir, flow_file)
        flow = cv2.imread(flow_path)
        if flow is not None:
            flow = cv2.resize(flow, img_size)  # Resize to target size
            flows.append(flow)

    # Convert lists to numpy arrays
    frames = np.array(frames)
    flows = np.array(flows)

    # Normalize pixel values
    frames = frames / 255.0
    flows = flows / 255.0

    return frames, flows

# Function to combine frame and optical flow data
def combine_frame_and_flow_data(frames, flows):
    # Ensure that both arrays have the same shape
    if frames.shape != flows.shape:
        raise ValueError("Frames and flows must have the same shape for concatenation.")

    # Average the frames and flows to reduce the combined data to 3 channels
    combined_data = (frames + flows) / 2.0

    return combined_data

# Function to save combined data to disk
def save_combined_data(combined_data, output_dir, actor, video, img_size, subset):
    # Create the directory based on the subset (train/test)
    actor_output_dir = os.path.join(output_dir, subset, actor)
    video_output_dir = os.path.join(actor_output_dir, video)

    os.makedirs(video_output_dir, exist_ok=True)

    # Save each combined frame and flow image
    for i, combined_image in enumerate(combined_data):
        save_path = os.path.join(video_output_dir, f'combined_{i}.jpg')
        combined_image_resized = cv2.resize(combined_image, img_size)
        cv2.imwrite(save_path, (combined_image_resized * 255).astype(np.uint8))

# Function to process data in batches, combine, and split into train and test
def process_and_split_data(base_frame_dir, base_flow_dir, output_dir, img_size=(224, 224), train_ratio=0.8):
    # Loop through each actor
    for actor in os.listdir(base_frame_dir):
        actor_frame_dir = os.path.join(base_frame_dir, actor)
        actor_flow_dir = os.path.join(base_flow_dir, actor)

        if os.path.isdir(actor_frame_dir) and os.path.isdir(actor_flow_dir):
            # Get all videos for this actor
            videos = os.listdir(actor_frame_dir)

            # Split videos into train and test sets
            train_videos, test_videos = train_test_split(videos, train_size=train_ratio)

            # Process train videos
            for video in train_videos:
                video_frame_dir = os.path.join(actor_frame_dir, video)
                video_flow_dir = os.path.join(actor_flow_dir, video)

                if os.path.isdir(video_frame_dir) and os.path.isdir(video_flow_dir):
                    # Load frames and flows for this video
                    frames, flows = load_frame_and_flow_data(video_frame_dir, video_flow_dir, img_size)

                    # Combine frames and flows
                    combined_data = combine_frame_and_flow_data(frames, flows)

                    # Save the combined data to the train directory
                    save_combined_data(combined_data, output_dir, actor, video, img_size, subset='train')

                    # Clear the data from memory
                    del frames, flows, combined_data
                    gc.collect()  # Explicit garbage collection to free up memory

            # Process test videos
            for video in test_videos:
                video_frame_dir = os.path.join(actor_frame_dir, video)
                video_flow_dir = os.path.join(actor_flow_dir, video)

                if os.path.isdir(video_frame_dir) and os.path.isdir(video_flow_dir):
                    # Load frames and flows for this video
                    frames, flows = load_frame_and_flow_data(video_frame_dir, video_flow_dir, img_size)

                    # Combine frames and flows
                    combined_data = combine_frame_and_flow_data(frames, flows)

                    # Save the combined data to the test directory
                    save_combined_data(combined_data, output_dir, actor, video, img_size, subset='test')

                    # Clear the data from memory
                    del frames, flows, combined_data
                    gc.collect()  # Explicit garbage collection to free up memory

                    # Optional: Check memory usage (for Colab)
                    os.system('free -h')

# Paths to directories
base_frame_dir = '/content/extracted_frames'
base_flow_dir = '/content/optical_flow_frames'
output_combined_dir = '/content/combined_frames_flows'

# Process and split data into train and test sets
process_and_split_data(base_frame_dir, base_flow_dir, output_combined_dir)

#code for data preparation and training model
import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import TimeDistributed, Conv2D, MaxPooling2D, Flatten, GRU, Dense, Dropout, BatchNormalization, LeakyReLU
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.regularizers import l2
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.utils.class_weight import compute_class_weight

# Data preparation function
def load_and_preprocess_image(path, target_size=(224, 224)):
    image = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    image = cv2.resize(image, target_size)
    image = image / 255.0  # Normalize the pixel values
    return image

# Define the emotion classes and create a class_indices mapping
emotion_classes = ['angry', 'calm', 'disgust', 'fearful', 'happy', 'neutral', 'sad', 'surprised']
class_indices = {label: idx for idx, label in enumerate(emotion_classes)}

# Get all image paths and labels for training and validation
def get_image_paths_and_labels(directory):
    image_paths = []
    labels = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith('.jpg'):
                image_paths.append(os.path.join(root, file))
                emotion_label = root.split(os.sep)[-1].split('_')[-1]
                labels.append(class_indices[emotion_label])
    return image_paths, labels

# Generator function with data augmentation
def data_generator(image_paths, labels, batch_size, time_steps=5, target_size=(224, 224)):
    num_classes = len(class_indices)
    datagen = ImageDataGenerator(
        rotation_range=30,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        brightness_range=[0.8, 1.2],  # Added brightness augmentation
        fill_mode='nearest'
    )
    while True:
        for start_idx in range(0, len(image_paths) - batch_size * time_steps, batch_size * time_steps):
            batch_image_paths = image_paths[start_idx:start_idx + batch_size * time_steps]
            batch_labels = labels[start_idx:start_idx + batch_size * time_steps]

            batch_images = [load_and_preprocess_image(path, target_size) for path in batch_image_paths]
            batch_images = np.array(batch_images).reshape(batch_size, time_steps, *target_size, 3)
            batch_labels = np.eye(num_classes)[batch_labels[:batch_size]]

            # Apply data augmentation
            for i in range(batch_size * time_steps):
                batch_images.reshape(-1, *target_size, 3)[i] = datagen.random_transform(batch_images.reshape(-1, *target_size, 3)[i])

            yield batch_images, batch_labels

# Directory paths
train_directory = '/content/combined_frames_flows/train'
test_directory = '/content/combined_frames_flows/test'

# Get all image paths and labels
train_image_paths, train_labels = get_image_paths_and_labels(train_directory)
test_image_paths, test_labels = get_image_paths_and_labels(test_directory)

# Calculate class weights for imbalanced dataset
class_weights = compute_class_weight('balanced', classes=np.unique(train_labels), y=train_labels)
# Manually adjust class weights if necessary
class_weights = {i: weight * 1.5 if i in [0, 1, 2] else weight for i, weight in enumerate(class_weights)}

# Define batch size, time steps, and target size
batch_size = 4
time_steps = 5
target_size = (224, 224)

# Convert lists to numpy arrays for indexing in generator
train_labels = np.array(train_labels)
test_labels = np.array(test_labels)

# Create training and validation datasets
train_steps_per_epoch = len(train_image_paths) // (batch_size * time_steps)
val_steps_per_epoch = len(test_image_paths) // (batch_size * time_steps)

train_dataset = tf.data.Dataset.from_generator(
    lambda: data_generator(train_image_paths, train_labels, batch_size, time_steps, target_size),
    output_signature=(
        tf.TensorSpec(shape=(batch_size, time_steps, *target_size, 3), dtype=tf.float32),
        tf.TensorSpec(shape=(batch_size, 8), dtype=tf.float32)
    )
)

val_dataset = tf.data.Dataset.from_generator(
    lambda: data_generator(test_image_paths, test_labels, batch_size, time_steps, target_size),
    output_signature=(
        tf.TensorSpec(shape=(batch_size, time_steps, *target_size, 3), dtype=tf.float32),
        tf.TensorSpec(shape=(batch_size, 8), dtype=tf.float32)
    )
)

# Prefetch data to improve performance
train_dataset = train_dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
val_dataset = val_dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

# Adjusted CNN-GRU Model for 3-channel combined frames with time steps
model = Sequential()

# CNN Feature Extractor (with 3 channels for combined frame and optical flow data)
model.add(TimeDistributed(Conv2D(16, (3, 3), padding='same', kernel_regularizer=l2(0.01)), input_shape=(time_steps, *target_size, 3)))
model.add(TimeDistributed(BatchNormalization()))
model.add(TimeDistributed(LeakyReLU(negative_slope=0.1)))
model.add(TimeDistributed(MaxPooling2D((2, 2))))

model.add(TimeDistributed(Conv2D(32, (3, 3), padding='same', kernel_regularizer=l2(0.01))))
model.add(TimeDistributed(BatchNormalization()))
model.add(TimeDistributed(LeakyReLU(negative_slope=0.1)))
model.add(TimeDistributed(MaxPooling2D((2, 2))))

model.add(TimeDistributed(Conv2D(64, (3, 3), padding='same', kernel_regularizer=l2(0.01))))
model.add(TimeDistributed(BatchNormalization()))
model.add(TimeDistributed(LeakyReLU(negative_slope=0.1)))
model.add(TimeDistributed(MaxPooling2D((2, 2))))

# Flatten the spatial dimensions to feed into the GRU
model.add(TimeDistributed(Flatten()))

# GRU Layer
model.add(GRU(128, return_sequences=False))

# Fully Connected Layers
model.add(Dense(128, activation='relu', kernel_regularizer=l2(0.01)))
model.add(Dropout(0.5))

# Output Layer (8 output classes for emotion)
model.add(Dense(8, activation='softmax'))

# Compile the model with a smaller learning rate
optimizer = Adam(learning_rate=0.0001)
model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

model.summary()

# Define early stopping to prevent overfitting and save time
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

# Learning rate scheduler to reduce learning rate when validation loss plateaus
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6, verbose=1)

# Train the model with known steps per epoch
history = model.fit(
    train_dataset,
    validation_data=val_dataset,
    epochs=100,
    steps_per_epoch=train_steps_per_epoch,
    validation_steps=val_steps_per_epoch,
    callbacks=[early_stopping, reduce_lr],  # Added learning rate scheduler
    class_weight=class_weights  # Apply adjusted class weights
)

# Save the trained model in the recommended Keras format
model.save('/content/saved_model.keras')

#code for testing
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import tensorflow as tf

# Load the trained model
model = tf.keras.models.load_model('emotion_recognition_model.h5')

# Recompile the model to ensure metrics are built
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Evaluate the model to build the metrics (required to avoid the warning)
loss, accuracy = model.evaluate(X_test, y_test)
print(f"Test Accuracy: {accuracy * 100:.2f}%")

# Set up labels
emotion_categories = ['neutral', 'calm', 'happy', 'sad', 'angry', 'fearful', 'disgust', 'surprised']

# Make predictions on the test set
predictions = model.predict(X_test)
predicted_classes = np.argmax(predictions, axis=1)
true_classes = np.argmax(y_test, axis=1)

# Calculate accuracy
accuracy = np.sum(predicted_classes == true_classes) / len(true_classes)
print(f'Test Accuracy: {accuracy * 100:.2f}%')

# Generate classification report
print("\nClassification Report:")
print(classification_report(true_classes, predicted_classes, target_names=emotion_categories))

# Generate confusion matrix
conf_matrix = confusion_matrix(true_classes, predicted_classes)

# Plot confusion matrix
plt.figure(figsize=(10, 8))
sns.heatmap(conf_matrix, annot=True, cmap='Blues', fmt='d', xticklabels=emotion_categories, yticklabels=emotion_categories)
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix')
plt.show()

model_path = '/content/drive/MyDrive/emotion_recognition_model.h5'
model.save(model_path)
print(f'Model saved to: {model_path}')
