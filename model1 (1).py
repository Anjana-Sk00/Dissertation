
!pip install --upgrade pip
!pip install numpy pandas opencv-python matplotlib scikit-learn

!pip install tensorflow pillow seaborn

# Step 2: Import necessary libraries for data handling, image processing, and visualization
import os
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img
from tensorflow.keras.utils import to_categorical
from tensorflow.keras import layers, models, Input
from tensorflow.keras.applications import VGG16
from tensorflow.keras.optimizers import Adam
from google.colab import drive
import zipfile

drive.mount('/content/drive')

import os

drive_zip_path = '/content/drive/MyDrive/'  
colab_unzip_path = '/content/unzipped_data/'  
processed_faces_path = '/content/processed_faces/'  

if not os.path.exists(processed_faces_path):
    os.makedirs(processed_faces_path)

import wget

base_url = 'https://zenodo.org/record/1188976/files/Video_Speech_Actor_'

for i in range(1, 25):  
    actor_num = str(i).zfill(2)  
    url = base_url + actor_num + '.zip?download=1'

    
    output_path = f'/content/drive/MyDrive/ravdess_video_actor_{actor_num}.zip'

    print(f"Downloading {url} to {output_path}")
    wget.download(url, output_path)

import zipfile

zip_files = [f"ravdess_video_actor_{str(i).zfill(2)}.zip" for i in range(1, 25)]

def process_zip_file(zip_file):
    zip_file_path = os.path.join(drive_zip_path, zip_file)

    with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
        print(f"Unzipping {zip_file}...")
        zip_ref.extractall(colab_unzip_path)

for zip_file in zip_files:  
    process_zip_file(zip_file)

import os

colab_unzip_path = '/content/unzipped_data/'  

video_counts_per_folder = {}

def count_videos_in_folders():
    for root, dirs, files in os.walk(colab_unzip_path):
        for folder in dirs:
            folder_path = os.path.join(root, folder)
            video_count = 0

            for file in os.listdir(folder_path):
                if file.endswith('.mp4'):  
                    video_count += 1

            video_counts_per_folder[folder] = video_count

count_videos_in_folders()

print("\nNumber of videos in each unzipped folder:")
for folder, video_count in video_counts_per_folder.items():
    print(f"{folder}: {video_count} videos")

total_videos = sum(video_counts_per_folder.values())
print(f"\nTotal number of videos across all folders: {total_videos}")

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

colab_unzip_path = '/content/unzipped_data/'  
processed_faces_path = '/content/processed_faces/'  

if not os.path.exists(processed_faces_path):
    os.makedirs(processed_faces_path)

videos_skipped = 0
videos_processed = 0

def extract_frames_with_labels(video_file, actor_id, video_id, emotion_label):
    cap = cv2.VideoCapture(video_file)
    frame_count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break  

        frame_filename = f"{emotion_label}_Actor_{actor_id}_video_{video_id}_frame_{frame_count}.jpg"
        frame_path = os.path.join(processed_faces_path, frame_filename)
        if cv2.imwrite(frame_path, frame):  
            print(f"Saved {frame_filename}")
        else:
            print(f"Failed to save {frame_filename}")
        frame_count += 1

    cap.release()
    print(f"Extracted {frame_count} frames from {video_file} (Expected: {frame_count})")

for root, dirs, files in os.walk(colab_unzip_path):
    for file in files:
        if file.endswith('.mp4'):
            video_file_path = os.path.join(root, file)

            try:
                
                actor_id = file.split('-')[6]  
                video_id = file.split('-')[1]  

                emotion_code = file.split('-')[2]
                emotion_label = emotion_dict.get(emotion_code, 'unknown')  

                if emotion_label == 'unknown':
                    raise ValueError(f"Invalid emotion code in {file}")

                print(f"Processing {file} with label {emotion_label}...")

                extract_frames_with_labels(video_file_path, actor_id, video_id, emotion_label)
                videos_processed += 1

            except (IndexError, ValueError) as e:
            
                videos_skipped += 1
                print(f"Failed to process {file}. Skipping... Error: {str(e)}")

print(f"\nTotal number of videos processed: {videos_processed}")
print(f"Total number of videos skipped: {videos_skipped}")

def count_extracted_frames():
    frame_counts_per_video = {}
    total_frames_extracted = 0

    for img_file in os.listdir(processed_faces_path):
       
        video_name = "_".join(img_file.split('_frame_')[0].split('_')[:4])  

        if video_name in frame_counts_per_video:
            frame_counts_per_video[video_name] += 1
        else:
            frame_counts_per_video[video_name] = 1

        total_frames_extracted += 1

    print("\nTotal frames extracted per video:")
    for video, frame_count in frame_counts_per_video.items():
        print(f"{video}: {frame_count} frames")

    print(f"\nOverall total frames extracted: {total_frames_extracted}")

count_extracted_frames()

import os

processed_faces_path = '/content/processed_faces/'

if os.path.exists(processed_faces_path):
   
    files = os.listdir(processed_faces_path)
    print(f"Total number of files (frames) in {processed_faces_path}: {len(files)}")
else:
    print(f"The directory {processed_faces_path} does not exist.")

pip install dlib

!wget http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2

!bzip2 -d shape_predictor_68_face_landmarks.dat.bz2

!ls

import dlib
predictor = dlib.shape_predictor("/content/shape_predictor_68_face_landmarks.dat")

#code for landmark extraction
import dlib
import cv2
import os

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("/content/shape_predictor_68_face_landmarks.dat")  

processed_faces_path = '/content/processed_faces/'

landmark_output_path = '/content/landmarks_faces/'
if not os.path.exists(landmark_output_path):
    os.makedirs(landmark_output_path)

def extract_facial_landmarks(image, image_filename):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)

    for face in faces:
        landmarks = predictor(gray, face)
        landmark_points = []
        for n in range(0, 68): 
            x = landmarks.part(n).x
            y = landmarks.part(n).y
            landmark_points.append((x, y))

            cv2.circle(image, (x, y), 2, (255, 0, 0), -1)

        landmark_img_path = os.path.join(landmark_output_path, image_filename)
        cv2.imwrite(landmark_img_path, image)

        return landmark_points
    return None

for img_file in os.listdir(processed_faces_path):
    if img_file.endswith('.jpg'):
        image_path = os.path.join(processed_faces_path, img_file)
        image = cv2.imread(image_path)

        landmarks = extract_facial_landmarks(image, img_file)
        if landmarks:
            print(f"Extracted landmarks from {img_file}")
        else:
            print(f"No face detected in {img_file}")

#code for data preparation
import os
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical
import cv2

processed_faces_path = '/content/processed_faces/'

data = []
labels = []

for img_file in os.listdir(processed_faces_path):
    if img_file.endswith('.jpg'):
        image_path = os.path.join(processed_faces_path, img_file)
        image = cv2.imread(image_path)

        if image is not None:
           
            image_resized = cv2.resize(image, (128, 128))
            data.append(image_resized)

            label = img_file.split('_')[0]
            labels.append(label)
        else:
            print(f"Could not read {img_file}")

data = np.array(data, dtype="float32") / 255.0 
labels = np.array(labels)

label_encoder = LabelEncoder()
labels_encoded = label_encoder.fit_transform(labels)
labels_encoded = to_categorical(labels_encoded)

X_train, X_test, y_train, y_test = train_test_split(data, labels_encoded, test_size=0.2, random_state=42)

print(f"Training data shape: {X_train.shape}")
print(f"Test data shape: {X_test.shape}")
print(f"Training labels shape: {y_train.shape}")
print(f"Test labels shape: {y_test.shape}")

#code for training model
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt

def create_cnn_model(input_shape, num_classes):
    model = models.Sequential()

    model.add(layers.Input(shape=input_shape))
    model.add(layers.Conv2D(32, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))

    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))

    model.add(layers.Conv2D(128, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))

    model.add(layers.Flatten())
    model.add(layers.Dense(128, activation='relu'))
    model.add(layers.Dropout(0.5))  
    model.add(layers.Dense(num_classes, activation='softmax'))

    return model

input_shape = (128, 128, 3)  
num_classes = y_train.shape[1]  

model = create_cnn_model(input_shape, num_classes)

learning_rate = 0.0001
optimizer = Adam(learning_rate=learning_rate)
model.compile(optimizer=optimizer,
              loss='categorical_crossentropy',
              metrics=['accuracy'])

data_generator = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

batch_size = 16
epochs = 100
history = model.fit(data_generator.flow(X_train, y_train, batch_size=batch_size),
                    validation_data=(X_test, y_test),
                    epochs=epochs,
                    batch_size=batch_size)

model.save('emotion_recognition_model.h5')

model.summary()

plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.title('Training and Validation Accuracy')
plt.show()

#code for testing
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import tensorflow as tf

model = tf.keras.models.load_model('emotion_recognition_model.h5')

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

loss, accuracy = model.evaluate(X_test, y_test)
print(f"Test Accuracy: {accuracy * 100:.2f}%")

emotion_categories = ['neutral', 'calm', 'happy', 'sad', 'angry', 'fearful', 'disgust', 'surprised']

predictions = model.predict(X_test)
predicted_classes = np.argmax(predictions, axis=1)
true_classes = np.argmax(y_test, axis=1)

accuracy = np.sum(predicted_classes == true_classes) / len(true_classes)
print(f'Test Accuracy: {accuracy * 100:.2f}%')

print("\nClassification Report:")
print(classification_report(true_classes, predicted_classes, target_names=emotion_categories))

conf_matrix = confusion_matrix(true_classes, predicted_classes)

plt.figure(figsize=(10, 8))
sns.heatmap(conf_matrix, annot=True, cmap='Blues', fmt='d', xticklabels=emotion_categories, yticklabels=emotion_categories)
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix')
plt.show()

model_path = '/content/drive/MyDrive/emotion_recognition_model.h5'
model.save(model_path)
print(f'Model saved to: {model_path}')
