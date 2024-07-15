import os
import librosa
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten
from tensorflow.keras.utils import to_categorical
import soundfile as sf

# Function to load audio files and features
def extract_mfcc_features(audio_path):
    signal, sample_rate = librosa.load(audio_path, sr=None)
    mfcc_features = librosa.feature.mfcc(y=signal, sr=sample_rate, n_mfcc=40)
    return mfcc_features

# Function to extract the emotion label from the filename
def extract_emotion_label(filename):
    emotion_mapping = {
        '01': 'neutral',
        '02': 'calm',
        '03': 'happy',
        '04': 'sad',
        '05': 'angry',
        '06': 'fearful',
        '07': 'disgust',
        '08': 'surprised'
    }
    return emotion_mapping.get(filename.split('-')[2], None)

# Load and preprocess the RAVDESS dataset
def load_audio_dataset(directory):
    features = []
    labels = []
    for root, _, files in os.walk(directory):
        print(f"Checking directory: {root}")  # Debugging statement
        for file in files:
            if file.endswith('.wav'):
                audio_path = os.path.join(root, file)
                print(f"Processing file: {audio_path}")  # Debugging statement
                mfcc_features = extract_mfcc_features(audio_path)
                emotion = extract_emotion_label(file)
                if emotion is not None:
                    features.append(mfcc_features)
                    labels.append(emotion)
                else:
                    print(f"Skipping file {audio_path}, could not extract emotion.")
    
    features = np.array(features)
    labels = np.array(labels)
    
    return features, labels

# Sample audio data 
sample_data_dir = 'Audio_Speech_Actors_01-24'

# Creating a sample directory structure and audio files
os.makedirs(sample_data_dir, exist_ok=True)
sample_structure = {
    'Actor_01': ['03-01-01-01-01-01-01.wav', '03-01-01-01-01-02-01.wav'],
    'Actor_02': ['03-01-01-01-01-01-02.wav', '03-01-01-01-01-02-02.wav'],
}

# Generate dummy audio files
for actor, files in sample_structure.items():
    actor_dir = os.path.join(sample_data_dir, actor)
    os.makedirs(actor_dir, exist_ok=True)
    for file in files:
        file_path = os.path.join(actor_dir, file)
        if not os.path.exists(file_path):
            sf.write(file_path, np.random.random(22050), 22050)

# Load the dataset
features, labels = load_audio_dataset(sample_data_dir)

# Check if data was loaded correctly
print(f"Loaded {len(features)} samples.")
print(f"Labels: {set(labels)}")

# If no data is loaded, raise an error
if len(features) == 0 or len(labels) == 0:
    raise ValueError("No data found. Please check the dataset path and file format.")

# Reshape features for CNN input
features = np.expand_dims(features, -1)

# Convert string labels to integers
unique_emotions = list(set(labels))
emotion_to_index = {emotion: index for index, emotion in enumerate(unique_emotions)}
labels = np.array([emotion_to_index[emotion] for emotion in labels])

# One-hot encode the labels
labels = to_categorical(labels, num_classes=len(unique_emotions))

# Split the dataset into training and testing sets
features_train, features_test, labels_train, labels_test = train_test_split(features, labels, test_size=0.2, random_state=42)

# Build a simple CNN model
emotion_model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(features_train.shape[1], features_train.shape[2], 1)),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(64, activation='relu'),
    Dense(len(unique_emotions), activation='softmax')
])

emotion_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
emotion_model.fit(features_train, labels_train, epochs=20, validation_data=(features_test, labels_test))

# Model Evaluation
model_loss, model_accuracy = emotion_model.evaluate(features_test, labels_test)
print(f'Test Accuracy: {model_accuracy}')
