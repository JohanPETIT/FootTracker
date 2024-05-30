import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import keras
#from tensorflow.keras.utils import to_categorical
#from tensorflow.keras import models, layers
import matplotlib.pyplot as plt
import torch
import torch.nn as nn # Contain classes of training
import torch.optim as optim # Contain algorithms to train model efficassily.
import torch.nn.functional as Function
from torch.utils.data import DataLoader
from vd3 import VideoDataset,transform


#Initialization of the dataset and DataLoader
video_dir = '/storage8To/student_projects/foottracker/detectionData/train'
csv_file = '/storage8To/student_projects/foottracker/detectionData/train.csv'

# Initialize dataset
dataset = VideoDataset(video_dir, csv_file=csv_file, transform=transform, frame_count=30, specific_video='1606b0e6_0.mp4')

#Encoding labels and split the dataset
data_loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=4)
all_frames, all_labels = [], []

for frames, _, _, labels in data_loader:
    if len(frames) > 0:
        all_frames.append(frames)
        all_labels.append(labels)

all_frames = torch.cat([torch.stack(frames) for frames in all_frames])
all_labels = [f"[{str(t)}]" for t in all_labels]

# Split the dataset into training and test sets
X_train, X_test, Y_train, Y_test = train_test_split(all_frames.numpy(), all_labels, test_size=0.3, random_state=42)

# Encode labels
label_encoder = LabelEncoder()
Y_train_encoded = label_encoder.fit_transform(Y_train)
Y_test_encoded = label_encoder.transform(Y_test)

# Convert to one-hot encoding
Y_train_encoded = to_categorical(Y_train_encoded)
Y_test_encoded = to_categorical(Y_test_encoded)

# Define the number of classes
num_classes = 5 # 3?

# Define a dictionary to map labels to numerical values
label_mapping = {
    'play': 1,
    'touchin': 4,
    'challenge': 3,
    'start': 2, #delete
    'end': 5 # delete
}

# Map the original labels to numerical values using the dictionary
Y_train_encoded = [label_mapping[label] for label in Y_train]
y_test_encoded = [label_mapping[label] for label in Y_test]

# Convert encoded labels to one-hot encoded format
Y_train_encoded = to_categorical(Y_train_encoded, num_classes=num_classes)
y_test_encoded = to_categorical(y_test_encoded, num_classes=num_classes)

# Define the neural network model
model = models.Sequential()
model.add(layers.Dense(128, activation='relu', input_shape=(3, 224, 224)))  
model.add(layers.Dropout(0.5))
model.add(layers.Dense(128, activation='relu'))
model.add(layers.Dense(num_classes, activation='softmax'))

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
history = model.fit(X_train, Y_train_encoded, epochs=60, batch_size=128, validation_split=0.2)

# Get training info
loss = history.history['loss']
val_loss = history.history['val_loss']
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

# Plot the training and validation loss
plt.figure()
plt.plot(loss, 'b', label='Training loss')
plt.plot(val_loss, 'm', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()
plt.show()

# Plot the training and validation accuracy
plt.figure()
plt.plot(acc, 'b', label='Training accuracy')
plt.plot(val_acc, 'm', label='Validation accuracy')
plt.title('Training and validation accuracy')
plt.legend()
plt.show()

# Test the network
preds = model.predict(X_test)
labels3 = preds.argmax(axis=1)

# Reverse mapping dictionary to get string labels from numerical values
reverse_label_mapping = {v: k for k, v in label_mapping.items()}

# Visualize the predictions with string labels
n_images = len(X_test)
n_rows = 10
n_cols = n_images // n_rows + 1

f, ax = plt.subplots(n_rows, n_cols, figsize=(12, 12))
for i in range(n_images):
    row_idx = i // n_cols
    col_idx = i % n_cols
    ax[row_idx, col_idx].imshow(X_test[i])  # Use imshow for 2D image data
    ax[row_idx, col_idx].set_title(reverse_label_mapping[labels3[i]])
    ax[row_idx, col_idx].axis("off")

# Hide empty subplots
for i in range(n_images, n_rows * n_cols):
    row_idx = i // n_cols
    col_idx = i % n_cols
    ax[row_idx, col_idx].axis("off")

plt.tight_layout()
plt.show()

# Evaluate the model on the test set
test_loss3, test_acc3 = model.evaluate(X_test, y_test_encoded)

print("Test Loss:", test_loss3)
print("Test Accuracy:", test_acc3)