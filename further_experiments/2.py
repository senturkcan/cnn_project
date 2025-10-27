from sklearn.metrics import confusion_matrix, classification_report
import numpy as np
import seaborn as sns
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
import os
import matplotlib.pyplot as plt

# Define the dataset path (same folder as the script)
dataset_path = os.path.join(os.getcwd(), 'dataset')

# Model parameters
input_shape = (128, 128, 3)  # Resize images to 128x128 (3 channels for RGB)
num_classes = 20             # Number of classes
batch_size = 32
epochs = 15
learning_rate = 0.001

# Data preprocessing and augmentation
datagen = tf.keras.preprocessing.image.ImageDataGenerator(
    rescale=1.0/255.0,           # Normalize pixel values to [0, 1]
    rotation_range=20,           # Data augmentation: Rotate images up to 20 degrees
    width_shift_range=0.2,       # Horizontal translation
    height_shift_range=0.2,      # Vertical translation
    zoom_range=0.2,              # Random zoom
    horizontal_flip=True         # Flip images horizontally
)

# Load the entire dataset using image_dataset_from_directory
train_data = tf.keras.preprocessing.image_dataset_from_directory(
    dataset_path,
    image_size=(128, 128),  # Ensure all images are resized to (128, 128)
    batch_size=batch_size,
    label_mode='categorical',  # One-hot encoding for labels
    validation_split=0.2,      # Split data: 80% train, 20% validation
    subset='training',         # Training set
    seed=123
)

val_data = tf.keras.preprocessing.image_dataset_from_directory(
    dataset_path,
    image_size=(128, 128),  # Ensure all images are resized to (128, 128)
    batch_size=batch_size,
    label_mode='categorical',  # One-hot encoding for labels
    validation_split=0.2,      # Split data: 80% train, 20% validation
    subset='validation',       # Validation set
    seed=123
)

# Use AUTOTUNE for performance optimization
AUTOTUNE = tf.data.AUTOTUNE
train_data = train_data.cache().prefetch(buffer_size=AUTOTUNE)
val_data = val_data.cache().prefetch(buffer_size=AUTOTUNE)

# Define the CNN model
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
    MaxPooling2D((2, 2)),

    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),

    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),

    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(num_classes, activation='softmax')
])

# Compile the model
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Train the model
history = model.fit(train_data, validation_data=val_data, epochs=epochs)

# Save the model
model.save('simple_animal_cnn.h5')

# Evaluate the model on validation data to generate predictions
val_data.reset()
y_true = np.concatenate([y for x, y in val_data], axis=0)  # Get true labels
y_pred = np.argmax(model.predict(val_data), axis=-1)         # Get predicted labels

# Confusion Matrix
conf_mat = confusion_matrix(np.argmax(y_true, axis=-1), y_pred)

# Plot the confusion matrix
plt.figure(figsize=(12, 10))
sns.heatmap(conf_mat, annot=True, fmt='d', cmap='Blues', xticklabels=val_data.class_names, yticklabels=val_data.class_names)
plt.title('Confusion Matrix')
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.show()

# Print classification report
print("\nClassification Report:\n")
print(classification_report(np.argmax(y_true, axis=-1), y_pred, target_names=val_data.class_names))
