import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix
import seaborn as sns
import os
from tensorflow.keras.applications import VGG16
from tensorflow.keras.applications.vgg16 import preprocess_input

# Set random seed for reproducibility
tf.random.set_seed(42)

# Parameters
IMG_HEIGHT = 150
IMG_WIDTH = 150
BATCH_SIZE = 32
EPOCHS = 25
NUM_CLASSES = 20

# Create data generators with data augmentation for training
# Modify the data preprocessing
train_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input,
    validation_split=0.2,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

# Load and prepare the data
train_generator = train_datagen.flow_from_directory(
    'dataset',
    target_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='training'
)

validation_generator = train_datagen.flow_from_directory(
    'dataset',
    target_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='validation'
)


# Load the pre-trained VGG16 model
base_model = VGG16(weights='imagenet',
                   include_top=False,
                   input_shape=(IMG_HEIGHT, IMG_WIDTH, 3))

# Freeze the pre-trained layers
base_model.trainable = False

# Create the new model with VGG16 as base
model = models.Sequential([
    base_model,
    layers.Flatten(),
    layers.Dense(256, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(NUM_CLASSES, activation='softmax')
])



# Compile the model
model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Train the model
history = model.fit(
    train_generator,
    epochs=EPOCHS,
    validation_data=validation_generator
)

# Plot training results
plt.figure(figsize=(12, 4))

# Plot training & validation accuracy
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

# Plot training & validation loss
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.tight_layout()
plt.show()

# Generate predictions for confusion matrix
predictions = []
true_labels = []

for i in range(len(validation_generator)):
    x, y = validation_generator[i]
    pred = model.predict(x)
    pred_classes = np.argmax(pred, axis=1)
    true_classes = np.argmax(y, axis=1)

    predictions.extend(pred_classes)
    true_labels.extend(true_classes)

    if len(predictions) >= len(validation_generator.labels):
        break
# Create and plot the overall confusion matrix
cm = confusion_matrix(true_labels[:len(validation_generator.labels)],
                     predictions[:len(validation_generator.labels)])

plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.show()

# Print the overall accuracy
accuracy = np.sum(np.diag(cm)) / np.sum(cm)
print(f"\nOverall Accuracy: {accuracy:.4f}")

# Calculate and print True Positives, False Positives, True Negatives, and False Negatives
TP = np.diag(cm)  # True Positives are on the diagonal
FP = np.sum(cm, axis=0) - TP  # False Positives are sum of columns minus TP
FN = np.sum(cm, axis=1) - TP  # False Negatives are sum of rows minus TP
TN = np.sum(cm) - (FP + FN + TP)  # True Negatives are total sum minus others

print("\nOverall Metrics:")
print(f"True Positives: {np.sum(TP)}")
print(f"False Positives: {np.sum(FP)}")
print(f"True Negatives: {np.sum(TN)}")
print(f"False Negatives: {np.sum(FN)}")



# Add these imports at the beginning
from sklearn.metrics import precision_score, recall_score, f1_score, classification_report

# Add this code after the confusion matrix plotting and before saving the model
# Calculate metrics for each class
precision = precision_score(true_labels[:len(validation_generator.labels)],
                          predictions[:len(validation_generator.labels)],
                          average=None)
recall = recall_score(true_labels[:len(validation_generator.labels)],
                     predictions[:len(validation_generator.labels)],
                     average=None)
f1 = f1_score(true_labels[:len(validation_generator.labels)],
              predictions[:len(validation_generator.labels)],
              average=None)

# Calculate average metrics
avg_precision = precision_score(true_labels[:len(validation_generator.labels)],
                              predictions[:len(validation_generator.labels)],
                              average='weighted')
avg_recall = recall_score(true_labels[:len(validation_generator.labels)],
                         predictions[:len(validation_generator.labels)],
                         average='weighted')
avg_f1 = f1_score(true_labels[:len(validation_generator.labels)],
                  predictions[:len(validation_generator.labels)],
                  average='weighted')

# Print metrics for each class
print("\nMetrics for each class:")
for i in range(NUM_CLASSES):
    print(f"\nClass {i}:")
    print(f"Precision: {precision[i]:.4f}")
    print(f"Recall: {recall[i]:.4f}")
    print(f"F1-score: {f1[i]:.4f}")


# Print average metrics
print("\nWeighted Average Metrics:")
print(f"Precision: {avg_precision:.4f}")
print(f"Recall: {avg_recall:.4f}")
print(f"F1-score: {avg_f1:.4f}")

# Print detailed classification report
print("\nDetailed Classification Report:")
print(classification_report(true_labels[:len(validation_generator.labels)],
                          predictions[:len(validation_generator.labels)]))