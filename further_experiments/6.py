import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix
import seaborn as sns
import os
from tensorflow.keras.applications import MobileNet
from tensorflow.keras.applications.mobilenet import preprocess_input
from sklearn.metrics import precision_score, recall_score, f1_score, classification_report

tf.random.set_seed(42)

# Parameters
IMG_HEIGHT = 150
IMG_WIDTH = 150
BATCH_SIZE = 32
EPOCHS = 25
NUM_CLASSES = 20


train_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input,
    validation_split=0.2,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)


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



base_model = MobileNet(weights='imagenet',
                   include_top=False,
                   input_shape=(IMG_HEIGHT, IMG_WIDTH, 3))

# Freeze the pre-trained layers
base_model.trainable = False


model = models.Sequential([
    base_model,
    layers.Flatten(),
    layers.Dense(256, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(NUM_CLASSES, activation='softmax')
])




# Compileing
model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Training
history = model.fit(
    train_generator,
    epochs=EPOCHS,
    validation_data=validation_generator
)


plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()


plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.tight_layout()
plt.show()

# predictions for confusion matrix
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
# overall confusion matrix
cm = confusion_matrix(true_labels[:len(validation_generator.labels)],
                     predictions[:len(validation_generator.labels)])

plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.show()

# overall accuracy
accuracy = np.sum(np.diag(cm)) / np.sum(cm)
print(f"\nOverall Accuracy: {accuracy:.4f}")


TP = np.diag(cm)
FP = np.sum(cm, axis=0) - TP
FN = np.sum(cm, axis=1) - TP
TN = np.sum(cm) - (FP + FN + TP)

print("\nOverall Metrics:")
print(f"True Positives: {np.sum(TP)}")
print(f"False Positives: {np.sum(FP)}")
print(f"True Negatives: {np.sum(TN)}")
print(f"False Negatives: {np.sum(FN)}")



precision = precision_score(true_labels[:len(validation_generator.labels)],
                          predictions[:len(validation_generator.labels)],
                          average=None)
recall = recall_score(true_labels[:len(validation_generator.labels)],
                     predictions[:len(validation_generator.labels)],
                     average=None)
f1 = f1_score(true_labels[:len(validation_generator.labels)],
              predictions[:len(validation_generator.labels)],
              average=None)


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


# average metrics
print("\nWeighted Average Metrics:")
print(f"Precision: {avg_precision:.4f}")
print(f"Recall: {avg_recall:.4f}")
print(f"F1-score: {avg_f1:.4f}")

print("\nDetailed Classification Report:")
print(classification_report(true_labels[:len(validation_generator.labels)],
                          predictions[:len(validation_generator.labels)]))