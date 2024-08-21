import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, roc_curve, auc, \
    classification_report
import seaborn as sns
from sklearn.utils import shuffle
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint, ReduceLROnPlateau
import tensorflow as tf

# Free up memory at the start
tf.keras.backend.clear_session()

# Define labels and directories
labels = ['glioma_tumor', 'no_tumor', 'meningioma_tumor', 'pituitary_tumor']
train_dir = '/kaggle/input/brain-tumor-classification-mri/Training'
test_dir = '/kaggle/input/brain-tumor-classification-mri/Testing'
image_size = 224  # Adjusted to 224 for EfficientNetB0


# Load images from directories
def load_images(data_dir, labels, image_size):
    X, y = [], []
    for label in labels:
        folder_path = os.path.join(data_dir, label)
        for image_file in tqdm(os.listdir(folder_path)):
            img = cv2.imread(os.path.join(folder_path, image_file))
            img = cv2.resize(img, (image_size, image_size))
            X.append(img)
            y.append(label)
    return np.array(X), np.array(y)


# Load and plot data distributions
X1, y1 = load_images(train_dir, labels, image_size)
X2, y2 = load_images(test_dir, labels, image_size)

# Combine and shuffle datasets
X, y = np.concatenate((X1, X2), axis=0), np.concatenate((y1, y2), axis=0)
X, y = shuffle(X, y, random_state=42)

# Split data into training, validation, and test sets
X_train_val, X_test, y_train_val, y_test = train_test_split(X, y, test_size=0.1, stratify=y, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=1 / 9, stratify=y_train_val,
                                                  random_state=42)

# Normalize images
X_train_normalized = X_train.astype('float32') / 255
X_val_normalized = X_val.astype('float32') / 255
X_test_normalized = X_test.astype('float32') / 255

# Data augmentation
datagen = ImageDataGenerator(
    rotation_range=20, width_shift_range=0.1, height_shift_range=0.1,
    shear_range=0.2, zoom_range=0.2, horizontal_flip=True, vertical_flip=True, fill_mode='nearest'
)

# Encode labels
le = LabelEncoder()
y_train_encoded = le.fit_transform(y_train)
y_val_encoded = le.transform(y_val)
y_test_encoded = le.transform(y_test)
y_train_cat = tf.keras.utils.to_categorical(y_train_encoded, len(labels))
y_val_cat = tf.keras.utils.to_categorical(y_val_encoded, len(labels))
y_test_cat = tf.keras.utils.to_categorical(y_test_encoded, len(labels))

# Load the ResNet101 model pretrained on ImageNet without the top layers
resnet = tf.keras.applications.ResNet101(weights='imagenet', include_top=False,
                                         input_shape=(image_size, image_size, 3))

# Build the custom model on top of the ResNet101 base

model = resnet.output
model = tf.keras.layers.GlobalAveragePooling2D()(model)
model = tf.keras.layers.Dropout(rate=0.5)(model)
model = tf.keras.layers.Dense(4, activation='softmax')(model)
model = tf.keras.models.Model(inputs=resnet.input, outputs=model)

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()

# Set up callbacks
tensorboard = TensorBoard(log_dir='logs')
checkpoint = ModelCheckpoint("effnet.keras", monitor="val_accuracy", save_best_only=True, mode="auto", verbose=1)
reduce_lr = ReduceLROnPlateau(monitor='val_accuracy', factor=0.3, patience=2, min_delta=0.001, mode='auto', verbose=1)

# Train the model
history = model.fit(
    datagen.flow(X_train_normalized, y_train_cat, batch_size=32),
    epochs=12,
    validation_data=(X_val_normalized, y_val_cat),
    callbacks=[tensorboard, checkpoint, reduce_lr],
    verbose=1
)

# Evaluate the model on the test set
test_loss, test_accuracy = model.evaluate(X_test_normalized, y_test_cat, verbose=0)
print(f"Test accuracy: {test_accuracy:.4f}")


# Plot accuracy and loss
def plot_metrics(history):
    plt.figure(figsize=(12, 5))

    # Plot Accuracy
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.title('Accuracy over Epochs')
    plt.legend()

    # Plot Loss
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Loss over Epochs')
    plt.legend()

    plt.tight_layout()
    plt.show()


# Call the function to plot the metrics
plot_metrics(history)

# Generate predictions on the test set
y_test_pred = model.predict(X_test_normalized)
y_test_pred_classes = np.argmax(y_test_pred, axis=1)

# Calculate confusion matrix
conf_matrix = confusion_matrix(y_test_encoded, y_test_pred_classes)

# Plot confusion matrix
plt.figure(figsize=(10, 8))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
plt.title('Confusion Matrix')
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.show()

# Calculate accuracy, precision, recall, F1 score
accuracy = accuracy_score(y_test_encoded, y_test_pred_classes)
precision = precision_score(y_test_encoded, y_test_pred_classes, average='weighted')
recall = recall_score(y_test_encoded, y_test_pred_classes, average='weighted')
f1 = f1_score(y_test_encoded, y_test_pred_classes, average='weighted')

# Print classification report
print(classification_report(y_test_encoded, y_test_pred_classes, target_names=labels))


# Calculate specificity for each class
def calculate_specificity(conf_matrix):
    specificity_per_class = []
    for i in range(conf_matrix.shape[0]):
        true_negatives = np.sum(np.delete(np.delete(conf_matrix, i, axis=0), i, axis=1))
        false_positives = np.sum(conf_matrix[:, i]) - conf_matrix[i, i]
        specificity = true_negatives / (true_negatives + false_positives)
        specificity_per_class.append(specificity)
    return specificity_per_class


specificity_per_class = calculate_specificity(conf_matrix)
for i, label in enumerate(labels):
    print(f"Specificity for {label}: {specificity_per_class[i]:.4f}")

# Calculate sensitivity for each class
sensitivity_per_class = recall_score(y_test_encoded, y_test_pred_classes, average=None)
for i, label in enumerate(labels):
    print(f"Sensitivity (Recall) for {label}: {sensitivity_per_class[i]:.4f}")

# Calculate and print AUC-ROC for each class
fpr, tpr, roc_auc = {}, {}, {}
for i in range(len(labels)):
    fpr[i], tpr[i], _ = roc_curve(y_test_cat[:, i], y_test_pred[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

# Plot ROC curves
plt.figure(figsize=(12, 8))
colors = ['blue', 'red', 'green', 'orange']
for i, color in enumerate(colors):
    plt.plot(fpr[i], tpr[i], color=color, lw=2, label=f'ROC curve of class {labels[i]} (area = {roc_auc[i]:.4f})')

plt.plot([0, 1], [0, 1], 'k--', lw=2)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curves')
plt.legend(loc="lower right")
plt.show()

# Print metrics
print(f"Accuracy: {accuracy:.4f}")
print(f"Precision (weighted): {precision:.4f}")
print(f"Sensitivity (Recall, weighted): {recall:.4f}")
print(f"F1 Score (weighted): {f1:.4f}")

# Print AUC for each class
for i, label in enumerate(labels):
    print(f"AUC-ROC for {label}: {roc_auc[i]:.4f}")
