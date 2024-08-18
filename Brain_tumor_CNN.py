import os
import cv2
import numpy as np
from tqdm import tqdm
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, roc_curve, auc, \
    classification_report
from sklearn.utils import shuffle
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
import tensorflow as tf

# Define labels and directories
labels = ['glioma_tumor', 'no_tumor', 'meningioma_tumor', 'pituitary_tumor']
train_dir = 'C:/Users/nthoa/PycharmProjects/Brain Tumor Project/Training'
test_dir = 'C:/Users/nthoa/PycharmProjects/Brain Tumor Project/Testing'
image_size = 150
batch_size = 32

# Load images in batches
def load_images_in_batches(data_dir, labels, image_size, batch_size):
    X_batch, y_batch = [], []
    for label in labels:
        folder_path = os.path.join(data_dir, label)
        for image_file in tqdm(os.listdir(folder_path)):
            img = cv2.imread(os.path.join(folder_path, image_file))
            img = cv2.resize(img, (image_size, image_size))
            X_batch.append(img)
            y_batch.append(label)
            if len(X_batch) == batch_size:
                X_batch, y_batch = np.array(X_batch), np.array(y_batch)
                yield X_batch, y_batch
                X_batch, y_batch = [], []
    if len(X_batch) > 0:
        yield np.array(X_batch), np.array(y_batch)

# Load and preprocess data in batches
def preprocess_and_augment_data(data_dir, labels, image_size, batch_size):
    le = LabelEncoder()
    datagen = ImageDataGenerator(
        rotation_range=20, width_shift_range=0.1, height_shift_range=0.1,
        shear_range=0.2, zoom_range=0.2, horizontal_flip=True, vertical_flip=True, fill_mode='nearest'
    )

    X_data, y_data = [], []
    for X_batch, y_batch in load_images_in_batches(data_dir, labels, image_size, batch_size):
        y_batch_encoded = le.fit_transform(y_batch)
        y_batch_cat = to_categorical(y_batch_encoded, len(labels))
        X_batch_normalized = X_batch.astype('float32') / 255

        X_augmented, y_augmented = [], []
        for image, label in zip(X_batch_normalized, y_batch_cat):
            X_augmented.append(image)
            y_augmented.append(label)
            image = image.reshape((1,) + image.shape)
            i = 0
            for batch in datagen.flow(image, batch_size=1):
                X_augmented.append(batch[0])
                y_augmented.append(label)
                i += 1
                if i >= 5:  # Number of augmented images per original image
                    break

        X_data.append(np.array(X_augmented))
        y_data.append(np.array(y_augmented))


    X_data = np.concatenate(X_data, axis=0)
    y_data = np.concatenate(y_data, axis=0)

    return X_data, y_data

# Load and preprocess the training data
X_train_augmented, y_train_augmented = preprocess_and_augment_data(train_dir, labels, image_size, batch_size)

# Load and preprocess the validation and test data without augmentation
X_val_augmented, y_val_augmented = preprocess_and_augment_data(test_dir, labels, image_size, batch_size)

# Shuffle and split data
X_train_augmented, y_train_augmented = shuffle(X_train_augmented, y_train_augmented, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_val_augmented, y_val_augmented, test_size=0.5, random_state=42)

# Build and compile the CNN model
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(image_size, image_size, 3)),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(64, activation='relu'),
    Dropout(0.5),
    Dense(len(labels), activation='softmax')
])

model.compile(optimizer=Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()

# Train the model with batch processing
history = model.fit(X_train_augmented, y_train_augmented, epochs=25, batch_size=batch_size,
                    validation_data=(X_val, y_val), verbose=1)

# Evaluate the model on the test set
test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)
print(f"Test accuracy: {test_accuracy:.4f}")

# Generate predictions on the test set
y_test_pred = model.predict(X_test)
y_test_pred_classes = np.argmax(y_test_pred, axis=1)

# Calculate confusion matrix
conf_matrix = confusion_matrix(np.argmax(y_test, axis=1), y_test_pred_classes)

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
recall = recall_score(y_test_encoded, y_test_pred_classes, average='weighted')  # This is also the weighted sensitivity
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
