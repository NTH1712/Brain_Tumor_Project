import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, classification_report, roc_curve, auc
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import shuffle
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
from tensorflow.keras.applications import InceptionV3
import tensorflow as tf
import seaborn as sns

# Clear previous session
tf.keras.backend.clear_session()

# Parameters
labels = ['glioma_tumor', 'no_tumor', 'meningioma_tumor', 'pituitary_tumor']
train_dir = 'C:/Users/nthoa/PycharmProjects/Brain Tumor Project/Training'
test_dir = 'C:/Users/nthoa/PycharmProjects/Brain Tumor Project/Testing'
image_size = 299

# Load images
def load_images(data_dir, labels, image_size):
    X, y = [], []
    for label in labels:
        folder_path = os.path.join(data_dir, label)
        for img_file in tqdm(os.listdir(folder_path)):
            img = cv2.imread(os.path.join(folder_path, img_file))
            img = cv2.resize(img, (image_size, image_size))
            X.append(img)
            y.append(label)
    return np.array(X), np.array(y)

# Load and shuffle data
X_train, y_train = load_images(train_dir, labels, image_size)
X_test, y_test = load_images(test_dir, labels, image_size)
X, y = shuffle(np.concatenate((X_train, X_test)), np.concatenate((y_train, y_test)), random_state=42)

# Split data into train, val, test
X_train_val, X_test, y_train_val, y_test = train_test_split(X, y, test_size=0.1, stratify=y, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=1/9, stratify=y_train_val, random_state=42)

# Normalize data
X_train = X_train.astype('float32') / 255
X_val = X_val.astype('float32') / 255
X_test = X_test.astype('float32') / 255

# Data augmentation
datagen = ImageDataGenerator(rotation_range=20, width_shift_range=0.1, height_shift_range=0.1, shear_range=0.2,
                             zoom_range=0.2, horizontal_flip=True, vertical_flip=True, fill_mode='nearest')

# Label encoding
le = LabelEncoder()
y_train_enc = le.fit_transform(y_train)
y_val_enc = le.transform(y_val)
y_test_enc = le.transform(y_test)

# One-hot encode labels
y_train_cat = tf.keras.utils.to_categorical(y_train_enc, len(labels))
y_val_cat = tf.keras.utils.to_categorical(y_val_enc, len(labels))
y_test_cat = tf.keras.utils.to_categorical(y_test_enc, len(labels))

# Build InceptionV3 model
base_model = InceptionV3(weights='imagenet', include_top=False, input_shape=(image_size, image_size, 3))
x = base_model.output
x = tf.keras.layers.GlobalAveragePooling2D()(x)
x = tf.keras.layers.Dropout(0.5)(x)
output = tf.keras.layers.Dense(len(labels), activation='softmax')(x)
model = tf.keras.models.Model(inputs=base_model.input, outputs=output)

# Compile model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()
# Callbacks
tensorboard = TensorBoard(log_dir='logs')
checkpoint = ModelCheckpoint("inceptionv3_model.keras", monitor="val_accuracy", save_best_only=True, mode="auto", verbose=1)
reduce_lr = ReduceLROnPlateau(monitor='val_accuracy', factor=0.3, patience=2, min_delta=0.001, mode='auto', verbose=1)
early_stopping = EarlyStopping(monitor='val_accuracy', patience=5, restore_best_weights=True)

# Train model
history = model.fit(datagen.flow(X_train, y_train_cat, batch_size=32), epochs=15, validation_data=(X_val, y_val_cat),
                    callbacks=[tensorboard, checkpoint, reduce_lr, early_stopping], verbose=1)

# Evaluate on test data
test_loss, test_accuracy = model.evaluate(X_test, y_test_cat, verbose=0)
print(f"Test accuracy: {test_accuracy:.4f}")

# Plot training metrics
def plot_metrics(history):
    plt.figure(figsize=(12, 5))

    # Plot accuracy
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.title('Accuracy over Epochs')
    plt.legend()

    # Plot loss
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Loss over Epochs')
    plt.legend()

    plt.tight_layout()
    plt.show()

plot_metrics(history)

# Generate predictions
y_pred = model.predict(X_test)
y_pred_classes = np.argmax(y_pred, axis=1)

# Confusion matrix
conf_matrix = confusion_matrix(y_test_enc, y_pred_classes)
plt.figure(figsize=(10, 8))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
plt.title('Confusion Matrix')
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.show()

# Classification metrics
print(classification_report(y_test_enc, y_pred_classes, target_names=labels))

# ROC and AUC
fpr, tpr, roc_auc = {}, {}, {}
for i in range(len(labels)):
    fpr[i], tpr[i], _ = roc_curve(y_test_cat[:, i], y_pred[:, i])
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

# Print weighted metrics
accuracy = accuracy_score(y_test_enc, y_pred_classes)
precision = precision_score(y_test_enc, y_pred_classes, average='weighted')
recall = recall_score(y_test_enc, y_pred_classes, average='weighted')
f1 = f1_score(y_test_enc, y_pred_classes, average='weighted')

print(f"Accuracy: {accuracy:.4f}")
print(f"Precision (weighted): {precision:.4f}")
print(f"Recall (weighted): {recall:.4f}")
print(f"F1 Score (weighted): {f1:.4f}")

# Print AUC for each class
for i, label in enumerate(labels):
    print(f"AUC-ROC for {label}: {roc_auc[i]:.4f}")

# Average AUC-ROC
average_auc = np.mean(list(roc_auc.values()))
print(f"Average AUC-ROC: {average_auc:.4f}")