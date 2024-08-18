import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import mplcyberpunk
from tqdm import tqdm
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, roc_curve, auc, classification_report
import seaborn as sns
from sklearn.utils import shuffle
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.applications import VGG16
from tensorflow.keras.applications.vgg16 import preprocess_input

# Define labels and directories
labels = ['glioma_tumor', 'no_tumor', 'meningioma_tumor', 'pituitary_tumor']
train_dir = 'C:/Users/nthoa/PycharmProjects/Brain Tumor Project/Training'
test_dir = 'C:/Users/nthoa/PycharmProjects/Brain Tumor Project/Testing'
image_size = 224


# Load images from directories in batches
def load_images_batch(data_dir, labels, image_size, batch_size=1000):
    while True:
        X, y = [], []
        for label in labels:
            folder_path = os.path.join(data_dir, label)
            for image_file in os.listdir(folder_path):
                if len(X) >= batch_size:
                    yield np.array(X), np.array(y)
                    X, y = [], []
                img = cv2.imread(os.path.join(folder_path, image_file))
                img = cv2.resize(img, (image_size, image_size))
                X.append(img)
                y.append(label)
        if X:
            yield np.array(X), np.array(y)


# Plot label distribution
def plot_distribution(label_counts, title):
    plt.figure(figsize=(12, 5))
    colors = ["C0", "C1", "C2", "C3"]
    bars = plt.bar(label_counts.keys(), label_counts.values(), color=colors)
    mplcyberpunk.add_bar_gradient(bars=bars)
    for bar, count in zip(bars, label_counts.values()):
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), str(count), ha='center', va='bottom',
                 color='black')
    plt.ylabel('Count')
    plt.title(title)
    plt.show()


# Load and plot data distributions
def process_data_in_batches(data_dir, labels, image_size, batch_size=1000):
    label_counts = {label: 0 for label in labels}
    X_all, y_all = [], []

    for X_batch, y_batch in load_images_batch(data_dir, labels, image_size, batch_size):
        X_all.extend(X_batch)
        y_all.extend(y_batch)
        for label in y_batch:
            label_counts[label] += 1

    return np.array(X_all), np.array(y_all), label_counts


X1, y1, train_counts = process_data_in_batches(train_dir, labels, image_size)
X2, y2, test_counts = process_data_in_batches(test_dir, labels, image_size)

plot_distribution(train_counts, 'Distribution of Training Labels')
plot_distribution(test_counts, 'Distribution of Testing Labels')

# Combine and shuffle datasets
X, y = np.concatenate((X1, X2), axis=0), np.concatenate((y1, y2), axis=0)
X, y = shuffle(X, y, random_state=42)
combined_counts = {label: np.sum(y == label) for label in labels}
plot_distribution(combined_counts, 'Distribution of Labels in Combined Dataset')

# Split data into training, validation, and test sets
X_train_val, X_test, y_train_val, y_test = train_test_split(X, y, test_size=0.1, stratify=y, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=1 / 9, stratify=y_train_val,
                                                  random_state=42)

# Plot distributions for each set
plot_distribution({label: np.sum(y_train == label) for label in labels}, 'Distribution of Training Labels')
plot_distribution({label: np.sum(y_val == label) for label in labels}, 'Distribution of Validation Labels')
plot_distribution({label: np.sum(y_test == label) for label in labels}, 'Distribution of Testing Labels')

# Data augmentation
datagen = ImageDataGenerator(
    rotation_range=20, width_shift_range=0.1, height_shift_range=0.1,
    shear_range=0.2, zoom_range=0.2, horizontal_flip=True, vertical_flip=True, fill_mode='nearest'
)


# Augment training data in batches
def augment_data_batch(X, y, num_augmented_per_image=5, batch_size=1000):
    for i in range(0, len(X), batch_size):
        X_batch, y_batch = X[i:i + batch_size], y[i:i + batch_size]
        X_augmented, y_augmented = [], []
        for image, label in zip(X_batch, y_batch):
            X_augmented.append(image)
            y_augmented.append(label)
            image = image.reshape((1,) + image.shape)
            aug_iter = 0
            for batch in datagen.flow(image, batch_size=1):
                X_augmented.append(batch[0])
                y_augmented.append(label)
                aug_iter += 1
                if aug_iter >= num_augmented_per_image:
                    break
        yield np.array(X_augmented), np.array(y_augmented)


# Process augmented data in batches
X_train_augmented, y_train_augmented = [], []
for X_aug_batch, y_aug_batch in augment_data_batch(X_train, y_train):
    X_train_augmented.extend(X_aug_batch)
    y_train_augmented.extend(y_aug_batch)

X_train_augmented, y_train_augmented = np.array(X_train_augmented), np.array(y_train_augmented)
plot_distribution({label: np.sum(y_train_augmented == label) for label in labels},
                  'Distribution of Augmented Training Labels')

# Encode labels and normalize images
le = LabelEncoder()
y_train_encoded = le.fit_transform(y_train_augmented)
y_val_encoded = le.transform(y_val)
y_test_encoded = le.transform(y_test)
y_train_cat = to_categorical(y_train_encoded, len(labels))
y_val_cat = to_categorical(y_val_encoded, len(labels))
y_test_cat = to_categorical(y_test_encoded, len(labels))

X_train_normalized = X_train_augmented.astype('float32') / 255
X_val_normalized = X_val.astype('float32') / 255
X_test_normalized = X_test.astype('float32') / 255

# New Model Building Using VGG16
base_model = VGG16(weights='imagenet', include_top=False, input_shape=(image_size, image_size, 3))

for layer in base_model.layers:
    layer.trainable = False

model = Sequential([
    base_model,
    GlobalAveragePooling2D(),
    Dense(1024, activation='relu'),
    Dropout(0.5),
    Dense(len(labels), activation='softmax')
])

model.compile(optimizer=Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()

# Preprocess inputs according to VGG16's requirements
X_train_preprocessed = preprocess_input(X_train_normalized * 255)
X_val_preprocessed = preprocess_input(X_val_normalized * 255)
X_test_preprocessed = preprocess_input(X_test_normalized * 255)

# Train the model
history = model.fit(X_train_preprocessed, y_train_cat, epochs=25, batch_size=32,
                    validation_data=(X_val_preprocessed, y_val_cat), verbose=1)

# Evaluate the model on the test set
test_loss, test_accuracy = model.evaluate(X_test_preprocessed, y_test_cat, verbose=0)
print(f"Test accuracy: {test_accuracy:.4f}")

# Generate predictions on the test set
y_test_pred = model.predict(X_test_preprocessed, batch_size=32)
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

# Calculate and print metrics
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
sensitivity_per_class = recall_score(y_test_encoded, y_test_pred_classes, average=None)

for i, label in enumerate(labels):
    print(f"Specificity for {label}: {specificity_per_class[i]:.4f}")
    print(f"Sensitivity (Recall) for {label}: {sensitivity_per_class[i]:.4f}")

# Calculate and plot AUC-ROC for each class
plt.figure(figsize=(10, 8))
for i in range(len(labels)):
    fpr, tpr, _ = roc_curve(y_test_cat[:, i], y_test_pred[:, i])
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, label=f'{labels[i]} (AUC = {roc_auc:.4f})')

plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curves')
plt.legend(loc='lower right')
plt.show()