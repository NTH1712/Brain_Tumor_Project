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
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical

# Define labels and directories
labels = ['glioma_tumor', 'no_tumor', 'meningioma_tumor', 'pituitary_tumor']
train_dir = 'C:/Users/nthoa/PycharmProjects/Brain Tumor Project/Training'
test_dir = 'C:/Users/nthoa/PycharmProjects/Brain Tumor Project/Testing'
image_size = 150

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

# Plot label distribution
def plot_distribution(label_counts, title):
    plt.figure(figsize=(12, 5))
    colors = ["C0", "C1", "C2", "C3"]
    bars = plt.bar(label_counts.keys(), label_counts.values(), color=colors)
    mplcyberpunk.add_bar_gradient(bars=bars)
    for bar, count in zip(bars, label_counts.values()):
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), str(count), ha='center', va='bottom', color='black')
    plt.ylabel('Count')
    plt.title(title)
    plt.show()

# Load and plot data distributions
X1, y1 = load_images(train_dir, labels, image_size)
X2, y2 = load_images(test_dir, labels, image_size)
plot_distribution({label: np.sum(y1 == label) for label in labels}, 'Distribution of Training Labels')
plot_distribution({label: np.sum(y2 == label) for label in labels}, 'Distribution of Testing Labels')

# Combine and shuffle datasets
X, y = np.concatenate((X1, X2), axis=0), np.concatenate((y1, y2), axis=0)
X, y = shuffle(X, y, random_state=42)
plot_distribution({label: np.sum(y == label) for label in labels}, 'Distribution of Labels in Combined Dataset')

# Split data into training, validation, and test sets
X_train_val, X_test, y_train_val, y_test = train_test_split(X, y, test_size=0.1, stratify=y, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=1/9, stratify=y_train_val, random_state=42)

# Plot distributions for each set
plot_distribution({label: np.sum(y_train == label) for label in labels}, 'Distribution of Training Labels')
plot_distribution({label: np.sum(y_val == label) for label in labels}, 'Distribution of Validation Labels')
plot_distribution({label: np.sum(y_test == label) for label in labels}, 'Distribution of Testing Labels')

# Data augmentation
datagen = ImageDataGenerator(
    rotation_range=20, width_shift_range=0.1, height_shift_range=0.1,
    shear_range=0.2, zoom_range=0.2, horizontal_flip=True, vertical_flip=True, fill_mode='nearest'
)

# Augment training data
def augment_data(X, y, num_augmented_per_image=5):
    X_augmented, y_augmented = [], []
    for image, label in zip(X, y):
        X_augmented.append(image)
        y_augmented.append(label)
        image = image.reshape((1,) + image.shape)
        i = 0
        for batch in datagen.flow(image, batch_size=1):
            X_augmented.append(batch[0])
            y_augmented.append(label)
            i += 1
            if i >= num_augmented_per_image:
                break
    return np.array(X_augmented), np.array(y_augmented)

X_train_augmented, y_train_augmented = augment_data(X_train, y_train)
plot_distribution({label: np.sum(y_train_augmented == label) for label in labels}, 'Distribution of Augmented Training Labels')