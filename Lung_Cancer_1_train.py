import numpy as np
import cv2
import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import os
import joblib

# Function to fit and save the encoder to a file
def fit_and_save_encoder(labels, encoder_file):
    encoder = LabelEncoder()
    encoder.fit(labels)
    joblib.dump(encoder, encoder_file)
    print("Encoder saved to:", encoder_file)

# Function to load the encoder from a file
def load_encoder(encoder_file):
    return joblib.load(encoder_file)

# Specify the directory path
directory_path = r'/Users/syedmasroor/Desktop/miniproject/dataset'

XData = []
yData = []
XBenign = []

if os.path.isdir(directory_path):
    for dirname, _, filenames in os.walk(directory_path):
        for filename in filenames:
            if filename.endswith(('.jpg', '.jpeg', '.png')):  # Filter by file extensions
                if (filename[-3:] != "txt"):
                    category = " ".join(filename.split()[:2])
                    img = cv2.imread(os.path.join(dirname, filename))
                    img = cv2.resize(img, (512, 512))
                    img = img / 255
                    
                    if category != "Bengin case":
                        XData.append(img)
                        yData.append(category)
                    else:
                        XBenign.append(img)
                
        print(len(filenames))
else:
    print(f"Directory not found: {directory_path}")



datagen = ImageDataGenerator(
    width_shift_range=0.2,
    height_shift_range=0.3
)
x = np.array(XBenign)
datagen.fit(x)

batch_size = 16  # Define your desired batch size

# Generate augmented images in batches
augmented_images = []
for batch in datagen.flow(x, batch_size=batch_size):
    augmented_images.append(batch)
    if len(augmented_images) * batch_size >= (510 - 120):
        break

XBenign_augmented = np.concatenate(augmented_images, axis=0)

del(x, augmented_images)

yBenign = ["Bengin case" for _ in range(XBenign_augmented.shape[0])]
XData.extend(XBenign_augmented)
yData.extend(yBenign)
del(yBenign)

XData = np.array(XData)
encoder = LabelEncoder()
encoder.fit(yData)

yEncoded = encoder.transform(yData)
encoder.inverse_transform([0, 1, 2])

Xtrain, Xtest, ytrain, ytest = train_test_split(XData, yEncoded, shuffle=True)

model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(512, 512, 3)),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(256, activation='relu'),
    tf.keras.layers.Dense(3, activation='softmax')
])

model.compile(optimizer='adam', loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

callback = tf.keras.callbacks.EarlyStopping(patience=7)
history = model.fit(Xtrain, ytrain, epochs=7, batch_size=batch_size,
                    validation_data=(Xtest, ytest), callbacks=[callback])

# Specify the paths and filenames
encoder_file = "//Users/syedmasroor/Desktop/untitled folder/ML-Medical-App1/encoder.pkl"
model_file = "/Users/syedmasroor/Desktop/untitled folder/ML-Medical-App1/lung_cancer.h5"
input_image_path = "/Users/syedmasroor/Desktop/check/check.jpg"

# Fit and save the encoder to a file
fit_and_save_encoder(yData, encoder_file)

# Save the model
model.save(model_file)

# Load the encoder
encoder = load_encoder(encoder_file)

# Load the model
model = tf.keras.models.load_model(model_file)

# # Perform the prediction
# predicted_class = predict_image(input_image_path, model, encoder)

# print("Predicted class:", predicted_class)
