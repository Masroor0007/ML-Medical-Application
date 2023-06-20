import numpy as np 
import pandas as pd 
import cv2
import tensorflow as tf
import os

from keras.preprocessing.image import ImageDataGenerator
from tqdm import tqdm
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split


XData = []
yData = []
XBenign = []

for dirname, _, filenames in os.walk(r'D:\Team-Hextech\Mini-Prroject\Lung_Cancer_Data\The IQ-OTHNCCD lung cancer dataset'):
    for filename in filenames:
        if (filename[-3:] != "txt"):
            category = " ".join(filename.split()[:2])
            img = cv2.imread(os.path.join(dirname, filename))
            img = cv2.resize(img, (512, 512))
            img = img / 255
            
            if(category != "Bengin case"):
                XData.append(img)
                yData.append(category)
                
            else:
                XBenign.append(img)
            
    print(len(filenames))

datagen = ImageDataGenerator(
    width_shift_range=0.2,
    height_shift_range=0.3
)
x = np.array(XBenign)
datagen.fit(x)

batch_size = 4

augmented_images = []
for batch in tqdm(datagen.flow(x, batch_size=batch_size)):
    augmented_images.append(batch)
    if len(augmented_images) * batch_size >= (510 - 120):
        break

XBenign_augmented = np.concatenate(augmented_images, axis=0)

del(x, augmented_images)

yBenign = ["Bengin case" for i in range(XBenign_augmented.shape[0])]
XData.extend(XBenign_augmented)
yData.extend(yBenign)
del(yBenign)

XData = np.array(XData)
encoder = LabelEncoder()
encoder.fit(yData)

yEncoded = encoder.transform(yData)
encoder.inverse_transform([0, 1, 2])

Xtrain,Xtest,ytrain,ytest = train_test_split(XData, yEncoded, shuffle = True)

model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(512, 512, 3)),
    tf.keras.layers.MaxPooling2D((2, 2)) ,
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(256, activation='relu'),
    tf.keras.layers.Dense(3, activation='softmax')
])

model.compile(optimizer='adam',loss=tf.keras.losses.SparseCategoricalCrossentropy(
            from_logits=True),metrics=['accuracy'])

callback = tf.keras.callbacks.EarlyStopping(patience = 7)
history = model. fit(Xtrain, ytrain, epochs=7, batch_size=batch_size,
                    validation_data=(Xtest, ytest), callbacks = [callback])

model.save("lung cancer.h5")

print("Fayez bot hai :P")