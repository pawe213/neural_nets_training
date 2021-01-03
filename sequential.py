import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
from keras_preprocessing.image import ImageDataGenerator

import numpy as np
from matplotlib import pyplot as plt
from scipy import stats
# load data
(train_dataset, train_labels), (test_dataset, test_labels) = tf.keras.datasets.cifar10.load_data()
validation_dataset, test_dataset = test_dataset[:5000], test_dataset[5000:]
validation_labels, test_labels = test_labels[:5000], test_labels[5000:]
print(f"""channel 0 max min: {train_dataset[0, :, :, 0].max(), train_dataset[0, :, :, 0].min() } \n
channel 1 max min: {train_dataset[0, :, :, 1].max(), train_dataset[0, :, :, 0].min() } \n
channel 2 max min: {train_dataset[0, :, :, 2].max(), train_dataset[0, :, :, 0].min() } \n
""")

train_dataset, test_dataset, validation_dataset = train_dataset/255.0, test_dataset/255.0, validation_dataset/255.0

# print(f'Train and test example array shape: \n train: {train_dataset[0].shape} \n test:  {test_dataset[0].shape}')
# f, axarr = plt.subplots(1, 2, figsize=(4, 4))
# axarr[0].imshow(train_dataset[2])
# axarr[0].set_title('class_nr: ' + str(test_labels[2]))
# axarr[1].imshow(train_dataset[5])
# axarr[1].set_title('class_nr: ' + str(test_labels[5]))
data_augmentation = tf.keras.Sequential([
  layers.experimental.preprocessing.RandomFlip("horizontal_and_vertical"),
  layers.experimental.preprocessing.RandomRotation(0.2),
])


sequential = keras.Sequential(
    [data_augmentation,
     layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3), strides=(1, 1), padding="same"),
     layers.BatchNormalization(axis=3),
     layers.MaxPooling2D((2, 2), padding="same"),
     layers.Conv2D(64, (3, 3), activation='relu'),
     layers.BatchNormalization(axis=3),
     layers.MaxPooling2D((2, 2)),
     layers.Conv2D(64, (3, 3), activation='relu'),
     layers.BatchNormalization(axis=3),
     layers.Flatten(),
     layers.Dense(64, activation='relu'),
     layers.Dropout(0.2),
     layers.Dense(32, activation='relu'),
     layers.Dropout(0.2),
     layers.Dense(10)]
)

# sequential = keras.Sequential(
#     [layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)),
#      layers.MaxPooling2D((2, 2)),
#      layers.Conv2D(64, (3, 3), activation='relu'),
#      layers.MaxPooling2D((2, 2)),
#      layers.Conv2D(64, (3, 3), activation='relu'),
#      layers.Flatten(),
#      layers.Dense(64, activation='relu'),
#      layers.Dense(10)]
# )

opt = keras.optimizers.Adam(learning_rate=0.0007)
sequential.compile(optimizer=opt, loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True), metrics=['accuracy'])
# sequential.summary()

sequential.fit(x=train_dataset, y=train_labels, epochs=14, batch_size=128,
               validation_data=(validation_dataset, validation_labels))

# sequential.fit_generator(train_generator.flow(train_dataset, train_labels, batch_size=64), epochs=14,
#                validation_data=val_generator.flow(validation_dataset, validation_labels, batch_size=64))

sequential.evaluate(test_dataset, test_labels)