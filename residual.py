import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models, Input, Model

# load data
(train_dataset, train_labels), (test_dataset, test_labels) = tf.keras.datasets.cifar10.load_data()
validation_dataset, test_dataset = test_dataset[:5000], test_dataset[5000:]
validation_labels, test_labels = test_labels[:5000], test_labels[5000:]
print(f"""channel 0 max min: {train_dataset[0, :, :, 0].max(), train_dataset[0, :, :, 0].min() } \n
channel 1 max min: {train_dataset[0, :, :, 1].max(), train_dataset[0, :, :, 0].min() } \n
channel 2 max min: {train_dataset[0, :, :, 2].max(), train_dataset[0, :, :, 0].min() } \n
""")

train_dataset, test_dataset, validation_dataset = train_dataset/255.0, test_dataset/255.0, validation_dataset/255.0

input_shape = train_dataset[0].shape
X_input = Input(input_shape)


def residual_block(X):

        X_shortcut = X
        X = keras.layers.Conv2D(32, (5, 5), padding='same', activation='relu')(X)
        X = keras.layers.BatchNormalization()(X)
        X = keras.layers.Dropout(0.1)(X)
        X = keras.layers.Conv2D(32, (3, 3), padding='same', activation='relu')(X)
        X = keras.layers.BatchNormalization()(X)
        X = keras.layers.Dropout(0.1)(X)
        X = keras.layers.Add()([X_shortcut, X])
        X = keras.layers.Activation('relu')(X)
        return X


X = keras.layers.ZeroPadding2D(padding=(0, 0))(X_input)
X = keras.layers.Conv2D(32, (3, 3), activation='relu')(X)
X = keras.layers.BatchNormalization(axis=3)(X)
X = keras.layers.MaxPooling2D((2, 2), padding='same')(X)
X = residual_block(X)
X = residual_block(X)
X = keras.layers.MaxPooling2D((2, 2), padding='same')(X)
X = residual_block(X)
X = residual_block(X)
# X = keras.layers.Conv2D(64, (3, 3), activation='relu')(X)
# X = keras.layers.BatchNormalization(axis=3)(X)
X = keras.layers.MaxPooling2D((2, 2), padding='valid')(X)
X = keras.layers.Flatten()(X)
X = keras.layers.Dense(64, activation='relu')(X)
X = keras.layers.Dropout(0.1)(X)
X = keras.layers.Dense(32, activation='relu')(X)
X = keras.layers.Dropout(0.1)(X)
X = keras.layers.Dense(10)(X)

loss_ = keras.losses.SparseCategoricalCrossentropy(from_logits=True)
opt = keras.optimizers.Adam(learning_rate=0.0008)

model = Model(inputs=X_input, outputs=X, name='MyResNet')

model.compile(optimizer=opt, loss=loss_, metrics=['accuracy'])

model.fit(train_dataset, train_labels, batch_size=256, epochs=50, verbose=2, validation_data=(validation_dataset, validation_labels))

model.evaluate(test_dataset, test_labels)

