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
train_dataset_p = keras.applications.resnet.preprocess_input(train_dataset)
validation_dataset_p = keras.applications.resnet.preprocess_input(validation_dataset)
train_labels= tf.keras.utils.to_categorical(train_labels, 10)
test_labels = tf.keras.utils.to_categorical(test_labels, 10)
validation_labels = tf.keras.utils.to_categorical(validation_labels, 10)
input_shape = train_dataset[0].shape
X_input = Input(input_shape)

upscale = tf.keras.layers.Lambda(lambda x: tf.image.resize_with_pad(x, 224, 224,
                                                                  method=tf.image.ResizeMethod.BILINEAR))(X_input)

res = tf.keras.applications.ResNet50(include_top=False, input_tensor=upscale, input_shape=(224, 224, 3), pooling='max', weights='imagenet')
for layer in res.layers[:-30]:
    layer.trainable = False

X = res.output
X = tf.keras.layers.Flatten()(X)
X = tf.keras.layers.BatchNormalization()(X)
X = tf.keras.layers.Dense(256, activation='relu')(X)
X = tf.keras.layers.Dropout(0.2)(X)
X = tf.keras.layers.BatchNormalization()(X)
X = tf.keras.layers.Dense(128, activation='relu')(X)
X = tf.keras.layers.Dropout(0.2)(X)
X = tf.keras.layers.BatchNormalization()(X)
X = tf.keras.layers.Dense(64, activation='relu')(X)
X = tf.keras.layers.Dropout(0.2)(X)
X = tf.keras.layers.Dense(10, activation='softmax')(X)


opt = keras.optimizers.Adam(learning_rate=0.001)
loss = tf.keras.losses.CategoricalCrossentropy(from_logits=True)

model = Model(inputs=X_input, outputs=X, name='NotMyResNet')

model.compile(optimizer=opt, loss=loss, metrics=['accuracy'])
model.fit(train_dataset_p, train_labels, batch_size=128, epochs=30, validation_data=(validation_dataset_p, validation_labels))
