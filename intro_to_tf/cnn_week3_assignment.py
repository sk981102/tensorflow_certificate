# Assignment
## Using the mnist dataset, reaching 99.8% accuracy on the training data set

import tensorflow as tf

mnist = tf.keras.datasets.mnist
(training_images, training_labels), (test_images, test_labels) = mnist.load_data()
size = len(training_images)
size2 = len(test_images)
training_images = training_images.reshape(size, 28, 28, 1)
training_images = training_images / 255.0
test_images = test_images.reshape(size2, 28, 28, 1)
test_images = test_images / 255.0


class myCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
        if logs.get('accuracy') > 0.998:
            print("\nReached 99.8% accuracy so cancelling training!")
            self.model.stop_training = True


callbacks = myCallback()

model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(64, (4, 4), activation='relu', input_shape=(28, 28, 1)),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Conv2D(64, (4, 4), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(10, activation='softmax')
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.summary()
model.fit(training_images, training_labels, epochs=3, callbacks=[callbacks])

test_loss = model.evaluate(test_images, test_labels)
print(test_loss)
