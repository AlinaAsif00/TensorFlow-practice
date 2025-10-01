# import tensorflow as tf
# from tensorflow import keras
# from tensorflow.keras import layers

# (X_train, y_train), (X_test, y_test) = keras.datasets.mnist.load_data()

# X_train = X_train.reshape(-1, 28, 28, 1).astype("float32") / 255.0
# X_test  = X_test.reshape(-1, 28, 28, 1).astype("float32") / 255.0

# inputs = keras.Input(shape=(28, 28, 1))

# x = layers.Conv2D(32, (3, 3), activation="relu")(inputs)
# x = layers.MaxPooling2D((2, 2))(x)                       
# x = layers.Conv2D(64, (3, 3), activation="relu")(x)      
# x = layers.MaxPooling2D((2, 2))(x)                        
# x = layers.Flatten()(x)                                 
# x = layers.Dense(128, activation="relu")(x)        

# outputs = layers.Dense(10, activation="softmax")(x)     

# model = keras.Model(inputs=inputs, outputs=outputs, name="mnist_cnn_functional")

# model.compile(optimizer="adam",
#               loss="sparse_categorical_crossentropy",
#               metrics=["accuracy"])


# model.fit(X_train, y_train, epochs=5, batch_size=64, validation_split=0.1, verbose=1)
# loss, acc = model.evaluate(X_test, y_test, verbose=0)
# print("\nTest accuracy:", acc)

#######################################################################################

import tensorflow as tf
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt
import numpy as np

(X_train, y_train), (X_test, y_test) = tf.keras.datasets.cifar10.load_data()


X_train, X_test = X_train / 255.0, X_test / 255.0

class_names = ['airplane','car','bird','cat','deer',
               'dog','frog','horse','ship','truck']

print("Train shape:", X_train.shape)
print("Test shape:", X_test.shape)
model = models.Sequential([
    layers.Conv2D(32, (3,3), activation='relu', input_shape=(32,32,3)),
    layers.MaxPooling2D((2,2)),

    layers.Conv2D(64, (3,3), activation='relu'),
    layers.MaxPooling2D((2,2)),

    layers.Conv2D(128, (3,3), activation='relu'),
    layers.Flatten(),

    layers.Dense(128, activation='relu'),
    layers.Dense(10, activation='softmax')
])

model.summary()
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

history = model.fit(X_train, y_train, epochs=10, 
                    validation_data=(X_test, y_test))

test_loss, test_acc = model.evaluate(X_test, y_test, verbose=2)
print("\nTest Accuracy:", test_acc)

plt.plot(history.history['accuracy'], label='train acc')
plt.plot(history.history['val_accuracy'], label='val acc')
plt.legend()
plt.show()
predictions = model.predict(X_test[:5])

for i in range(5):
    plt.imshow(X_test[i])
    plt.title(f"Actual: {class_names[y_test[i][0]]}, Predicted: {class_names[np.argmax(predictions[i])]}")
    plt.show()
