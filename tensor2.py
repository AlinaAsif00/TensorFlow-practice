import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

(X_train, y_train), (X_test, y_test) = keras.datasets.mnist.load_data()

X_train = X_train.reshape(-1, 28, 28, 1).astype("float32") / 255.0
X_test  = X_test.reshape(-1, 28, 28, 1).astype("float32") / 255.0

inputs = keras.Input(shape=(28, 28, 1))

x = layers.Conv2D(32, (3, 3), activation="relu")(inputs)
x = layers.MaxPooling2D((2, 2))(x)                       
x = layers.Conv2D(64, (3, 3), activation="relu")(x)      
x = layers.MaxPooling2D((2, 2))(x)                        
x = layers.Flatten()(x)                                 
x = layers.Dense(128, activation="relu")(x)        

outputs = layers.Dense(10, activation="softmax")(x)     

model = keras.Model(inputs=inputs, outputs=outputs, name="mnist_cnn_functional")

model.compile(optimizer="adam",
              loss="sparse_categorical_crossentropy",
              metrics=["accuracy"])


model.fit(X_train, y_train, epochs=5, batch_size=64, validation_split=0.1, verbose=1)
loss, acc = model.evaluate(X_test, y_test, verbose=0)
print("\nTest accuracy:", acc)

