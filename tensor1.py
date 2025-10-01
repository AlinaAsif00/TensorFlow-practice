import tensorflow as tf
from tensorflow import keras
import numpy as np

# X = np.array(range(-10, 11), dtype=float)
# y = 2 * X + 1

# model = keras.Sequential([
#     keras.layers.Dense(1, input_shape=[1])   
# ])

# model.compile(optimizer='sgd', loss='mse')

# model.fit(X, y, epochs=5, verbose=1)

# print(model.predict(np.array([[20.0]])))

###################################################################
# from sklearn.datasets import make_classification
# from sklearn.model_selection import train_test_split

# X, y = make_classification(
#     n_samples=1000,
#     n_features=2, 
#     n_informative=2, 
#     n_redundant=0,     
#     n_repeated=0,  
#     n_classes=2, 
#     random_state=42
# )

# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# model = keras.Sequential([
#     keras.Input(shape=(2,)),
#     keras.layers.Dense(8, activation='relu'),
#     keras.layers.Dense(1, activation='sigmoid')
# ])

# model.compile(optimizer='adam',
#              loss='binary_crossentropy', 
#              metrics=['accuracy'])

# model.fit(X_train, y_train, epochs=20, validation_data=(X_test, y_test), verbose=1)

# loss, acc = model.evaluate(X_test, y_test)
# print("Test Accuracy:", acc)

########################################################################
# import numpy as np
# import tensorflow as tf
# from tensorflow import keras
# from sklearn.datasets import load_breast_cancer
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import StandardScaler

# data = load_breast_cancer()
# X, y = data.data, data.target 

# X_train, X_test, y_train, y_test = train_test_split(
#     X, y, test_size=0.2, random_state=42
# )
# scaler = StandardScaler()
# X_train = scaler.fit_transform(X_train)
# X_test = scaler.transform(X_test)

# model = keras.Sequential([
#     keras.Input(shape=(X_train.shape[1],)),  
#     keras.layers.Dense(16, activation='relu'),
#     keras.layers.Dense(8, activation='relu'),
#     keras.layers.Dense(1, activation='sigmoid')  
# ])
# model.compile(
#     optimizer='adam',
#     loss='binary_crossentropy',
#     metrics=['accuracy']
# )
# model.fit(
#     X_train, y_train,
#     validation_split=0.2,
#     epochs=10,
#     batch_size=32,
#     verbose=1
# )
# loss, acc = model.evaluate(X_test, y_test, verbose=0)
# print(f"Test Accuracy: {acc:.4f}")

#######################################################################
# import numpy as np
# from sklearn.datasets import load_iris
# from tensorflow.keras.utils import to_categorical

# iris = load_iris()
# X = iris.data  
# y = iris.target 
# y = to_categorical(y)

# model = keras.Sequential([
#     keras.layers.Dense(10, activation='relu', input_shape=(4,)),
#     keras.layers.Dense(3, activation='softmax')
# ])
# model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
# model.fit(X, y, epochs=50, verbose=1)
# pred = model.predict(np.array([X[0]]))
# print("Predicted:", pred)
# print("Actual:", y[0])
