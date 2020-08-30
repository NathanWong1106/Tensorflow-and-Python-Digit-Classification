import sys
import numpy as np
import tensorflow as tf

rows, cols = 28, 28
DIGITS = 10 # 0-9
EPOCHS = 10

(imageTrain, labelTrain), (imageTest, labelTest) = tf.keras.datasets.mnist.load_data()

# normlalize to values from 0 through 1
imageTrain, imageTest = imageTrain / 255.0, imageTest / 255.0
labelTrain = tf.keras.utils.to_categorical(labelTrain)
labelTest = tf.keras.utils.to_categorical(labelTest)

# flatten image to 1D vector / 1D array of length 28*28
imageTrain = imageTrain.reshape(imageTrain.shape[0], rows, cols, 1)
imageTest = imageTest.reshape(imageTest.shape[0], rows, cols, 1)

# make and compile the model
model = tf.keras.models.Sequential([
    # Convolution: feature extraction
    tf.keras.layers.Conv2D(32, (3,3), activation="relu", input_shape=(rows, cols, 1)),
    # Max Pooling: reduces size by selecting max value in a 2x2 pooling size
    tf.keras.layers.MaxPooling2D(pool_size=(2,2)),
    #flatten to connect layers
    tf.keras.layers.Flatten(),

    tf.keras.layers.Dense(128, activation="relu"),
    tf.keras.layers.Dropout(0.5),

    # softmax activation returns probabilities of results
    tf.keras.layers.Dense(DIGITS, activation="softmax")
])

model.compile(
    optimizer="adam",
    loss = "categorical_crossentropy",
    metrics=['accuracy']
)
model.fit(x=imageTrain, y=labelTrain, epochs=EPOCHS)
model.evaluate(imageTest, labelTest)

if len(sys.argv) == 2:
    model.save(sys.argv[1])
    print("Model saved as " + sys.argv[1])

