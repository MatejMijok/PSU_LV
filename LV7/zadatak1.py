import numpy as np
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay
import tensorflow as tf

# Model / data parameters
num_classes = 10
input_shape = (28, 28, 1)

# train i test podaci
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

# prikaz karakteristika train i test podataka
print('Train: X=%s, y=%s' % (x_train.shape, y_train.shape))
print('Test: X=%s, y=%s' % (x_test.shape, y_test.shape))

# TODO: prikazi nekoliko slika iz train skupa

plt.figure()
for i in range(0,9):
    plt.subplot(330 + 1 + i)
    pixels = x_train[i].reshape((28,28))
    plt.imshow(pixels, cmap="gray")

# skaliranje slike na raspon [0,1]
x_train_s = x_train.astype("float32") / 255
x_test_s = x_test.astype("float32") / 255

# slike trebaju biti (28, 28, 1)
x_train_s = np.expand_dims(x_train_s, -1)
x_test_s = np.expand_dims(x_test_s, -1)


print("x_train shape:", x_train_s.shape)
print(x_train_s.shape[0], "train samples")
print(x_test_s.shape[0], "test samples")

# pretvori labele
y_train_s = tf.keras.utils.to_categorical(y_train, num_classes)
y_test_s = tf.keras.utils.to_categorical(y_test, num_classes)


# TODO: kreiraj model pomocu keras.Sequential(); prikazi njegovu strukturu

model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Conv2D(32, (3, 3), activation='relu',  input_shape=(28, 28, 1)))
model.add(tf.keras.layers.MaxPooling2D((2, 2)))
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(100, activation='relu'))
model.add(tf.keras.layers.Dense(10, activation='softmax'))

# TODO: definiraj karakteristike procesa ucenja pomocu .compile()

model.compile(loss='categorical_crossentropy', 
              optimizer='sgd', 
              metrics=['accuracy'])

# TODO: provedi ucenje mreze
model.fit(x_train_s, y_train_s, epochs=15, batch_size=128)

# TODO: Prikazi test accuracy i matricu zabune
loss_and_metrics = model.evaluate(x_test_s, y_test_s, batch_size=128) 

y_pred = model.predict(x_test_s, batch_size=128)
y_pred = np.argmax(y_pred, axis=1)
y_test = np.argmax(y_test_s, axis=1)

cm = confusion_matrix(y_test, y_pred)
model.summary() 
cm_display = ConfusionMatrixDisplay(cm).plot()
plt.show()

# TODO: spremi model
model.save("model.h5")
