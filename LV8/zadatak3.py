import numpy as np
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay
import tensorflow as tf
from keras.utils import image_dataset_from_directory
from keras.regularizers import l2

# ucitavanje podataka iz odredenog direktorija 
train_ds = image_dataset_from_directory( 
    directory='/home/matej/LV8/archive/Train', 
    labels='inferred', 
    label_mode='categorical', 
    batch_size=32, 
    image_size=(48, 48)) 

test_ds = image_dataset_from_directory( 
    directory='/home/matej/LV8/archive/Test', 
    labels='inferred', 
    label_mode='categorical', 
    batch_size=32, 
    image_size=(48, 48),
    shuffle=False) 

# TODO: kreiraj model pomocu keras.Sequential(); prikazi njegovu strukturu

model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Conv2D(32, (3, 3), activation='relu',  input_shape=(48, 48, 3)))
model.add(tf.keras.layers.Conv2D(32, (3, 3), activation='relu'))
model.add(tf.keras.layers.MaxPooling2D((2, 2)))
model.add(tf.keras.layers.Dropout(0.2))
model.add(tf.keras.layers.Conv2D(32, (3, 3), activation='relu'))
model.add(tf.keras.layers.Conv2D(32, (3, 3), activation='relu'))
model.add(tf.keras.layers.MaxPooling2D((2, 2)))
model.add(tf.keras.layers.Dropout(0.2))
model.add(tf.keras.layers.Conv2D(32, (3, 3), activation='relu'))
model.add(tf.keras.layers.Conv2D(32, (3, 3), activation='relu'))
model.add(tf.keras.layers.MaxPooling2D((2, 2)))
model.add(tf.keras.layers.Dropout(0.2))
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(2048, activation='relu'))
model.add(tf.keras.layers.Dense(512, activation='relu'))
model.add(tf.keras.layers.Dropout(0.5))
model.add(tf.keras.layers.Dense(43, activation='softmax'))

# TODO: definiraj karakteristike procesa ucenja pomocu .compile()

model.compile(loss='categorical_crossentropy', 
              optimizer='sgd', 
              metrics=['accuracy'])

# TODO: provedi ucenje mreze
model.fit(train_ds, epochs=25, batch_size=128)

# TODO: Prikazi test accuracy i matricu zabune
loss, metrics = model.evaluate(test_ds, batch_size=128) 
model.summary() 

print("Loss: ", loss)
print("Accuracy: ", metrics)

y_pred = model.predict(test_ds, batch_size=128)
y_pred = np.argmax(y_pred, axis=1)

y_true = np.concatenate([y for x, y in test_ds], axis=0)
y_true = np.argmax(y_true, axis=1)

cm = confusion_matrix(y_true, y_pred, labels=list(range(43)))

cm_display = ConfusionMatrixDisplay(cm).plot()
plt.show()

# TODO: spremi model
model.save("/home/matej/LV8/model.h5")
