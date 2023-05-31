import numpy as np
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay
import tensorflow as tf
from keras.utils import image_dataset_from_directory
 
# ucitavanje podataka iz odredenog direktorija 
train_ds = image_dataset_from_directory( 
    directory='C:/Users/student/Downloads/archive/Train', 
    labels='inferred', 
    label_mode='categorical', 
    batch_size=32, 
    image_size=(48, 48)) 

test_ds = image_dataset_from_directory( 
    directory='C:/Users/student/Downloads/archive/Test', 
    labels='inferred', 
    label_mode='categorical', 
    batch_size=32, 
    image_size=(48, 48)) 

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
model.add(tf.keras.layers.Dense(512, activation='relu'))
model.add(tf.keras.layers.Dropout(0.2))
model.add(tf.keras.layers.Dense(43, activation='softmax'))

# TODO: definiraj karakteristike procesa ucenja pomocu .compile()

model.compile(loss='categorical_crossentropy', 
              optimizer='sgd', 
              metrics=['accuracy'])

# TODO: provedi ucenje mreze
model.fit(train_ds, epochs=5, batch_size=128)

# TODO: Prikazi test accuracy i matricu zabune
loss_and_metrics = model.evaluate(test_ds, batch_size=128) 

test_labels = []

for images,labels in test_ds:
    test_labels.extend(np.argmax(labels.numpy()), axis=1)

y_test = np.array(test_labels)
y_pred = np.argmax(model.predict(test_ds), axis=1)

cm = confusion_matrix(y_test, y_pred)
model.summary() 
cm_display = ConfusionMatrixDisplay(cm,display_labels=range(43)).plot()
plt.show()

# TODO: spremi model
model.save("model.h5")
