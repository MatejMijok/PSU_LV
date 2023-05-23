from matplotlib import pyplot as plt
from skimage.transform import resize
from skimage import color
import matplotlib.image as mpimg
import numpy as np
import keras

filename = 'test.png'

img = mpimg.imread(filename)[:,:,:3]
img = color.rgb2gray(img)
img = resize(img, (28, 28))


plt.figure()
plt.imshow(img, cmap=plt.get_cmap('gray'))
plt.show()


img = img.reshape(1, 28, 28, 1)
img = img.astype('float32')


# TODO: ucitaj model
model = keras.models.load_model('model.h5')

# TODO: napravi predikciju 
y_pred = model.predict(img)
y_pred = np.argmax(y_pred, axis=1)

# TODO: ispis rezultat
print("------------------------")
print(y_pred)
