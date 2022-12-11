from keras.models import load_model
import numpy as np
import os
import cv2
import matplotlib.pyplot as plt

model_load = load_model('model_mnist.h5')



img_number = 1
print()
while os.path.isfile(f"digits/digit{img_number}.jpg"):
    try:
        img = cv2.imread(f"digits/digit{img_number}.jpg")[:,:,0]
        img = np.invert(np.array([img]))
        prediction = model_load.predict(img)
        print(f"this is probably {np.argmax(prediction)}")
        plt.imshow(img[0], cmap='binary')
        plt.show()

    except:
        print("Error")

    finally:
        img_number += 1