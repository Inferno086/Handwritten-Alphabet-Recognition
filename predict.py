import cv2
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

N = 14

labels = ['0','1','2','3','4','5','6','7','8','9',
          'A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y','Z',
          'a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p','q','r','s','t','u','v','w','x','y','z'
         ]


img = [
        cv2.imread('images/0.jpg', cv2.IMREAD_GRAYSCALE),
        cv2.imread('images/1.jpg', cv2.IMREAD_GRAYSCALE),
        cv2.imread('images/2.jpg', cv2.IMREAD_GRAYSCALE),
        cv2.imread('images/3.jpeg', cv2.IMREAD_GRAYSCALE),
        cv2.imread('images/4.jpg', cv2.IMREAD_GRAYSCALE),
        cv2.imread('images/5.jpg', cv2.IMREAD_GRAYSCALE),
        cv2.imread('images/6.jpg', cv2.IMREAD_GRAYSCALE),
        cv2.imread('images/7.jpg', cv2.IMREAD_GRAYSCALE),
        cv2.imread('images/8.jpg', cv2.IMREAD_GRAYSCALE),
        cv2.imread('images/9.jpg', cv2.IMREAD_GRAYSCALE),
        cv2.imread('images/E.jpg', cv2.IMREAD_GRAYSCALE),
        cv2.imread('images/Q.jpg', cv2.IMREAD_GRAYSCALE),
        cv2.imread('images/R.jpg', cv2.IMREAD_GRAYSCALE),
        cv2.imread('images/K.jpg', cv2.IMREAD_GRAYSCALE),
    ]

# # Below commented code has to be used with NN Model
# for i in range(k):
#     img[i] = img[i].flatten()

# for i in range(k):
#     for j in range(784):
#         if img[i][j] > 150:
#             img[i][j] = 255
#         elif img[i][j] < 50:
#             img[i][j] = 0

# Comment out below code if not using CNN Model
for i in range(N):
    for j in range(28):
        for k in range(28):
            if img[i][j][k] > 150:
                img[i][j][k] = 255
            elif img[i][j][k] < 50:
                img[i][j][k] = 0



img = np.array(img)
img = cv2.bitwise_not(img)
img = img/255.0


model = tf.keras.models.load_model('my_CNN_model2.h5')

predictions = model.predict(img)

for i in range(N):
    f = plt.figure()

    plt.imshow(img[i], cmap=plt.cm.binary)
    # plt.imshow(img[i].reshape(28,28), cmap=plt.cm.binary)
    print(labels[np.argmax(predictions[i])])

    f.set_figwidth(3)
    f.set_figheight(3)
    plt.colorbar()
    plt.grid(False)
    plt.show()