import cv2
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

labels = ['0','1','2','3','4','5','6','7','8','9',
          'A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y','Z',
          'a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p','q','r','s','t','u','v','w','x','y','z'
         ]

cap = cv2.VideoCapture(0)


while True:
    ret, frame = cap.read()
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    frame = cv2.rectangle(frame, (300, 50), (340, 90), 0, 1)


    cv2.imshow('Align the Character', frame)
    image = frame[50:90, 300:340]

    resized_image = cv2.resize(image, (28,28))


    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()


#Preprocessing the Captured image
# resized_image = resized_image.flatten()


for i in range(28):
    for j in range(28):
        if resized_image[i][j] > 130:
            resized_image[i][j] = 255
        elif resized_image[i][j] < 80:
            resized_image[i][j] = 0



resized_image = cv2.bitwise_not(resized_image)
resized_image = resized_image/255.0


images = [resized_image]


model = tf.keras.models.load_model('my_CNN_model2.h5')

predictions = model.predict(np.array(images))

f = plt.figure()
print(f'{labels[np.argmax(predictions[0])]}')
plt.imshow(images[0], cmap=plt.cm.binary)
f.set_figwidth(3)
f.set_figheight(3)
plt.colorbar()
plt.grid(False)
plt.show()