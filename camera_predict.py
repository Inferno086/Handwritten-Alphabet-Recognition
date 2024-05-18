import cv2
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

def preProcessor(image):
    for i in range(28):
        for j in range(28):
            if image[i][j] > 130:
                image[i][j] = 255
            elif image[i][j] < 80:
                image[i][j] = 0

    resized_imagel = cv2.bitwise_not(image)
    resized_imagel = resized_imagel/255.0
    return resized_imagel

labels = ['0','1','2','3','4','5','6','7','8','9',
          'A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y','Z',
          'a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p','q','r','s','t','u','v','w','x','y','z'
         ]

cap = cv2.VideoCapture(0)

num_less_than_equalto_8 = False

while not num_less_than_equalto_8:
    number = int(input("Enter the no. of letters in the word: "))
    if number > 8:
        print('The entered number is greater than 8! Try Again.')
    else:
        num_less_than_equalto_8 = True

while True:
    ret, frame = cap.read()
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    for i in range(number):
        frame = cv2.rectangle(frame, (300 + i*40, 50), (340 + i*40, 90), 0 ,1)

    cv2.imshow('Align the Character', frame)
    image = frame[50:90, 300:340]
    image2 = frame[50:90, 340:380]
    image3 = frame[50:90, 380:420]
    image4 = frame[50:90, 420:460]
    image5 = frame[50:90, 460:500]
    image6 = frame[50:90, 500:540]
    image7 = frame[50:90, 540:580]
    image8 = frame[50:90, 580:620]

    display = frame[50:90, 300:(300+number*40)]
    display1 = display

    resized_image = cv2.resize(image, (28,28))
    resized_image2 = cv2.resize(image2, (28,28))
    resized_image3 = cv2.resize(image3, (28,28))
    resized_image4 = cv2.resize(image4, (28,28))
    resized_image5 = cv2.resize(image5, (28,28))
    resized_image6 = cv2.resize(image6, (28,28))
    resized_image7 = cv2.resize(image7, (28,28))
    resized_image8 = cv2.resize(image8, (28,28))
    display = cv2.resize(display, (28*number,28))

    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

#Preprocessing the Captured image

resized_image = preProcessor(resized_image)
resized_image2 = preProcessor(resized_image2)
resized_image3 = preProcessor(resized_image3)
resized_image4 = preProcessor(resized_image4)
resized_image5 = preProcessor(resized_image5)

display = cv2.bitwise_not(display)
for i in range(len(display)):
    for j in range(len(display[0])):
        if display[i][j] > 130:
            display[i][j] = 255
        elif display[i][j] < 80:
            display[i][j] = 0

display = display/255.0

if number == 1:
    images = [resized_image]
elif number == 2:
    images = [resized_image, resized_image2]
elif number == 3:
    images = [resized_image, resized_image2, resized_image3]
elif number == 4:
    images = [resized_image, resized_image2, resized_image3, resized_image4]
elif number == 5:
    images = [resized_image, resized_image2, resized_image3, resized_image4,
               resized_image5]
elif number == 6:
    images = [resized_image, resized_image2, resized_image3, resized_image4, 
              resized_image5, resized_image6]
elif number == 7:
    images = [resized_image, resized_image2, resized_image3, resized_image4, 
              resized_image5, resized_image6, resized_image7]
elif number == 8:
    images = [resized_image, resized_image2, resized_image3, resized_image4, 
              resized_image5, resized_image6, resized_image7, resized_image8]



model = tf.keras.models.load_model('my_CNN_model2.h5')

predictions = model.predict(np.array(images))

for i in range(number):
    if i == 0:
        print('Prediction: ', end='')

    print(f'{labels[np.argmax(predictions[i])]}', end='')

f = plt.figure()
plt.imshow(display, cmap=plt.cm.binary)
f.set_figwidth(3)
f.set_figheight(3)
plt.colorbar()
plt.grid(False)
plt.show()