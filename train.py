import numpy as np
import csv
import tensorflow as tf
import matplotlib.pyplot as plt
import pandas as pd

"""LOADING THE DATASET"""

with open('dataset_final.csv', 'r') as f:
    reader = csv.reader(f)
    data = list(reader)


"""PREPROCESSING THE DATASET"""


#Definitions
labels = ['0','1','2','3','4','5','6','7','8','9',
          'A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y','Z',
          'a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p','q','r','s','t','u','v','w','x','y','z'
         ]

rows = len(data)
columns = len(data[0])

datax = pd.read_csv('datax.csv')
datay = pd.read_csv('datay.csv')
datax = datax.to_numpy()
datay = datay.to_numpy().flatten()

shuffler = np.random.permutation(datax.shape[0])
datax = datax[shuffler]
datay = datay[shuffler]

#Making a copy of daray and testy
datay1 = datay.copy()
testy1 = datay[130000:]


#Splitting the dataset
trainx = datax[:130000, :]
trainy = datay[:130000]
testx = datax[130000:, :]
testy = datay[130000:]

#Normalizing the inputs
trainx = trainx/255.0
testx = testx/255.0


"""**Building the NN Model**"""

# model = tf.keras.Sequential([
#     tf.keras.layers.Dense(784),                     # input layer (1)
#     tf.keras.layers.Dense(256, activation='relu'),  # hidden layer (2)
#     tf.keras.layers.Dense(128, activation='relu'),  # hidden layer (3)
#     tf.keras.layers.Dense(62, activation='softmax') # output layer (4)
# ])

# """**Compile the Model**"""

# model.compile(optimizer='adam',
#               loss='sparse_categorical_crossentropy',
#               metrics=['accuracy'])

# """**Training the Model**"""

# model.fit(trainx, trainy, epochs=4)

# # # Save the model
# model.save('my_new_model.h5')

"""CNN Model"""

model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28,28,1)))
model.add(tf.keras.layers.MaxPooling2D((2, 2)))
model.add(tf.keras.layers.Conv2D(64, (3, 3), activation='relu'))
model.add(tf.keras.layers.MaxPooling2D((2, 2)))
model.add(tf.keras.layers.Conv2D(64, (3, 3), activation='relu'))
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(128, activation='relu'))
model.add(tf.keras.layers.Dense(36))

model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

trainx = trainx.reshape(trainx.shape[0], 28, 28)
testx = testx.reshape(testx.shape[0], 28, 28)

#Training the Model
model.fit(trainx, trainy, epochs=5)

#Saving the model
model.save('my_CNN_model4.h5')

# model = tf.keras.models.load_model('my_CNN_model2.h5')

"""**Evaluating the Model**"""

test_loss, test_acc = model.evaluate(testx,  testy, verbose=2)

print('Test accuracy:', test_acc)

"""**Making Predictions**"""

predictions = model.predict(testx)
# trainx = trainx.reshape(trainx.shape[0], 784)

for i in range(rows):
    f = plt.figure()
    plt.imshow(testx[i], cmap=plt.cm.binary)

    print('\nActual Value : ' + labels[int(testy1[i])])
    print('Prediction   : ' + labels[np.argmax(predictions[i])])

    f.set_figwidth(3)
    f.set_figheight(3)
    plt.colorbar()
    plt.grid(False)
    plt.show()