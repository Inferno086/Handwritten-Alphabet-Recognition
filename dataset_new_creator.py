import numpy as np
import csv
import LabelHandler

with open('dataset_final.csv', 'r') as f:
    reader = csv.reader(f)
    data = list(reader)

print('Dataset Loaded Successfully!')

"""PREPROCESSING THE DATASET"""


#Definitions
labels = ['0','1','2','3','4','5','6','7','8','9',
          'A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y','Z',
          'a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p','q','r','s','t','u','v','w','x','y','z'
         ]


rows = len(data)
columns = len(data[0])

#Seperating Input and Label
datax = [row[0:columns-1] for row in data]
datay = [row[columns-1:] for row in data]

for i in range(len(datay)):
    datay[i] = datay[i][0]

datax = np.array(datax, dtype=int)
datay = LabelHandler.labelHandler(datay)
datay = np.array(datay, dtype=int)

k = 0
for i in datay:
    if i > 9:
        datax[k] = np.rot90(datax[k].reshape(28,28), 3).flatten()
        datax[k] = np.fliplr(datax[k].reshape(28,28)).flatten()
    k = k + 1

print('Dataset updated successfully!')
print(datax.shape)
print(datay.shape)

np.savetxt("datax.csv", datax, delimiter=",")
np.savetxt("datay.csv", datay, delimiter=",")