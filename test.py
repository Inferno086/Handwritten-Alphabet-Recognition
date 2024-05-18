import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

labels = ['0','1','2','3','4','5','6','7','8','9',
          'A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y','Z',
          'a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p','q','r','s','t','u','v','w','x','y','z'
         ]

datax = pd.read_csv('datax.csv')
datay = pd.read_csv('datay.csv')
datax = datax.to_numpy()
datay = datay.to_numpy().flatten()

print(datax.shape)
print(datay.shape)

shuffler = np.random.permutation(datax.shape[0])
datax = datax[shuffler]
datay = datay[shuffler]

for i in range(130000):
    if int(datay[i]) == 5:
        f = plt.figure()

        plt.imshow(datax[i].reshape(28,28), cmap=plt.cm.binary)
        print(labels[int(datay[i])])
        f.set_figwidth(3)
        f.set_figheight(3)
        plt.colorbar()
        plt.grid(False)
        plt.show()