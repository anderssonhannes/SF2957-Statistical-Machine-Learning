# Display images from the digits data set

import matplotlib.pyplot as plt
from sklearn import datasets

# import some data to play with
digits = datasets.load_digits()

im = digits.images[1]
print(im.shape)
print(digits.images.shape)
plt.gray() 
plt.matshow(digits.images[12]) 
plt.matshow(digits.images[13]) 
plt.show() 
