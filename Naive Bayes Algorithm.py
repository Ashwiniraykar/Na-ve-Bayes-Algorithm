
# coding: utf-8

# In[1]:


from sklearn.datasets import load_digits
digits = load_digits()
import numpy as np
from matplotlib import pyplot as plt


# In[2]:


fig = plt.figure(figsize=(6, 6))
fig.subplots_adjust(left=0, right=1, bottom=0, top=1, hspace=0.05, wspace=0.05)
for i in range(64):
    ax = fig.add_subplot(8, 8, i + 1, xticks=[], yticks=[])
    ax.imshow(digits.images[i], cmap=plt.cm.binary, interpolation='nearest')
    ax.text(0, 7, str(digits.target[i]))


# In[3]:


from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split

# split the data into training and validation sets
X_train, X_test, y_train, y_test = train_test_split(digits.data, digits.target)

# train the model
C = GaussianNB()
C.fit(X_train, y_train)

# use the model to predict the labels of the test data
predicted = C.predict(X_test)
expected = y_test
fig = plt.figure(figsize=(6, 6))
fig.subplots_adjust(left=0, right=1, bottom=0, top=1, hspace=0.05, wspace=0.05)

# plot the digits: each image is 8x8 pixels
for i in range(64):
    plot = fig.add_subplot(8, 8, i + 1, xticks=[], yticks=[])
    plot.imshow(X_test.reshape(-1, 8, 8)[i], cmap=plt.cm.binary,interpolation='nearest')

    # label the image with the target value
    if predicted[i] == expected[i]:
        plot.text(0, 7, str(predicted[i]), color='green')
    else:
        plot.text(0, 7, str(predicted[i]), color='red')


# In[4]:


from sklearn import metrics
print(metrics.accuracy_score(expected,predicted))


# In[7]:


from sklearn.datasets import load_digits
digits = load_digits()
n_samples = len(digits.images)
X = digits.images.reshape((n_samples, 64))
Y = digits.target.reshape((n_samples,1))
from sklearn import metrics
from sklearn.naive_bayes import GaussianNB
model = GaussianNB()
model.fit(X, np.ravel(Y, order='C'))
predicted = model.predict(X)
print(metrics.accuracy_score(Y, predicted))

