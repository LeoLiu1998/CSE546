from sklearn.linear_model import LogisticRegression
from sklearn import svm
from load import loadData
from clean import LabelInfo
import numpy as np


images, labels = loadData()

# If sum across one dimension
#for i in range(len(images)):
#    images[i] = np.sum(images[i], axis=2)

for i in range(len(images)):
    images[i] = images[i].flatten()

X_train = images[:len(images) // 10 * 9]
X_valid = images[len(images) // 10 * 9:]
y_train = labels[:len(images) // 10 * 9]
y_valid = labels[len(images) // 10 * 9:]



clf1 = LogisticRegression(random_state=0, solver='lbfgs', multi_class='multinomial').fit(X_train, y_train)

print('logistic regression score is ', clf1.score(X_valid, y_valid))

clf2 = svm.SVC(kernel='linear').fit(X_train, y_train)

print('SVM score is ', clf2.score(X_valid, y_valid))

