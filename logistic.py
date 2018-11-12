from sklearn.linear_model import LogisticRegression
from load import loadData
from clean import LabelInfo


images, labels = loadData()
for i in range(len(images)):
    images[i] = images[i].flatten()

clf = LogisticRegression(random_state=0, solver='lbfgs', multi_class='multinomial').fit(images, labels)

print(clf.score(images, labels))
