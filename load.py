import pickle
import numpy as np
from clean import LabelInfo


def loadData():

    label_list = []
    label = pickle.load(open('Label.p', 'rb'))
    for subjuct in range(9):
        for trail in range(360):
            label_list.append(label.info[subjuct][trail]['cond_number'])

    image = pickle.load(open('images.p', 'rb'))
    image_list = [np.array(x) for x in image]

    return image_list, label_list
