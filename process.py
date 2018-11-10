import scipy.io
import numpy as np
import pickle


def load_data():
    data = []
    for i in range(1, 10):
        data.append(scipy.io.loadmat('data-science-P' + str(i) + '.mat'))
    return data


datas = load_data()


# People with pictures
P_n_images = []
for data in datas:
    col_to_labels = data['meta']['colToCoord'][0][0]
    images = []
    print('process first person')
    for i in range(360):
        image = np.zeros((51, 61, 23))
        for j in range(len(col_to_labels)):
            vox_coor = col_to_labels[j]
            image[vox_coor[0], vox_coor[1], vox_coor[2]] = data['data'][i][0][0][j]
        images.append(image)
    P_n_images.append(images)

pickle.dump(P_n_images, open('P_n_images.p', 'wb'))

# Only the pictures
datas = pickle.load(open('P_n_images.p', 'rb'))
images = []
for data in datas:
    for image in data:
        images.append(image)

pickle.dump(images, open('images.p', 'wb'))


