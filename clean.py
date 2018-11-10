import glob
import pickle
import numpy as np
from scipy.io import loadmat


class label_info:
    FILES = glob.glob("./data/*.mat")

    def __init__(self):
        self.word2index = {}
        self.index2word = {}

        self.class2index = {}
        self.index2class = {}

        self.num_class = 0
        self.num_word = 0
        self.info = {}

        self.__process()
        self.subject_first = np.array(self.__subject_first())
        self.trial_first = np.swapaxes(self.subject_first, 0, 1)
        self.sets_by_label = self.__by_label()

    def __process(self):

        for j in range(len(label_info.FILES)):
            filename = label_info.FILES[j]
            subject = loadmat(filename)
            # v_index = 0
            subject_infos = subject["info"][0]
            self.info[j] = {}
            for i in range(len(subject_infos)):
                trial = subject_infos[i]
                class_name = trial[0][0]
                class_num = trial[1][0, 0] - 2
                word = trial[2][0]
                word_num = trial[3][0, 0]
                epoch = trial[4][0, 0]
                new_info = {"num_trial": i,
                            "condition": class_name,
                            "cond_number": class_num,
                            "word": word,
                            "word_number": word_num,
                            "epoch": epoch}

                if word not in self.word2index:
                    self.word2index[word] = self.num_word
                    self.index2word[self.num_word] = word
                    self.num_word += 1

                if class_name not in self.class2index:
                    self.class2index[class_name] = self.num_class
                    self.index2class[self.num_class] = class_name
                    self.num_class += 1

                # print(self.class2index)
                new_info["class_index"] = self.class2index[class_name]
                new_info["word_index"] = self.word2index[word]

                self.info[j][i] = new_info

        # assert that every trials done on each subject are the same
        # in the exact same order
        for trial in range(360):
            dic0 = self.info[0][trial]
            for sub in range(1, 9):
                dic_sub = self.info[sub][trial]
                for key, value in dic_sub.items():
                    assert key in dic0
                    assert dic0[key] == value

    def __subject_first(self):
        """
        return a numpy array (9, 360, 1)
        9 rows, indicating cadidates' indices
        360 columns, indicating each experiment's true label
        1 slots, indicating class label
        """
        result = []
        for sub in range(0, 9):
            lst = []
            for trial in range(360):
                lst.append([self.info[sub][trial]["class_index"]])
            result.append(lst)
        return result

    def __by_label(self):
        """
        return a list of numpy array.
        each element is labels for that class (0-11)
        """
        datasets = []
        for target in range(12):
            lst = []
            for sub in range(0, 9):
                for trial in range(360):
                    label = self.info[sub][trial]["cond_number"]
                    # print(label)
                    if label == target:
                        lst.append(label)
            datasets.append(np.array(lst))
        return datasets


if __name__ == "__main__":
    a = label_info()
    pickle.dump(a, open("label_info().p", "wb"))
