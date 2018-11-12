import glob
import pickle
import numpy as np
from scipy.io import loadmat


class LabelInfo:
    FILES = ['./data/data-science-P' + str(i) + '.mat' for i in range(1, 10)]

    """
    info[i] i=0,1,2...8 represent a subject's info
    each info[i] is a dictionary having key (0-359)
    representing each trial

    For each trial, a dictionary is stored
    as
    {"num_trial": j, (0-359)
     "condition": class_name,
     "cond_number": class_num,
     "word": word,
     "word_number": word_num,
     "word_index": (class_num, word_num)
     "epoch": epoch}
     
    
    index2word -> dict[tuple:str] # {(class_num, word_num): word}
    word2index -> dict[str:tuple] # {word:(class_num, word_num)}

    class2index -> dict # {class_name: class_num}
    index2class -> dict # {class_num: class_name}

    num_class -> int # number of categories(0-11)
    num_word -> int # number of words 
    """

    def __init__(self):
        self.word2index = {}
        self.index2word = {}

        self.class2index = {}
        self.index2class = {}

        self.num_word = 0
        self.num_class = 0
        self.info = {}

        self.__process()
        self.subject_first = np.array(self.__subject_first())  # np.array
        self.trial_first = np.swapaxes(self.subject_first, 0, 1)  # np.array
        self.sets_by_label = self.__by_label()  # list

    def __process(self):

        for j in range(len(LabelInfo.FILES)):
            filename = LabelInfo.FILES[j]
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
                tmp = (class_num, word_num)

                new_info = {"num_trial": i,
                            "condition": class_name,
                            "cond_number": class_num,
                            "word": word,
                            "word_number": word_num,
                            "word_index": tmp,
                            "epoch": epoch}

                if tmp not in self.index2word:
                    self.word2index[word] = tmp
                    self.index2word[tmp] = word
                    self.num_word += 1

                if class_name not in self.class2index:
                    self.class2index[class_name] = class_num
                    self.index2class[class_num] = class_name
                    self.num_class += 1

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

    def __subject_first(self) -> list:
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
                lst.append([self.info[sub][trial]["cond_number"]])
            result.append(lst)
        return result

    def __by_label(self) -> list:
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
    a = LabelInfo()
    pickle.dump(a, open("LabelInfo().p", "wb"))
