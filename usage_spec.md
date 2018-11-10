# Usage spec

## LabelInfo()

### Methods

Method's are hidden, see code for detailed spec

### Variables (also contained in code)

```python
index2word -> dict[tuple:str] # {(class_num, word_num): word}
word2index -> dict[str:tuple] # {word:(class_num, word_num)}

class2index -> dict # {class_name: class_num}
index2class -> dict # {class_num: class_name}

num_class -> int # number of categories(0-11)
num_word -> int # number of words

```


```Python
info -> dict 
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
"""
```

## images.p
A pickled image dataset stored in Google drive (https://drive.google.com/drive/u/0/folders/1rF2EbrzOjRjG6e0BX563fJymsn44Ad7j)
```Python
Retrive: images = pickle.load(open('images.p', 'rb')) # A list contains 3240 trials, each trial is an image in numpy (51 * 61 * 23). Trials are in order.
```


**Note:**  ``` "cond_number": class_num ```, this value is the original value minus 2
