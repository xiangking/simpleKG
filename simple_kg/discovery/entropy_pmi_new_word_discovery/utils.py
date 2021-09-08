import re
import math
import codecs
import random
import numpy as np


def is_not_chinese(uchar:str):
    """
    判断一个unicode是否是汉字

    :param uchar: (str) 待判断的字符
    """
    if uchar.isalpha() is True:
        return False
    elif uchar >= u'\u4e00' and uchar <= u'\u9fa5':
        return False
    else:
        return True


def ngram_segment(text, n=3):
    """
    使用n-gram分割句子
    
    :param text: (str) 待分割的文本
    :param n: (int) n-gram length, 默认值为3
    """

    text_length = len(text)
    skip = min(n, text_length)

    # 生成ngram
    for j in range(text_length):
        for k in range(j + 1, min(j + 1 + skip, text_length + 1)):
            yield text[j:k]


def get_bigram(text):
    """
    一个单词拆分为所有可能的两两组合。例如，ABB可以分为（a，bb），（ab，b）

    :param text: (str) 待分割的字符串
    """

    return [(text[0:i], text[i:]) for i in range(1, len(text))]


class NewWordCandidateInfo(object):
    """
    记录N-gram信息，包括左邻居，右邻居，频率，PMI

    :param text: N-gram单词

    Reference: 
        https://github.com/DenseAI/kaitian-xinci
    """

    def __init__(self, text):
        super(NewWordCandidateInfo, self).__init__()
        self.text = text
        self.freq = 0.0
        self.left = []         # record left neighbors
        self.left_dict = {}
        self.right = []        # record right neighbors
        self.right_dict = {}
        self.pmi = 0

        self.raw_freq = 0
        self.raw_length = 0

    def update_data(self, left, right):
        """添加出现在N-gram单词左右两边的字

        Args:
            left(str): 出现在N-gram单词左边的字
            right(str): 出现在N-gram单词右边的字
        """
        self.freq += 1.0
        if left:
            self.left.append(left)
        if right:
            self.right.append(right)

    def compute_indexes(self, length):
        """计算单词的频率和左/右熵

        Args:
            总
        """       
        self.raw_freq = self.freq
        self.raw_length = length
        self.freq /= length
        self.left, self.left_dict = NewWordCandidateInfo.compute_entropy(self.left)
        self.right, self.right_dict = NewWordCandidateInfo.compute_entropy(self.right)

    @staticmethod 
    def compute_entropy(_list):
        """计算左/右熵

        Args:
            _list(list): 出现在N-gram单词左／右的词列表

        Formula:
            [1] https://www.hankcs.com/nlp/extraction-and-identification-of-mutual-information-about-the-phrase-based-on-information-entropy.html
        """
        length = float(len(_list))
        frequence = {}
        if length == 0:
            return 0, frequence
        else:
            for i in _list:
                frequence[i] = frequence.get(i, 0) + 1
            return sum(map(lambda x: - x / length * math.log(x / length), frequence.values())), frequence

    def compute_pmi(self, words_dict):
        """计算互信息

        Args:
            words_dict(dict): {N-gram单词: NewWordCandidateInfo对象}

        Formula:
            [1] https://ww1.sinaimg.cn/large/6cbb8645gw1el41boc5q9j20jw02oaa3.jpg
        """        
        sub_part = get_bigram(self.text)
        if len(sub_part) > 0:
            self.pmi = min(
                map(lambda word: math.log(self.freq / words_dict[word[0]].freq / words_dict[word[1]].freq), sub_part))