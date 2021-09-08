import re
import math
import codecs

from .utils import is_not_chinese
from .utils import ngram_segment
from .utils import get_bigram
from .utils import NewWordCandidateInfo


class EntropyPMINewWordDiscovery(object):
    """
    基于左右熵和PMI的新词发现算法

    :param doc: (str) 待挖掘的文档
    :param max_word_len: (int) N-gram的最大长度
    :param min_tf: (float) 词频／文档的长度 的阈值 
    :param min_freq: (float) 词频的阈值
    :param min_entropy: (float) 左右熵的阈值 
    :param min_pmi: (float) 互信息的阈值 

    Reference: 
        [1] https://github.com/DenseAI/kaitian-xinci
        [2] http://www.matrix67.com/blog/archives/5044
        
    To Do:
        [1] 使用字典树优化代码 https://github.com/zhanzecheng/Chinese_segment_augment
        [2] kenlm统计ngram词 https://github.com/kpu/kenlm/ + https://spaces.ac.cn/archives/6920
    """

    def __init__(
        self, 
        max_word_len=5, 
        min_tf=0.000005, 
        min_freq=5, 
        min_entropy=0.05, 
        min_pmi=3.0
    ):

        super(EntropyPMINewWordDiscovery, self).__init__()

        self.max_word_len = max_word_len
        self.min_tf = min_tf
        self.min_freq = min_freq
        self.min_entropy = min_entropy
        self.min_pmi = min_pmi

    @staticmethod 
    def extract_candidate_word(_doc, _max_word_len):
        """提取候选词

        :param _doc: (str) 待挖掘的文档
        :param _max_word_len: (int) N-gram的最大长度
        """
        candidates = []
        doc_length = len(_doc)
        for ii in range(doc_length):
            for jj in range(ii + 1, min(ii + 1 + _max_word_len, doc_length + 1)):

                # 判断是否是中文字符，非中文字符，不再组合
                if is_not_chinese(_doc[ii:jj]) is True or is_not_chinese(_doc[jj - 1:jj]) is True:
                    break

                word = _doc[ii:jj]

                previous_word = ' '
                if ii - 1 >= 0:
                    previous_word = _doc[ii - 1:ii]
                next_word = ' '
                if jj + 1 < doc_length + 1:
                    next_word = _doc[jj:jj + 1]

                # TODO: 存在性能问题，大型文档例如Wikipedia dumps等，几个G的文档可能内存顶不住
                candidates.append([previous_word, word, next_word])
                
        return candidates

    def get_ngram_word(self, _doc):
        """获取N-gram信息

        :param _doc(str): 待挖掘的文档
        """
        # 过滤掉非中文字符
        pattern = re.compile(u'[\\s\\d,.<>/?:;\'\"[\\]{}()\\|~!@#$%^&*\\-_=+，。《》、？：；“”‘’｛｝【】（）…￥！—┄－]+')
        _doc = pattern.sub(r' ', _doc)

        word_index = self.extract_candidate_word(_doc, self.max_word_len)
        word_cad = {}  # 候选词的字典
        for suffix in word_index:
            word = suffix[1]
            previous_word = suffix[0]
            next_word = suffix[2]
            if word not in word_cad:
                word_cad[word] = NewWordCandidateInfo(word)

            # 记录候选词的左右邻居
            word_cad[word].update_data(previous_word, next_word)
        length = len(_doc)

        # 计算候选词的频率、以及左右熵
        for word in word_cad:
            word_cad[word].compute_indexes(length)

        # 按词的长度排序
        values = sorted(word_cad.values(), key=lambda x: len(x.text))
        for v in values:
            if len(v.text) == 1:
                continue
            v.compute_pmi(word_cad)
        return sorted(values, key=lambda v: len(v.text), reverse=False)
    
    def extract_phrase(
        self,
        doc,
        return_phrase_scores=False,
        return_doc_avg_frequency=False,
        return_doc_avg_entropy=False,
        return_doc_avg_pmi=False
    ):
        word_info = self.get_ngram_word(doc)
        word_info_length = float(len(word_info))
        
        phrases = []
        for _word in word_info:
            if (len(_word.text) > 1 and _word.pmi > self.min_pmi and _word.freq > self.min_tf 
                and min(_word.left, _word.right) > self.min_entropy and _word.raw_freq > self.min_freq):
                                
                if return_phrase_scores:
                    phrases.append({
                        'phrase': _word.text,
                        'phrase_length': len(_word.text),
                        'phrase_frequency': _word.freq,
                        'phrase_pmi': _word.pmi,
                        'phrase_left_entropy': _word.left,
                        'phrase_right_entropy': _word.right
                    })
                else:
                    phrases.append(_word.text)
        
        result = {}
        result['phrase'] = phrases
        
        if return_doc_avg_frequency:
            result['avg_frequency'] = sum(map(lambda w: w.freq, word_info)) / word_info_length
            
        if return_doc_avg_entropy:
            result['avg_entropy'] = sum(map(lambda w: min(w.left, w.right), word_info)) / word_info_length
            
        if return_doc_avg_pmi:
            result['avg_pm'] = sum(map(lambda w: w.pmi, word_info)) / word_info_length
                    
        if len(result) < 2:
            return result['phrase']
        else:
            return  result
