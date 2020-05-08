#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''=================================================
@Project -> File: text-generate -> data_generator
@IDE    : PyCharm
@Author : fengchengli
@Date   : 2020/4/21 14:48
=================================================='''
import os
import re
import random
import pickle
import itertools
import numpy as np
from collections import Counter
home_dir = os.path.dirname(__file__)


class DataUnit:
    # 特殊标签
    PAD = '<PAD>'
    UNK = '<UNK>'
    START = '<SOS>'
    END = '<EOS>'

    # 特殊标签索引
    PAD_INDEX = 0
    UNK_INDEX = 1
    START_INDEX = 2
    END_INDEX = 3

    def __init__(self, corpus_path, processed_path, word2index_path, max_key_len, max_chatmsg_len):
        self.corpus_path = corpus_path
        self.processed_path = processed_path
        self.word2index_path = word2index_path
        self.max_key_len = max_key_len
        self.max_chatmsg_len = max_chatmsg_len
        self.cateid2name = {1: 'yxlm', 3: 'dota2', 174: 'ecy', 181: 'wzry', 194: 'ms',
                               201: 'yz', 270: 'jdqs', 350: 'hpjy', 1008: 'wd'}
        self.data = self.load_data()
        self._fit_data_()

    def next_batch(self, batch_size):
        """
        生成一批训练数据
        """
        data_batch = random.sample(self.data, batch_size)
        batch = []
        for data in data_batch:
            encoded_key = self.transform_sentence(data[0])
            encoded_chatmsg_in = [self.START_INDEX] + self.transform_sentence(data[1])
            encoded_chatmsg_target = self.transform_sentence(data[1]) + [self.END_INDEX]
            key_len = len(encoded_key)
            chatmsg_len = len(encoded_chatmsg_in)
            # 填充句子
            encoded_key = encoded_key + [self.func_word2index(self.PAD)] * (self.max_key_len - key_len)
            encoded_chatmsg_in = encoded_chatmsg_in + [self.func_word2index(self.PAD)] * (self.max_chatmsg_len - chatmsg_len)
            encoded_chatmsg_target = encoded_chatmsg_target + [self.func_word2index(self.PAD)] * (self.max_chatmsg_len - chatmsg_len)
            batch.append((encoded_key, key_len, encoded_chatmsg_in, encoded_chatmsg_target, chatmsg_len))
        batch = zip(*batch)
        batch = [np.asarray(x) for x in batch]
        return batch

    def generate_predit_inputs(self, predict_inputs):
        """
        构建预测输入序列
        :param predict_inputs:
        :return:
        """
        inputs = []
        key_lens = []
        for predict_input in predict_inputs:
            encoded_key = self.transform_sentence(predict_input)
            key_len = len(encoded_key)
            key_lens.append(key_len)
            encoded_padding = encoded_key + [self.func_word2index(self.PAD)] * (self.max_key_len - key_len)
            inputs.append(encoded_padding)
        return np.array(inputs), np.array(key_lens)

    def transform_sentence(self, sentence):
        """
        将句子转化为索引序列
        :param sentence:
        :return:
        """
        return [self.func_word2index(word) for word in sentence]

    def transform_indexs(self, indexs):
        """
        将句子转化为索引序列
        :param indexs:
        :return:
        """
        res = []
        for index in indexs:
            if index == self.START_INDEX or index == self.PAD_INDEX or \
                    index == self.END_INDEX or index == self.UNK_INDEX:
                continue
            res.append(self.func_index2word(index))
        return ''.join(res)

    def func_word2index(self, word):
        return self.word2index.get(word, self.word2index[self.UNK])

    def func_index2word(self, index):
        return self.index2word.get(index, self.UNK)

    def _fit_data_(self):
        if not os.path.exists(self.word2index_path):
            vocabularies = [x[0] + x[1] for x in self.data]
            self._fit_word_(itertools.chain(*vocabularies))
            with open(self.word2index_path, 'wb') as fw:
                pickle.dump(self.word2index, fw)
        else:
            with open(self.word2index_path, 'rb') as fr:
                self.word2index = pickle.load(fr)
            self.index2word = dict([(v, k) for k, v in self.word2index.items()])
        self.vocab_size = len(self.word2index)

    def _fit_word_(self, vocabularies):
        """
        将词表中所有的词转化为索引，过滤吊出现次数少于4次的词
        :param vocabularies:
        :return:
        """
        vocab_counter = Counter(vocabularies)
        index2word = [self.PAD] + [self.UNK] + [self.START] + [self.END] + \
                     [x[0] for x in vocab_counter if vocab_counter.get(x[0]) > 4]
        self.name2cateid = dict([w, i] for i, w in enumerate(self.cateid2name))
        self.word2index = dict([w, i] for i, w in enumerate(index2word))
        self.index2word = dict([i, w] for i, w in enumerate(index2word))

    def load_data(self):
        """
        获取处理后的语料库
        :return:
        """
        if not os.path.exists(self.processed_path):
            data = self._extract_data()
            with open(self.processed_path, 'wb') as fw:
                pickle.dump(data, fw)
        else:
            with open(self.processed_path, 'rb') as fr:
                data = pickle.load(fr)
        # 根据CONFIG文件中配置的最大值和最小值问答对长度来进行数据过滤
        return data

    def _regular_(self, sen):
        """
        句子规范化
        :param sen:
        :return:
        """
        sen = sen.replace('/', '')
        sen = re.sub(r'…{1,100}', '…', sen)
        sen = re.sub(r'\.{3,100}', '…', sen)
        sen = re.sub(r'···{2,100}', '…', sen)
        sen = re.sub(r',{1,100}', '，', sen)
        sen = re.sub(r'\.{1,100}', '。', sen)
        sen = re.sub(r'。{1,100}', '。', sen)
        sen = re.sub(r'\?{1,100}', '？', sen)
        sen = re.sub(r'？{1,100}', '？', sen)
        sen = re.sub(r'!{1,100}', '！', sen)
        sen = re.sub(r'！{1,100}', '！', sen)
        sen = re.sub(r'~{1,100}', '', sen)
        sen = re.sub(r'～{1,100}', '', sen)
        sen = re.sub(r'[“”]{1,100}', '"', sen)
        sen = re.sub('[^\w\u4e00-\u9fff"。，？！～·]+', '', sen)
        sen = re.sub(r'[ˇˊˋˍεπのゞェーω]', '', sen)
        return sen

    def _extract_data(self):
        res = []
        for path, _, files in os.walk(self.corpus_path):
            for file in files:
                label = file.split('_')[0]
                label = self.cateid2name[int(label)]
                with open(os.path.join(path, file), 'r', encoding='utf-8') as f:
                    for content in f.readlines():
                        content = self._regular_(content)
                        content = content.replace(' ', '')
                        content = content.strip()
                        # 去空格，实际上没用到
                        if content != '':
                            res.append((label, content))
        return res

    def __len__(self):
        """
        返回处理后语料库中的数量
        :return:
        """
        return len(self.data)
