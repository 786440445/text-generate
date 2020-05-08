#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''=================================================
@Project -> File: text-generate -> predict
@IDE    : PyCharm
@Author : fengchengli
@Date   : 2020/4/25 17:22
=================================================='''
import os
import argparse
import random
import tensorflow as tf

from src.model import Model
from src.data_loader import DataUnit
from config import BASE_MODEL_DIR, MODEL_NAME, add_arguments, data_config


def predict():
    """
    训练模型
    :return:
    """
    # 解析参数
    parser = argparse.ArgumentParser()
    add_arguments(parser)
    args = parser.parse_args()
    args.batch_size = 1

    # 数据集获取
    du = DataUnit(**data_config)
    # 模型路径
    save_path = os.path.join(BASE_MODEL_DIR, MODEL_NAME)

    # 创建session的时候设置显存根据需要动态申请
    tf.reset_default_graph()
    config = tf.ConfigProto()


    with tf.Graph().as_default():
        with tf.Session(config=config) as sess:
            model = Model(args,
                          encoder_vocab_size=du.vocab_size,
                          decoder_vocab_size=du.vocab_size,
                          mode='decode')
            model.load(sess, save_path)
            encoder_inputs = random.choices(type_list, k=args.batch_size)
            print(encoder_inputs)
            x, xl = du.generate_predit_inputs(encoder_inputs)
            preds = model.predict(sess, x, xl)
            for pred in preds:
                out = du.transform_indexs(pred)
                print(out)


if __name__ == '__main__':
    type_list = ['yxlm', 'jdqs', 'hpjy', 'yz', 'ms', 'wd', 'wzry']
    predict()
