#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''=================================================
@Project -> File: text-generate -> train.py
@IDE    : PyCharm
@Author : fengchengli
@Date   : 2020/4/21 11:52
=================================================='''
import os
import argparse
import numpy as np
import tensorflow as tf
from tqdm import tqdm

from src.model import Model
from src.data_loader import DataUnit
from config import BASE_MODEL_DIR, MODEL_NAME, add_arguments, data_config

continue_train = True


def train():
    """
    训练模型
    :return:
    """
    # 解析参数
    parser = argparse.ArgumentParser()
    add_arguments(parser)
    args = parser.parse_args()

    # 数据集获取
    du = DataUnit(**data_config)
    steps = int(len(du) / args.batch_size) + 1

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
                          mode='train')
            init = tf.global_variables_initializer()
            sess.run(init)
            if continue_train and os.path.exists(save_path):
                model.load(sess, save_path)
            for epoch in range(1, args.epochs + 1):
                costs = []
                bar = tqdm(range(steps), total=steps, desc='epoch {}, loss=0.000000'.format(epoch))
                for step in bar:
                    x, xl, y_in, y_target, yl = du.next_batch(args.batch_size)
                    max_len = np.max(yl)
                    y_in = y_in[:, 0:max_len]
                    y_target = y_target[:, 0:max_len]
                    cost, lr = model.train(sess, x, xl, y_in, y_target, yl, args.keep_prob)
                    costs.append(cost)
                    bar.set_description('epoch {}, step {}, loss={:.6f} lr={:.6f}'.format(epoch, (step+1), np.mean(costs), lr))
                        # print('-----save : {}----'.format((step+1)))
                model.save(sess, save_path=save_path)


if __name__ == '__main__':
    train()
