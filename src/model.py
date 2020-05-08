#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''=================================================
@Project -> File: text-generate -> model
@IDE    : PyCharm
@Author : fengchengli
@Date   : 2020/4/21 11:53
=================================================='''
import tensorflow as tf
from tensorflow import layers
from tensorflow.contrib.rnn import LSTMCell, GRUCell, DropoutWrapper, ResidualWrapper
from tensorflow.contrib.rnn import MultiRNNCell, LSTMStateTuple
from tensorflow.contrib.seq2seq import BahdanauAttention, AttentionWrapper, TrainingHelper, BasicDecoder, BeamSearchDecoder
from tensorflow.python.ops import array_ops
from src.data_loader import DataUnit
from tensorflow.python import debug as tf_debug

class Model:
    def __init__(self, args, encoder_vocab_size, decoder_vocab_size, mode):
        self.encoder_vocab_size = encoder_vocab_size
        self.decoder_vocab_size = decoder_vocab_size
        self.mode = mode

        self.beam_width = args.beam_width
        self.batch_size = args.batch_size
        self.hidden_size = args.hidden_size
        self.embedding_size = args.embedding_size
        self.layer_size = args.layer_size
        self.learning_rate = args.learning_rate
        self.cell_type = args.cell_type
        self.bidirection = args.bidirection
        self.share_embedding = args.share_embedding
        self.max_decode_step = args.max_decode_step
        self.dacay_step = args.dacay_step
        self.min_learining_rate = args.min_learning_rate
        self.max_gradient_norm = args.max_gradient_norm

        self.global_step = tf.Variable(0, trainable=False, name='global_step')
        self.build_model()

    def build_model(self):
        """
        构建完整的模型
        :return:
        """
        self.init_placeholder()
        self.embedding()
        self.encoder_outputs, self.encoder_state = self.build_encoder()
        self.build_decoder(self.encoder_outputs, self.encoder_state)
        if self.mode == 'train':
            self.build_optimizer()
        self.saver = tf.train.Saver()

    def init_placeholder(self):
        self.encoder_inputs = tf.placeholder(shape=[self.batch_size, None],
                                             dtype=tf.int32,
                                             name='encoder_inputs')
        self.encoder_inputs_length = tf.placeholder(shape=[self.batch_size],
                                                    dtype=tf.int32,
                                                    name='encoder_inputs_length')
        self.keep_prob = tf.placeholder(tf.float32, shape=(), name='keep_prob')
        if self.mode == 'train':
            self.decoder_inputs = tf.placeholder(shape=[self.batch_size, None],
                                                 dtype=tf.int32,
                                                 name='decoder_inputs')
            self.decoder_inputs_length = tf.placeholder(shape=[self.batch_size],
                                                        dtype=tf.int32,
                                                        name='decoder_inputs_length')
            self.decoder_inputs_train = tf.placeholder(shape=[self.batch_size, None],
                                                       dtype=tf.int32,
                                                       name='decoder_inputs_train')

    def embedding(self):
        """
        词嵌入操作
        :param share:编码器和解码器是否共用embedding
        :return:
        """
        with tf.variable_scope('embedding'):
            encoder_embedding = tf.Variable(
                tf.truncated_normal(shape=[self.encoder_vocab_size, self.embedding_size], stddev=0.1),
                name='encoder_embeddings')
            if not self.share_embedding:
                decoder_embedding = tf.Variable(
                    tf.truncated_normal(shape=[self.decoder_vocab_size, self.embedding_size], stddev=0.1),
                    name='decoder_embeddings')
                self.encoder_embeddings = encoder_embedding
                self.decoder_embeddings = decoder_embedding
            else:
                self.encoder_embeddings = encoder_embedding
                self.decoder_embeddings = encoder_embedding

    def one_cell(self, hidden_size, cell_type):
        if cell_type == 'gru':
            c = GRUCell
        else:
            c = LSTMCell
        cell = c(hidden_size)
        cell = DropoutWrapper(cell,
                              dtype=tf.float32,
                              output_keep_prob=self.keep_prob)
        cell = ResidualWrapper(cell)
        return cell

    def build_encoder_cell(self, hidden_size, cell_type, layer_size):
        """
        构建编码器所有层
        :param hidden_size:
        :param cell_type:
        :param layer_size:
        :return:
        """
        cells = [self.one_cell(hidden_size, cell_type) for _ in range(layer_size)]
        return MultiRNNCell(cells)

    def build_encoder(self):
        """
        构建完整编码器
        :return:
        """
        with tf.variable_scope('encoder'):
            encoder_cell = self.build_encoder_cell(self.hidden_size, self.cell_type, self.layer_size)
            encoder_inputs_embedded = tf.nn.embedding_lookup(self.encoder_embeddings, self.encoder_inputs)
            encoder_inputs_embedded = layers.dense(encoder_inputs_embedded,
                                                   self.hidden_size,
                                                   use_bias=False,
                                                   name='encoder_residual_projection')
            initial_state = encoder_cell.zero_state(self.batch_size, dtype=tf.float32)

            if self.bidirection:
                encoder_cell_bw = self.build_encoder_cell(self.hidden_size, self.cell_type, self.layer_size)
                (
                    (encoder_fw_outputs, encoder_bw_outputs),
                    (encoder_fw_state, encoder_bw_state)
                ) = tf.nn.bidirectional_dynamic_rnn(
                    cell_bw=encoder_cell_bw,
                    cell_fw=encoder_cell,
                    inputs=encoder_inputs_embedded,
                    sequence_length=self.encoder_inputs_length,
                    dtype=tf.float32,
                    swap_memory=True)

                encoder_outputs = tf.concat(
                    (encoder_bw_outputs, encoder_fw_outputs), 2)
                encoder_final_state = []

                for i in range(self.layer_size):
                    c_fw, h_fw = encoder_fw_state[i]
                    c_bw, h_bw = encoder_bw_state[i]
                    c = tf.concat((c_fw, c_bw), axis=-1)
                    h = tf.concat((h_fw, h_bw), axis=-1)
                    encoder_final_state.append(LSTMStateTuple(c=c, h=h))
                encoder_final_state = tuple(encoder_final_state)
            else:
                encoder_outputs, encoder_final_state = tf.nn.dynamic_rnn(
                    cell=encoder_cell,
                    inputs=encoder_inputs_embedded,
                    sequence_length=self.encoder_inputs_length,
                    dtype=tf.float32,
                    initial_state=initial_state,
                    swap_memory=True)

            return encoder_outputs, encoder_final_state

    def build_decoder_cell(self, encoder_outputs, encoder_final_state, hidden_size, cell_type, layer_size):
        """
        构建解码器所有层
        :param encoder_outputs:
        :param encoder_final_state:
        :param hidden_size:
        :param cell_type:
        :param layer_size:
        :return:
        """
        sequence_length = self.encoder_inputs_length
        if self.mode == 'decode':
            encoder_outputs = tf.contrib.seq2seq.tile_batch(
                encoder_outputs, multiplier=self.beam_width)
            encoder_final_state = tf.contrib.seq2seq.tile_batch(
                encoder_final_state, multiplier=self.beam_width)
            sequence_length = tf.contrib.seq2seq.tile_batch(
                sequence_length, multiplier=self.beam_width)

        if self.bidirection:
            cell = MultiRNNCell([self.one_cell(hidden_size * 2, cell_type) for _ in range(layer_size)])
        else:
            cell = MultiRNNCell([self.one_cell(hidden_size, cell_type) for _ in range(layer_size)])

        # 使用attention机制
        self.attention_mechanism = BahdanauAttention(num_units=self.hidden_size,
                                                     memory=encoder_outputs,
                                                     memory_sequence_length=sequence_length)

        def cell_input_fn(inputs, attention):
            mul = 2 if self.bidirection else 1
            attn_projection = layers.Dense(self.hidden_size * mul,
                                           dtype=tf.float32,
                                           use_bias=False,
                                           name='attention_cell_input_fn')
            return attn_projection(array_ops.concat([inputs, attention], -1))
        # tf.contrib.
        cell = AttentionWrapper(
            cell=cell,
            attention_mechanism=self.attention_mechanism,
            attention_layer_size=self.hidden_size,
            cell_input_fn=cell_input_fn,
            initial_cell_state=encoder_final_state,
            name='Attention_Wrapper',
        )
        if self.mode == 'decode':
            decoder_initial_state = cell.zero_state(batch_size=self.batch_size * self.beam_width,
                                                    dtype=tf.float32).clone(
                cell_state=encoder_final_state)
        else:
            decoder_initial_state = cell.zero_state(batch_size=self.batch_size,
                                                    dtype=tf.float32).clone(
                cell_state=encoder_final_state)
        return cell, decoder_initial_state

    def build_decoder(self, encoder_outputs, encoder_final_state):
        """
        构建完整解码器
        :return:
        """
        with tf.variable_scope("decode"):
            self.decoder_cell, self.decoder_initial_state = self.build_decoder_cell(encoder_outputs, encoder_final_state,
                                                                          self.hidden_size, self.cell_type,
                                                                          self.layer_size)

            # 输出层投影
            decoder_output_projection = layers.Dense(self.decoder_vocab_size, dtype=tf.float32, use_bias=False,
                                                     kernel_initializer=tf.truncated_normal_initializer(mean=0.0,
                                                                                                        stddev=0.1),
                                                     name='decoder_output_projection')
            tf.print(self.decoder_cell)
            tf.print(self.decoder_initial_state)

            if self.mode == 'train':
                # 训练模式
                decoder_inputs_embdedded = tf.nn.embedding_lookup(self.decoder_embeddings, self.decoder_inputs_train)
                training_helper = TrainingHelper(
                    inputs=decoder_inputs_embdedded,
                    sequence_length=self.decoder_inputs_length,
                    name='training_helper'
                )
                training_decoder = BasicDecoder(self.decoder_cell, training_helper,
                                                self.decoder_initial_state, decoder_output_projection)

                max_decoder_length = tf.reduce_max(self.decoder_inputs_length)
                training_decoder_output, _, _ = tf.contrib.seq2seq.dynamic_decode(training_decoder,
                                                                                  maximum_iterations=max_decoder_length)
                self.masks = tf.sequence_mask(self.decoder_inputs_length, maxlen=max_decoder_length, dtype=tf.float32, name='masks')
                self.loss = tf.contrib.seq2seq.sequence_loss(logits=training_decoder_output.rnn_output,
                                                             targets=self.decoder_inputs,
                                                             weights=self.masks,
                                                             average_across_timesteps=True,
                                                             average_across_batch=True
                                                             )
            else:
                # 预测模式
                self.start_token = [DataUnit.START_INDEX] * self.batch_size
                self.end_token = DataUnit.END_INDEX
                self.inference_decoder = BeamSearchDecoder(
                    cell=self.decoder_cell,
                    embedding=lambda x: tf.nn.embedding_lookup(self.decoder_embeddings, x),
                    start_tokens=self.start_token,
                    end_token=self.end_token,
                    initial_state=self.decoder_initial_state,
                    beam_width=self.beam_width,
                    output_layer=decoder_output_projection)

                self.inference_decoder_output, _, _ = tf.contrib.seq2seq.dynamic_decode(self.inference_decoder,
                                                                                        maxinum_iterations=self.max_decode_step)
                self.predicted_ids = self.inference_decoder_output.predicted_ids
                self.decoder_pred_decode = tf.transpose(self.predicted_ids, perm=[0, 2, 1])

    def check_feeds(self, encoder_inputs, encoder_inputs_length, decoder_inputs, decoder_inputs_train, decoder_inputs_length, keep_prob, decode):
        """
        检查输入，返回输入字典
        :param encoder_inputs:
        :param encoder_inputs_length:
        :param decoder_inputsx:
        :param decoder_inputs_length:
        :param keep_prob:
        :param decode:
        :return:
        """
        input_batch_size = encoder_inputs.shape[0]
        assert input_batch_size == encoder_inputs_length.shape[0]
        if not decode:
            target_batch_size = decoder_inputs.shape[0]
            assert target_batch_size == decoder_inputs_length.shape[0]
        input_feed = {self.encoder_inputs.name: encoder_inputs,
                      self.encoder_inputs_length.name: encoder_inputs_length,
                      self.keep_prob.name: keep_prob}
        if not decode:
            input_feed[self.decoder_inputs.name] = decoder_inputs
            input_feed[self.decoder_inputs_train.name] = decoder_inputs_train
            input_feed[self.decoder_inputs_length.name] = decoder_inputs_length
        return input_feed

    def build_optimizer(self):
        """
        构建优化器
        :return:
        """
        learning_rate = tf.train.polynomial_decay(self.learning_rate, self.global_step,
                                                  self.dacay_step, self.min_learining_rate, power=0.5)
        self.current_learning = learning_rate
        trainable_params = tf.trainable_variables()
        gradients = tf.gradients(self.loss, trainable_params)
        # 优化器
        self.opt = tf.train.AdamOptimizer(learning_rate=learning_rate)
        # 梯度裁剪
        clip_gradients, _ = tf.clip_by_global_norm(gradients, self.max_gradient_norm)
        # 更新梯度
        self.update = self.opt.apply_gradients(zip(clip_gradients, trainable_params), global_step=self.global_step)

    def train(self, sess, encoder_inputs, encoder_inputs_length, decoder_inputs_in, decoder_inputs_target, decoder_inputs_length, keep_prob):
        """
        训练
        :param sess:
        :param encoder_inputs:
        :param encoder_inputs_length:
        :param decoder_inputs_in:
        :param decoder_inputs_target:
        :param decoder_inputs_length:
        :param keep_prob:
        :return:
        """
        input_feed = self.check_feeds(encoder_inputs, encoder_inputs_length,
                                      decoder_inputs_in, decoder_inputs_target, decoder_inputs_length,
                                      keep_prob, False)
        output_feed = [self.update, self.loss, self.current_learning]
        _, cost, lr = sess.run(output_feed, input_feed)
        return cost, lr

    def predict(self, sess, encoder_inputs, encoder_inputs_length):
        """
        预测
        :param sess:
        :param encoder_inputs:
        :param encoder_inputs_length:
        :return:
        """
        input_feed = self.check_feeds(encoder_inputs, encoder_inputs_length,
                                      None, None, None, 1, True)
        print(input_feed)

        # sess = tf_debug.LocalCLIDebugWrapperSession(sess)
        # output = [self.inference_decoder, self.inference_decoder_output, self.predicted_ids, self.decoder_pred_decode]
        output = [self.inference_decoder_output]
        # start_token, end_token, inference_deocer, inference_decoder_output, predicted_ids, pred = sess.run(output, input_feed)
        inference_decoder_output = sess.run(output, input_feed)
        # start_token = sess.run(output, input_feed)
        print('----1----')
        print(inference_decoder_output)
        # print(encoder_outputs)  # [1, 10, 64]
        # print('----2----')
        # print(encoder_state)    # [1, 64]
        # print('----3----')
        # print(decoder_initial_state)    # cell_state=[5, 64] attention=[5, 64]
        # print(end_token)
        # print(inference_deocer)
        # return pred[0]

    def save(self, sess, save_path='model/chatmsg_model.ckpt'):
        """
        保存模型
        :param sess:
        :param save_path:
        :return:
        """
        self.saver.save(sess, save_path)

    def load(self, sess, save_path='model/chatmsg_model.ckpt'):
        """
        加载模型
        :param sess:
        :param save_path:
        :return:
        """
        return self.saver.restore(sess, save_path)
