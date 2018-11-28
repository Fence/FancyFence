import ipdb
import numpy as np
import tensorflow as tf
from tensorflow.python.util import nest


class Seq2SeqModel(object):
    """docstring for Seq2Seq"""
    def __init__(self, args, embedding):
        self.rnn_size = args.rnn_size
        self.num_layers = args.num_layers
        self.dropout_rate = args.dropout_rate
        self.learning_rate = args.learning_rate
        self.max_grad_norm = args.max_grad_norm

        self.word2id = args.word2id
        self.vocab_size = len(self.word2id)
        self.emb_size = args.emb_size
        self.mode = args.mode
        self.beam_search = args.beam_search
        self.beam_size = args.beam_size
        self.max_decode_len = args.max_decode_len
        self.build_model()
        self.embedding.assign(embedding)


    def _create_rnn_cell(self):
        def a_cell():
            single_cell = tf.contrib.rnn.LSTMCell(self.rnn_size)
            cell = tf.contrib.rnn.DropoutWrapper(single_cell, output_keep_prob=self.keep_prob)
            return cell

        rnn_cell = tf.contrib.rnn.MultiRNNCell([a_cell() for _ in range(self.num_layers)])
        return rnn_cell


    def build_model(self):
        #ipdb.set_trace()
        print('Building model ...')
        self.encoder_input = tf.placeholder(tf.int32, [None, None], name='encoder_input')
        self.encoder_length = tf.placeholder(tf.int32, [None], name='encoder_length')

        self.batch_size = tf.placeholder(tf.int32, [], name='batch_size')
        self.keep_prob = tf.placeholder(tf.float32, [], name='keep_prob')

        self.decoder_target = tf.placeholder(tf.int32, [None, None], name='decoder_target')
        self.decoder_length = tf .placeholder(tf.int32, [None], name='decoder_length')
        self.max_target_length = tf.reduce_max(self.decoder_length, name='max_target_length')
        self.mask = tf.sequence_mask(self.decoder_length, self.max_target_length, dtype=tf.float32, name='mask')

        print('Constructing encoder ...')
        with tf.variable_scope('encoder'):
            encoder_cell = self._create_rnn_cell()
            self.embedding = tf.get_variable('embedding', [self.vocab_size, self.emb_size], dtype=tf.float32)
            encoder_input_emb = tf.nn.embedding_lookup(self.embedding, self.encoder_input)

            # encoder_output = [batch_size * encoder_length * rnn_size], used for computing attention socres
            # encoder_state is the last state of the rnn, used as the initial input of decoder
            encoder_output, encoder_state = tf.nn.dynamic_rnn(encoder_cell, encoder_input_emb, 
                                                            sequence_length=self.encoder_length, 
                                                            dtype=tf.float32)

        print('Constructing decoder ...')
        with tf.variable_scope('decoder'):
            encoder_length = self.encoder_length
            # if use the beam search trick, encoder outputs should be copy beam_size times through tile_batch() 
            if self.beam_search:
                print('Using beam search for decoding ...')
                encoder_output = tf.contrib.seq2seq.tile_batch(encoder_output, multiplier=self.beam_size)
                encoder_state = nest.map_structure(lambda s: tf.contrib.seq2seq.tile_batch(s, self.beam_size), encoder_state)
                encoder_length = tf.contrib.seq2seq.tile_batch(self.encoder_length, multiplier=self.beam_size)
            # choose an attention mechanism, BahdanauAttention of LuongAttention
            attention_mechanism = tf.contrib.seq2seq.BahdanauAttention(num_units=self.rnn_size,
                                                                    memory=encoder_output, # context
                                                                    memory_sequence_length=encoder_length)
            decoder_cell = self._create_rnn_cell()
            decoder_cell = tf.contrib.seq2seq.AttentionWrapper(cell=decoder_cell,
                                                                attention_mechanism=attention_mechanism,
                                                                attention_layer_size=self.rnn_size,
                                                                name='attention_wrapper')
            decoder_initial_state = decoder_cell.zero_state(batch_size=self.batch_size, dtype=tf.float32)
            decoder_initial_state = decoder_initial_state.clone(cell_state=encoder_state)
            output_layer = tf.layers.Dense(self.vocab_size, 
                                        kernel_initializer=tf.truncated_normal_initializer(mean=0.0, stddev=0.1))

            print('Model mode is %s' % self.mode)
            if self.mode == 'train':
                # delete the <end> symbol of each sequence 
                ending = tf.strided_slice(self.decoder_target, [0, 0], [self.batch_size, -1], [1, 1])
                # and add a <go> symbol at the beginning
                decoder_input = tf.concat([tf.fill([self.batch_size, 1], self.word2id['<go>']), ending], 1)
                decoder_input_emb = tf.nn.embedding_lookup(self.embedding, decoder_input)
                # when training, usually apply TrainingHelper + BasicDecoder
                training_helper = tf.contrib.seq2seq.TrainingHelper(inputs=decoder_input_emb,
                                                                    sequence_length=self.decoder_length,
                                                                    name='training_helper')
                training_decoder = tf.contrib.seq2seq.BasicDecoder(cell=decoder_cell, 
                                                                    helper=training_helper,
                                                                    initial_state=decoder_initial_state,
                                                                    output_layer=output_layer)
                # decode returns: (final_outputs, final_state, final_sequence_lengths)
                decoder_ouput, _, _ = tf.contrib.seq2seq.dynamic_decode(decoder=training_decoder,
                                                                        impute_finished=True, # mask values
                                                                        maximum_iterations=self.max_target_length)
                self.logits = tf.identity(decoder_ouput.rnn_output)
                self.predict_train = tf.argmax(self.logits, axis=-1, name='logit_to_predict')
                self.loss = tf.contrib.seq2seq.sequence_loss(logits=self.logits,
                                                            targets=self.decoder_target,
                                                            weights=self.mask)
                # Training summary of the current batch loss
                tf.summary.scalar('loss', self.loss)
                self.summary_op = tf.summary.merge_all()

                optimizer = tf.train.AdamOptimizer(self.learning_rate)
                trainable_params = tf.trainable_variables()
                gradients = tf.gradients(self.loss, trainable_params)
                clip_gradients, _ = tf.clip_by_global_norm(gradients, self.max_grad_norm)
                self.train_op = optimizer.apply_gradients(zip(clip_gradients, trainable_params))

            elif self.mode == 'decode':
                start_tokens = tf.ones([self.batch_size, ], tf.int32) * self.word2id['<go>']
                end_token = self.word2id['<eos>']

                if self.beam_search:
                    inference_decoder = tf.contrib.seq2seq.BeamSearchDecoder(cell=decoder_cell,
                                                                            embedding=self.embedding,
                                                                            start_tokens=start_tokens,
                                                                            end_token=end_token,
                                                                            initial_state=decoder_initial_state,
                                                                            beam_width=self.beam_size,
                                                                            output_layer=output_layer)
                else:
                    decoding_helper = tf.contrib.seq2seq.GreedyEmbeddingHelper(embedding=self.embedding,
                                                                                start_tokens=start_tokens,
                                                                                end_token=end_token)
                    inference_decoder = tf.contrib.seq2seq.BasicDecoder(cell=decoder_cell,
                                                                        helper=decoding_helper,
                                                                        initial_state=decoder_initial_state,
                                                                        output_layer=output_layer)
                # call the dynamci_decode function to decode, decoder_output is a named tuple
                # if use beam search, decoder_output contains (rnn_outputs, sample_id)
                #   where rnn_outputs = [batch_size, decoder_length, vocab_size]
                #         sameple_id = [batch_size, decoder_length], dtype=tf.int32
                # else, decoder_output contains (predicted_ids, beam_search_decoder_output)
                #   where predicted_ids = [batch_size, decoder_length, beam_size]
                #         beam_search_decoder_output is a named tuple of (scores, predicted_ids, parent_ids)
                decoder_output, _, _ = tf.contrib.seq2seq.dynamic_decode(decoder=inference_decoder,
                                                                        maximum_iterations=self.max_decode_len)
                if self.beam_search:
                    self.predict_decode = decoder_output.predicted_ids
                else:
                    self.predict_decode = tf.expand_dims(decoder_output.sample_id, -1)
        # the module for saving the model
        self.saver = tf.train.Saver(tf.global_variables())


    def train(self, sess, batch):
        feed_dict = {self.encoder_input: batch.encoder_input,
                    self.encoder_length: batch.encoder_length,
                    self.decoder_target: batch.decoder_target,
                    self.decoder_length: batch.decoder_length,
                    self.keep_prob: 1.0 - self.dropout_rate,
                    self.batch_size: len(batch.encoder_input)}
        _, loss, summary = sess.run([self.train_op, self.loss, self.summary_op], feed_dict=feed_dict)
        return loss, summary


    def eval(self, sess, batch):
        feed_dict = {self.encoder_input: batch.encoder_input,
                    self.encoder_length: batch.encoder_length,
                    self.decoder_target: batch.decoder_target,
                    self.decoder_length: batch.decoder_length,
                    self.keep_prob: 1.0,
                    self.batch_size: len(batch.encoder_input)}
        loss, summary = sess.run([self.loss, self.summary_op], feed_dict=feed_dict)
        return loss, summary


    def infer(self, sess, batch):
        feed_dict = {self.encoder_input: batch.encoder_input,
                    self.encoder_length: batch.encoder_length,
                    self.decoder_target: batch.decoder_target,
                    self.decoder_length: batch.decoder_length,
                    self.keep_prob: 1.0,
                    self.batch_size: len(batch.encoder_input)}
        predict = sess.run([self.predict_decode], feed_dict=feed_dict)
        return predict











