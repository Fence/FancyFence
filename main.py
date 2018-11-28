import os
import ipdb
import math
import argparse
import tensorflow as tf
from tqdm import tqdm
from utils import timeit, str2bool, print_args
from model import Seq2SeqModel
from data_processor import getBatches, loadData, sent2input, predict_ids_to_words



def predict(args):
    # loading data
    args.word2id, args.id2word, embedding, train_data = loadData(args, turns='single')
    # set GPU fraction
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=args.gpu_fraction)
    with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
        # build model structure
        model = Seq2SeqModel(args, embedding)
        # load model
        ckpt = tf.train.get_checkpoint_state(args.model_dir)
        if ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path):
            print('Reloading model from %s' % ckpt.model_checkpoint_path)
            model.saver.restore(sess, ckpt.model_checkpoint_path)
        else:
            print('Error: no pre-trained model %s ...' % args.model_dir)
            return

        ipdb.set_trace()
        sentence = input('> ')
        while sentence:
            batch = sent2input(sentence, args.word2id)
            predict_ids = model.infer(sess, batch)
            predict_ids_to_words(predict_ids, args.id2word, args.beam_size)
            print('> ', ' ')
            sentence = input('\n> ')




def train_model(args):
    # loading data
    args.word2id, args.id2word, embedding, train_data = loadData(args, turns='single')
    # set GPU fraction
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=args.gpu_fraction)
    with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
        # build model
        model = Seq2SeqModel(args, embedding)

        ipdb.set_trace()
        ckpt = tf.train.get_checkpoint_state(args.model_dir)
        if ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path):
            print('Reloading model from %s' % ckpt.model_checkpoint_path)
            model.saver.restore(sess, ckpt.model_checkpoint_path)
        else:
            print('Created new model parameters ...')
            sess.run(tf.global_variables_initializer())

        current_step = 0
        summary_writer = tf.summary.FileWriter(args.model_dir, graph=sess.graph)
        for epoch in range(args.num_epochs):
            print('----- Epoch {}/{} -----'.format(epoch + 1, args.num_epochs))
            batches = getBatches(train_data, args.batch_size)

            for nextBatch in tqdm(batches, desc='Training'):
                loss, summary = model.train(sess, nextBatch)
                current_step += 1

                if current_step % args.save_per_steps == 0:
                    perplexity = math.exp(float(loss)) if loss < 300 else float('inf')
                    tqdm.write('----- Step %d -- Loss %.2f -- Perplexity %.2f' % (current_step, loss, perplexity))
                    summary_writer.add_summary(summary, current_step)
                    checkpoint_path = os.path.join(args.model_dir, args.model_name)
                    model.saver.save(sess, checkpoint_path, global_step=current_step)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_dir',       type=str,       default='models/',  help='')
    parser.add_argument('--model_name',      type=str,       default='chat_bot', help='')
    parser.add_argument('--data_dir',        type=str,       default='data/train_data.pkl', help='')
    parser.add_argument('--num_epochs',      type=int,       default=100,        help='')
    parser.add_argument('--save_per_steps',  type=int,       default=100,        help='')
    parser.add_argument('--gpu_fraction',    type=float,     default=0.2,        help='')

    parser.add_argument('--rnn_size',        type=int,       default=100,        help='')
    parser.add_argument('--num_layers',      type=int,       default=2,          help='')
    parser.add_argument('--emb_size',        type=int,       default=100,        help='')
    parser.add_argument('--max_decode_len',  type=int,       default=20,         help='')
    parser.add_argument('--batch_size',      type=int,       default=32,         help='')
    parser.add_argument('--dropout_rate',    type=float,     default=0.5,        help='')
    parser.add_argument('--learning_rate',   type=float,     default=0.001,      help='')
    parser.add_argument('--max_grad_norm',   type=float,     default=5.0,        help='')
    
    parser.add_argument('--beam_search',     type=str2bool,  default=True,       help='')
    parser.add_argument('--beam_size',       type=int,       default=5,          help='')
    parser.add_argument('--mode',            type=str,       default='train',    help='')
    
    args = parser.parse_args()
    #args.word2idx = {'eos': 1, '<go>': 2, '<unk>': 3}
    if not args.beam_search:
        args.beam_size = 1
    print_args(args)

    if args.mode == 'train':
        train_model(args)
    elif args.mode == 'decode':
        predict(args)




