import os
import ipdb
import math
import argparse
import tensorflow as tf
from tqdm import tqdm
from utils import timeit, str2bool, print_args
from model import Seq2SeqModel
from data_processor import getBatches


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_dir',       type=str,       default='models/',  help='')
    parser.add_argument('--model_name',      type=str,       default='chat_bot', help='')
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
    args.word2idx = {'eos': 1, '<go>': 2, '<unk>': 3}
    print_args(args)

    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=args.gpu_fraction)
    with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
        ipdb.set_trace()
        model = Seq2SeqModel(args)

        ckpt = tf.train.get_checkpoint_state(args.model_dir)
        if ckpt and tf.train.checkpoint_exists(cpkt.model_checkpoint_path):
            print('Reloading model from %s' % cpkt.model_checkpoint_path)
            model.restore(sess, ckpt.model_checkpoint_path)
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

                if current_step % save_per_steps == 0:
                    perplexity = math.exp(float(loss)) if loss < 300 else float('inf')
                    tqdm.write('----- Step %d -- Loss %.2f -- Perplexity %.2f' % (current_step, loss, perplexity))
                    summary_writer.add_summary(summary, current_step)
                    checkpoint_path = os.path.join(args.model_dir, args.model_name)
                    model.saver.save(sess, checkpoint_path, global_step=current_step)




