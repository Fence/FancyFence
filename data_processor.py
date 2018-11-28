import re
import os
import ipdb
import pickle
import random
import numpy as np
from tqdm import tqdm
from gensim.models import KeyedVectors


class Batch(object):
    """docstring for Batch"""
    def __init__(self):
        self.encoder_input = []
        self.encoder_length = []
        self.decoder_target = []
        self.decoder_length = []

padToken, goToken, eosToken, markToken, unkToken = 0, 1, 2, 3, 4

def loadData(args, turns='single'):
    load_success = False
    if os.path.exists(args.data_dir):
        print('Loading data from %s ...' % args.data_dir)
        try:
            word2id, id2word, embedding, train_data = pickle.load(open(args.data_dir, 'rb'))
            print('Loading Finished!\n')
            load_success = True
        except:
            print('Error occurs when loading data!\n')
    
    if not load_success:
        print('Preparing training data ...')
        wv = KeyedVectors.load_word2vec_format('data/word2vec100', binary=True)
        #ipdb.set_trace()
        data = open('data/dialogues_text.txt').readlines()
        data = [d.lower().split('__eou__') for d in data]
        train_data = []
        word2id = {'<pad>': 0, '<go>': 1, '<eos>': 2, '<mark>': 3, '<unk>': 4}
        embedding = [np.random.normal(0, 0.1, 100) for _ in range(len(word2id))]
        for dial in data:
            for utterance in dial:
                for w in utterance.split():
                    if w not in word2id and w in wv.vocab:
                        word2id[w] = len(word2id)
                        embedding.append(wv[w])
        id2word = {v: k for k, v in word2id.items()}
        embedding = np.array(embedding)

        for dial in tqdm(data, desc='Dial'):
            utterance = sent2ids(re.split(r'[\.\?\!]', dial[0])[0], word2id)
            response = sent2ids(re.split(r'[\.\?\!]', dial[1])[0], word2id)
            train_data.append([utterance, response])

        with open(args.data_dir, 'wb') as f:
            print('Saving train_data to %s ...' % args.data_dir)
            pickle.dump([word2id, id2word, embedding, train_data], f)
            print('Saving Finished!\n')

    return word2id, id2word, embedding, train_data


def sent2ids(raw_sent, word2id):
    #ipdb.set_trace()
    utterance = [] # a list of word indices
    for w in raw_sent.strip().split():
        if w in word2id:
            utterance.append(word2id[w])
        elif w in """,';":""":
            utterance.append(word2id['<mark>'])
        else:
            utterance.append(word2id['<unk>'])
    utterance.append(word2id['<eos>'])
    return utterance


def createBatch(samples):
    # padding 
    batch = Batch()
    batch.encoder_length = [len(sample[0]) for sample in samples]
    batch.decoder_length = [len(sample[1]) for sample in samples]

    max_source_length = max(batch.encoder_length)
    max_target_length = max(batch.decoder_length)

    for utterance, response in samples:
        padding = [padToken] * (max_source_length - len(utterance))
        batch.encoder_input.append(utterance + padding)

        padding = [padToken] * (max_target_length - len(response))
        batch.decoder_target.append(response + padding)

    return batch


def getBatches(data, batch_size):
    random.shuffle(data)
    batches = []
    num_samples = len(data)

    def getNextBatch():
        for i in range(0, num_samples, batch_size):
            yield data[i : min(i + batch_size, num_samples)]

    for samples in getNextBatch():
        batch = createBatch(samples)
        batches.append(batch)
    return batches


def sent2input(sent, word2id):
    wordIds = sent2ids(sent, word2id)
    batch = createBatch([[wordIds, []]])
    return batch


def predict_ids_to_words(predict_ids, id2word, beam_size):
    #ipdb.set_trace()
    for single_sent in predict_ids:
        for i in range(beam_size):
            predict_list = single_sent[:, :, i]
            predict_sent = [id2word[idx] for idx in predict_list[0]]
            print(' '.join(predict_sent))



