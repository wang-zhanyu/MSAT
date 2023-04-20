import nltk
import pickle
import argparse
from collections import Counter
import json
import numpy as np
from tqdm import tqdm
import re
import os
from random import shuffle


class Vocabulary(object):
    """Simple vocabulary wrapper."""
    def __init__(self):
        self.word2idx = {}
        self.idx2word = {}
        self.idx = 0

    def add_word(self, word):
        if not word in self.word2idx:
            self.word2idx[word] = self.idx
            self.idx2word[self.idx] = word
            self.idx += 1

    def __call__(self, word):
        if not word in self.word2idx:
            return self.word2idx['<unk>']
        return self.word2idx[word]

    def __len__(self):
        return len(self.word2idx)


def clean_report_mimic_cxr(report):
    report_cleaner = lambda t: t.replace('\n', ' ').replace('__', '_').replace('__', '_').replace('__', '_') \
        .replace('__', '_').replace('__', '_').replace('__', '_').replace('__', '_').replace('  ', ' ') \
        .replace('  ', ' ').replace('  ', ' ').replace('  ', ' ').replace('  ', ' ').replace('  ', ' ') \
        .replace('..', '.').replace('..', '.').replace('..', '.').replace('..', '.').replace('..', '.') \
        .replace('..', '.').replace('..', '.').replace('..', '.').replace('1. ', '').replace('. 2. ', '. ') \
        .replace('. 3. ', '. ').replace('. 4. ', '. ').replace('. 5. ', '. ').replace(' 2. ', '. ') \
        .replace(' 3. ', '. ').replace(' 4. ', '. ').replace(' 5. ', '. ') \
        .strip().lower().split('. ')
    sent_cleaner = lambda t: re.sub('[.,?;*!%^&_+():-\[\]{}]', '', t.replace('"', '').replace('/', '')
                                    .replace('\\', '').replace("'", '').strip().lower())
    tokens = [sent_cleaner(sent) for sent in report_cleaner(report) if sent_cleaner(sent) != []]
    report = ' . '.join(tokens) + ' .'
    return report


def build_vocab(cap_path, threshold):
    """Build a simple vocabulary wrapper."""
    captions = json.load(open(cap_path, 'r'))
    captions = captions['images']
    counter = Counter()
    for i, c in tqdm(enumerate(captions)):
        caption = c['caption']
        caption = clean_report_mimic_cxr(caption)
        tokens = nltk.tokenize.word_tokenize(caption.lower())
        counter.update(tokens)

    # If the word frequency is less than 'threshold', then the word is discarded.
    words = [word for word, cnt in counter.items() if cnt >= threshold]

    # Create a vocab wrapper and add some special tokens.
    vocab = Vocabulary()
    vocab.add_word('<pad>')
    vocab.add_word('<unk>')

    # Add the words to the vocabulary.
    for i, word in enumerate(words):
        vocab.add_word(word)
    vocab = {'idx': vocab.idx, 'word2idx': vocab.word2idx, 'idx2word': vocab.idx2word}
    return vocab


def cap2idx(args, vocab):
    data = json.load(open(args.annotation, 'r'))
    data = data['images']
    mimic_train_input = {}
    mimic_train_target = {}
    mimic_train_gt = {}
    max_len = 60
    for line in tqdm(data):
        idx = line['id']
        cap = line['caption']
        cap = clean_report_mimic_cxr(cap)
        tokens = nltk.tokenize.word_tokenize(cap.lower())
        word2idx = [vocab['word2idx'][i] if i in vocab['word2idx'] else vocab['word2idx']['<unk>'] for i in tokens]
        input = [0] + word2idx
        if len(word2idx) > max_len-1:
            target = word2idx[:max_len-1] + [0]
        else:
            target = word2idx + [0]
        if len(input) < max_len:
            while len(input) < max_len:
                input.append(0)
        else:
            input = input[:max_len]

        if len(target) < max_len:
            while len(target) < max_len:
                target.append(-1)
        else:
            target = target[:max_len]

        if len(word2idx) < max_len:
            target_gt = word2idx + [0]
        else:
            target_gt = word2idx[:max_len]

        mimic_train_input[idx] = np.array([input])
        mimic_train_target[idx] = np.array([target])
        mimic_train_gt[idx] = [target_gt]

    with open(os.path.join(args.save_path, 'mimic_train_gt.pkl'), 'wb') as f:
        pickle.dump(mimic_train_gt, f)
    with open(os.path.join(args.save_path, 'mimic_train_input.pkl'), 'wb') as f:
        pickle.dump(mimic_train_input, f)
    with open(os.path.join(args.save_path, 'mimic_train_target.pkl'), 'wb') as f:
        pickle.dump(mimic_train_target, f)


def get_mlc_label(args):
    max_len = 768
    data = json.load(open(args.annotation, 'r'))
    radgraph = json.load(open(args.radgraph, 'r'))
    tokens = [j['tokens'] for k, v in radgraph.items() for i, j in v.items() for x, y in j.items()]
    tokens = list(set(tokens))
    mlc_label = Counter(tokens)
    mlc_label = sorted(mlc_label.items(), key=lambda x: x[1], reverse=True)
    mlc_label = [i[0] for i in mlc_label][:max_len]
    shuffle(mlc_label)
    label2idx = {label: i for i, label in enumerate(mlc_label)}
    data = data['images']
    mimic_mlc_label = {}
    for line in tqdm(data):
        idx = line['id']
        cap = line['caption']
        cap = clean_report_mimic_cxr(cap)
        tokens = cap.split(' ')
        label = np.zeros((1, max_len))
        for i in tokens:
            if i in mlc_label:
                j = label2idx[i]
                label[:, j] = 1 
        mimic_mlc_label[idx] = np.array(label)

    with open(os.path.join(args.save_path, 'mimic_mlc_label.pkl'), 'wb') as f:
        pickle.dump(mimic_mlc_label, f)


def main(args):
    vocab = build_vocab(args.annotation, threshold=args.threshold)
    vocab_path = args.vocab_path
    with open(os.path.join(args.save_path, vocab_path), 'w') as f:
        json.dump(vocab, f)
    cap2idx(args, vocab=vocab)
    get_mlc_label(args)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--annotation', type=str,
                        default='data/mimic/mimic.json',
                        help='path for train annotation file')
    parser.add_argument('--save_path', type=str, default='./data/mimic',
                        help='path for saving vocabulary wrapper')
    parser.add_argument('--radgraph', type=str, default='data/mimic/MIMIC-CXR_graphs.json',
                        help='path for saving vocabulary wrapper')
    parser.add_argument('--threshold', type=int, default=5,
                        help='minimum word count threshold')

    args = parser.parse_args()
    main(args)
