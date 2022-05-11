import numpy as np
import pickle
from lib.config import cfg
from evalcap.bleu.bleu import Bleu
from evalcap.cider.cider import Cider
from evalcap.rouge.rouge import Rouge
from evalcap.meteor.meteor import Meteor
import json

factory = {
    'Bleu_4': Bleu(),
    'Rouge': Rouge(),
    'Meteor': Meteor(),
    'CIDEr': Cider()
}


def get_sents(sent):
    words = []
    for word in sent:
        words.append(word)
        if word == 0:
            break
    return words


class Scorer(object):
    def __init__(self, vocab=None, json_path=None):
        super(Scorer, self).__init__()
        self.scorers = []
        self.weights = cfg.SCORER.WEIGHTS
        self.gts = pickle.load(open(cfg.SCORER.GT_PATH, 'rb'), encoding='bytes')
        if vocab is not None:
            self.vocab = json.load(open(vocab, 'r'))
        if json_path is not None:
            if isinstance(json_path, str):
                self.eval_ids = json.load(open(json_path, 'r'))
            else:
                self.eval_ids = json_path
        for name in cfg.SCORER.TYPES:
            self.scorers.append(factory[name])

    def list2dict(self, gts, hypo):
        new_gts = {}
        new_hypo = {}
        for idx, (caps_gts, caps_hypo) in enumerate(zip(gts, hypo)):
            sents = []
            for tokens in caps_gts:
                sent = ' '.join([str(i) for i in tokens])
                sents.append(sent)
            new_gts[idx] = sents
            sents = []
            for tokens in caps_hypo:
                sent = ' '.join([str(i) for i in tokens])
                sents.append(sent)
            new_hypo[idx] = sents
        return new_gts, new_hypo

    def __call__(self, ids, res):
        hypo = [[get_sents(r)] for r in res]
        gts = [self.gts[i] for i in ids]
        rewards_info = {}
        rewards = np.zeros(len(ids))
        for i, scorer in enumerate(self.scorers):
            new_gts, new_hypo = self.list2dict(gts, hypo)
            if 'bleu' in str(type(scorer)):
                score, scores = scorer.compute_score(new_gts, new_hypo, verbose=0)
                score, scores = np.array(score[-1]), np.array(scores[-1])
            else:
                score, scores = scorer.compute_score(new_gts, new_hypo)
            rewards += self.weights[i] * scores
            rewards_info[cfg.SCORER.TYPES[i]] = score
        return rewards, rewards_info