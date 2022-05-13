import os
import torch
import tqdm
import json
import lib.utils as utils
import datasets.data_loader as data_loader
from lib.config import cfg
from evalcap.bleu.bleu import Bleu
from evalcap.rouge.rouge import Rouge
from evalcap.cider.cider import Cider
from evalcap.meteor.meteor import Meteor


def score(ref, hypo):
    scorers = [
        (Bleu(4), ["Bleu_1", "Bleu_2", "Bleu_3", "Bleu_4"]),
        (Rouge(), "ROUGE_L"),
        (Meteor(), "METEOR"),
        (Cider(), "CIDEr")
    ]
    final_scores = {}
    for scorer, method in scorers:
        score, scores = scorer.compute_score(ref, hypo)
        if type(score) == list:
            for m, s in zip(method, score):
                final_scores[m] = s
        else:
            final_scores[method] = score
    return final_scores


class Evaler(object):
    def __init__(
        self,
        json_path,
        vocab,
        debug=False
    ):
        super(Evaler, self).__init__()
        self.vocab = json.load(open(vocab, 'r'))
        self.eval_ids = json.load(open(json_path, 'r'))
        self.eval_loader = data_loader.load_val(json_path)
        self.debug = debug

    def make_kwargs(self, indices, gv_feat, att_feats, att_mask):
        kwargs = {}
        kwargs[cfg.PARAM.INDICES] = indices
        kwargs[cfg.PARAM.GLOBAL_FEAT] = gv_feat
        kwargs[cfg.PARAM.ATT_FEATS] = att_feats
        kwargs[cfg.PARAM.ATT_FEATS_MASK] = att_mask
        kwargs['BEAM_SIZE'] = cfg.INFERENCE.BEAM_SIZE
        kwargs['GREEDY_DECODE'] = cfg.INFERENCE.GREEDY_DECODE
        return kwargs

    def __call__(self, model, rname):
        model.eval()
        
        results = {}
        ref = {}
        with torch.no_grad():
            for step, (indices, input_seq, target_seq, gv_feat, att_feats, att_mask, mlc_label) in tqdm.tqdm(enumerate(self.eval_loader)):
                target_seq = target_seq.cuda()
                gv_feat = gv_feat.cuda()
                att_feats = att_feats.cuda()
                if len(att_mask) == 2:
                    att_mask = (att_mask[0].cuda(), att_mask[1].cuda())
                else:
                    att_mask = att_mask.cuda()
                kwargs = self.make_kwargs(indices, gv_feat, att_feats, att_mask)
                if kwargs['BEAM_SIZE'] > 1:
                    seq, _ = model.module.decode_beam(**kwargs)
                else:
                    seq, _ = model.module.decode(**kwargs)
                sents = utils.decode_sequence(self.vocab, seq.data)
                gts = utils.decode_sequence(self.vocab, target_seq)
                for sid, sent, gt in zip(indices, sents, gts):
                    results[sid] = [sent]
                    ref[sid] = [gt]

        eval_res = score(ref=ref, hypo=results)
        result_folder = os.path.join(cfg.ROOT_DIR, 'result')
        if not os.path.exists(result_folder):
            os.mkdir(result_folder)
        json.dump(results, open(os.path.join(result_folder, 'result_' + rname +'.json'), 'w'))
        model.train()
        return eval_res
