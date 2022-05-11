import copy
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from lib.config import cfg

from layers.low_rank import LowRank
import blocks
import lib.utils as utils
from models.basic_model import BasicModel
from layers.positional_encoding import PositionalEncoding
from models.xtransformer import XTransformer

def mutimodel_decode_beam(models, **kwargs):
    att_feats_origin = kwargs[cfg.PARAM.ATT_FEATS]
    att_mask = kwargs[cfg.PARAM.ATT_FEATS_MASK]
    beam_size = kwargs['BEAM_SIZE']
    batch_size = att_feats_origin.size(0)
    seq_logprob = torch.zeros((batch_size, 1, 1)).cuda()
    log_probs = []
    selected_words = None # only
    seq_mask = torch.ones((batch_size, beam_size, 1)).cuda()

    models_data = []
    for model in models:
        att_feats = model.module.att_embed(att_feats_origin)
        gx, encoder_out = model.module.encoder(att_feats, att_mask)
        p_att_feats = model.module.decoder.precompute(encoder_out)

        state = None
        wt = Variable(torch.zeros(batch_size, dtype=torch.long).cuda())
        # kwargs[cfg.PARAM.ATT_FEATS] = encoder_out
        # kwargs[cfg.PARAM.GLOBAL_FEAT] = gx
        # kwargs[cfg.PARAM.P_ATT_FEATS] = p_att_feats

        outputs = []
        model.module.decoder.init_buffer(batch_size)
        data_dict = dict()
        data_dict['ATT_FEATS'] = encoder_out
        data_dict['GLOBAL_FEAT'] = gx
        data_dict['P_ATT_FEATS'] = p_att_feats
        data_dict['WT'] = wt
        data_dict['STATE'] = state
        models_data.append(data_dict)
        
    for t in range(cfg.MODEL.SEQ_LEN):
        cur_beam_size = 1 if t == 0 else beam_size
        temp_word_logprob_list = []
        for i in range(len(models)):
            kwargs[cfg.PARAM.WT] = models_data[i]['WT']
            kwargs[cfg.PARAM.STATE] = models_data[i]['STATE']
            kwargs[cfg.PARAM.ATT_FEATS] = models_data[i]['ATT_FEATS']
            kwargs[cfg.PARAM.GLOBAL_FEAT] = models_data[i]['GLOBAL_FEAT']
            kwargs[cfg.PARAM.P_ATT_FEATS] = models_data[i]['P_ATT_FEATS']
            word_logprob_, state = models[i].module.get_logprobs_state(**kwargs)
            word_logprob_ = word_logprob_.view(batch_size, cur_beam_size, -1)
            temp_word_logprob_list.append(word_logprob_)
            models_data[i]['STATE'] = state
        word_logprob = sum(temp_word_logprob_list)/len(models)
        candidate_logprob = seq_logprob + word_logprob

        # Mask sequence if it reaches EOS
        if t > 0:
            mask = (selected_words.view(batch_size, cur_beam_size) != 0).float().unsqueeze(-1)
            seq_mask = seq_mask * mask
            word_logprob = word_logprob * seq_mask.expand_as(word_logprob)
            old_seq_logprob = seq_logprob.expand_as(candidate_logprob).contiguous()
            old_seq_logprob[:, :, 1:] = -999
            candidate_logprob = seq_mask * candidate_logprob + old_seq_logprob * (1 - seq_mask)
        selected_idx, selected_logprob = models[0].module.select(batch_size, beam_size, t, candidate_logprob)
        selected_beam = selected_idx // candidate_logprob.shape[-1]
        # selected_words = selected_idx - selected_beam * candidate_logprob.shape[-1]
        selected_words = selected_idx % candidate_logprob.shape[-1]
        for i in range(len(models)):
            models[i].module.decoder.apply_to_states(models[i].module._expand_state(batch_size, beam_size, cur_beam_size, selected_beam))
        seq_logprob = selected_logprob.unsqueeze(-1)
        seq_mask = torch.gather(seq_mask, 1, selected_beam.unsqueeze(-1))
        outputs = list(torch.gather(o, 1, selected_beam.unsqueeze(-1)) for o in outputs)
        outputs.append(selected_words.unsqueeze(-1))

        this_word_logprob = torch.gather(word_logprob, 1,
            selected_beam.unsqueeze(-1).expand(batch_size, beam_size, word_logprob.shape[-1]))
        this_word_logprob = torch.gather(this_word_logprob, 2, selected_words.unsqueeze(-1))
        log_probs = list(
            torch.gather(o, 1, selected_beam.unsqueeze(-1).expand(batch_size, beam_size, 1)) for o in log_probs)
        log_probs.append(this_word_logprob)
        selected_words = selected_words.view(-1, 1)
        for i in range(len(models)):
            models_data[i]['WT'] = selected_words.squeeze(-1)

        if t == 0:
            for i in range(len(models)):
                models_data[i]['ATT_FEATS'] = utils.expand_tensor(models_data[i]['ATT_FEATS'], beam_size)
                models_data[i]['GLOBAL_FEAT'] = utils.expand_tensor(models_data[i]['GLOBAL_FEAT'], beam_size)
                att_mask_expand = utils.expand_tensor(att_mask, beam_size)
                models_data[i]['STATE'][0] = models_data[i]['STATE'][0].squeeze(0)
                models_data[i]['STATE'][0] = utils.expand_tensor(models_data[i]['STATE'][0], beam_size)
                models_data[i]['STATE'][0] = models_data[i]['STATE'][0].unsqueeze(0)

                p_att_feats_tmp = []
                for p_feat in models_data[i]['P_ATT_FEATS']:
                    p_key, p_value2 = p_feat
                    p_key = utils.expand_tensor(p_key, beam_size)
                    p_value2 = utils.expand_tensor(p_value2, beam_size)
                    p_att_feats_tmp.append((p_key, p_value2))

                kwargs[cfg.PARAM.ATT_FEATS_MASK] = att_mask_expand
                models_data[i]['P_ATT_FEATS'] = p_att_feats_tmp

    seq_logprob, sort_idxs = torch.sort(seq_logprob, 1, descending=True)
    outputs = torch.cat(outputs, -1)
    outputs = torch.gather(outputs, 1, sort_idxs.expand(batch_size, beam_size, cfg.MODEL.SEQ_LEN))
    log_probs = torch.cat(log_probs, -1)
    log_probs = torch.gather(log_probs, 1, sort_idxs.expand(batch_size, beam_size, cfg.MODEL.SEQ_LEN))

    outputs = outputs.contiguous()[:, 0]
    log_probs = log_probs.contiguous()[:, 0]
    for i in range(len(models)):
        models[i].module.decoder.clear_buffer()
    return outputs, log_probs

