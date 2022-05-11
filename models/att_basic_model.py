import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn import init
import blocks
import lib.utils as utils
from lib.config import cfg
from models.basic_model import BasicModel

class CrossEn(nn.Module):
    def __init__(self,):
        super(CrossEn, self).__init__()

    def forward(self, sim_matrix):
        logpt = F.log_softmax(sim_matrix, dim=-1)
        logpt = torch.diag(logpt)
        nce_loss = -logpt
        sim_loss = nce_loss.mean()
        return sim_loss

class ExternalFusionAttention(nn.Module):

    def __init__(self, d_model,S=64):
        super().__init__()
        self.fc_clip = nn.Linear(512, 1024, bias=False)
        self.weight_clip = nn.Linear(1024, 1, bias=False)
        self.fc_slowfast = nn.Linear(2304, 1024, bias=False)
        self.weight_slowfast = nn.Linear(1024, 1, bias=False)
        self.fc_s3d = nn.Linear(1024, 1024, bias=False)
        self.weight_s3d = nn.Linear(1024, 1, bias=False)  # 512
        self.fc_resent = nn.Linear(2048, 1024, bias=False)
        self.weight_resnet = nn.Linear(1024, 1, bias=False)  # 512
        self.sigmoid = nn.Sigmoid()
        self.activate = nn.ReLU()
        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                m = m.cuda()
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def forward(self, queries):
        clip = self.fc_clip(queries[0])
        clip = self.activate(clip)
        weight_clip = self.sigmoid(self.weight_clip(clip).mean(1).unsqueeze(1))

        slowfast = self.fc_slowfast(queries[3])
        slowfast = self.activate(slowfast)
        weight_slowfast = self.sigmoid(self.weight_slowfast(slowfast).mean(1).unsqueeze(1))

        s3d = self.fc_s3d(queries[1])
        s3d = self.activate(s3d)
        weight_s3d = self.sigmoid(self.weight_s3d(s3d).mean(1).unsqueeze(1))

        resnet = self.fc_resent(queries[2])
        resnet = self.activate(resnet)
        weight_resnet = self.sigmoid(self.weight_resnet(resnet).mean(1).unsqueeze(1))

        out = clip * weight_clip + slowfast * weight_slowfast + s3d * weight_s3d + resnet * weight_resnet
        # out = 0.4*clip+0.3*slowfast+0.2*s3d+0.1*resnet

        return out


class Wav2Vec2LayerNormConvLayer(nn.Module):
    def __init__(self, input_dim, output_dim, norm=False, drop=0):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.conv = nn.Conv1d(
            self.input_dim,
            self.output_dim,
            kernel_size=2,
            stride=2
        )
        self.layer_norm = nn.LayerNorm(self.output_dim, elementwise_affine=True)
        if drop>0:
            self.dropout = nn.Dropout(drop)
        else:
            self.dropout = None
        # self.activation = ACT2FN[config.feat_extract_activation]

    def forward(self, hidden_states):
        hidden_states = self.conv(hidden_states)

        hidden_states = hidden_states.transpose(-2, -1)
        hidden_states = self.layer_norm(hidden_states)
        if self.dropout is not None:
            hidden_states = self.dropout(hidden_states)
        hidden_states = hidden_states.transpose(-2, -1)

        # hidden_states = self.activation(hidden_states)
        return hidden_states

class AttBasicModel(BasicModel):
    def __init__(self):
        super(AttBasicModel, self).__init__()
        self.ss_prob = 0.0                               # Schedule sampling probability
        self.vocab_size = cfg.MODEL.VOCAB_SIZE + 1       # include <BOS>/<EOS>
        self.sub_vocab_size = cfg.MODEL.SUB_VOCAB_SIZE
        self.use_sub = cfg.DATA_LOADER.SUBTITLES_PATH is not None
        self.att_dim = cfg.MODEL.ATT_FEATS_EMBED_DIM \
            if cfg.MODEL.ATT_FEATS_EMBED_DIM > 0 else cfg.MODEL.ATT_FEATS_DIM

        # word embed
        sequential = [nn.Embedding(self.vocab_size, cfg.MODEL.WORD_EMBED_DIM)]
        sequential.append(utils.activation(cfg.MODEL.WORD_EMBED_ACT))
        if cfg.MODEL.WORD_EMBED_NORM == True:
            sequential.append(nn.LayerNorm(cfg.MODEL.WORD_EMBED_DIM))
        if cfg.MODEL.DROPOUT_WORD_EMBED > 0:
            sequential.append(nn.Dropout(cfg.MODEL.DROPOUT_WORD_EMBED))
        self.word_embed = nn.Sequential(*sequential)
        # sub_embed
        if self.use_sub:
            sequential = []
            # sequential = [nn.Embedding(self.sub_vocab_size, cfg.MODEL.WORD_EMBED_DIM)]
            sequential = [nn.Linear(self.sub_vocab_size, cfg.MODEL.ATT_FEATS_EMBED_DIM)]
            # sequential.append(utils.activation(cfg.MODEL.WORD_EMBED_ACT))
            # if cfg.MODEL.WORD_EMBED_NORM == True:
            #     sequential.append(nn.LayerNorm(cfg.MODEL.WORD_EMBED_DIM))
            # if cfg.MODEL.DROPOUT_WORD_EMBED > 0:
            #     sequential.append(nn.Dropout(cfg.MODEL.DROPOUT_WORD_EMBED))
            # sequential.append(nn.Linear(cfg.MODEL.WORD_EMBED_DIM, cfg.MODEL.ATT_FEATS_EMBED_DIM))
            sequential.append(utils.activation(cfg.MODEL.ATT_FEATS_EMBED_ACT))
            if cfg.MODEL.DROPOUT_ATT_EMBED > 0:
                sequential.append(nn.Dropout(cfg.MODEL.DROPOUT_ATT_EMBED))
            if cfg.MODEL.ATT_FEATS_NORM == True:
                sequential.append(torch.nn.LayerNorm(cfg.MODEL.ATT_FEATS_EMBED_DIM))
            self.sub_embed = nn.Sequential(*sequential)
            self.token_type_embeddings = nn.Embedding(2, cfg.MODEL.ATT_FEATS_EMBED_DIM)
            self.cross_layer = nn.Sequential(
                nn.LayerNorm(cfg.MODEL.ATT_FEATS_EMBED_DIM),
                nn.Dropout(0.1)
            )
        # global visual feat embed
        sequential = []
        if cfg.MODEL.GVFEAT_EMBED_DIM > 0:
            sequential.append(nn.Linear(cfg.MODEL.GVFEAT_DIM, cfg.MODEL.GVFEAT_EMBED_DIM))
        sequential.append(utils.activation(cfg.MODEL.GVFEAT_EMBED_ACT))
        if cfg.MODEL.DROPOUT_GV_EMBED > 0:
            sequential.append(nn.Dropout(cfg.MODEL.DROPOUT_GV_EMBED))
        self.gv_feat_embed = nn.Sequential(*sequential) if len(sequential) > 0 else None

        # attention feats embed
        sequential = []
        # speech attention feats embed
        if cfg.DATA_LOADER.DATA_TYPE == 'speech' and cfg.MODEL.SPEECH_TYPE == 'CNN':
            sequential.append(Wav2Vec2LayerNormConvLayer(cfg.MODEL.ATT_FEATS_DIM, cfg.MODEL.ATT_FEATS_EMBED_DIM,cfg.MODEL.ATT_FEATS_NORM,cfg.MODEL.DROPOUT_ATT_EMBED))
            for i in range(cfg.MODEL.CNN_LAYER-1):
                sequential.append(Wav2Vec2LayerNormConvLayer(cfg.MODEL.ATT_FEATS_EMBED_DIM, cfg.MODEL.ATT_FEATS_EMBED_DIM,cfg.MODEL.ATT_FEATS_NORM,cfg.MODEL.DROPOUT_ATT_EMBED))
            self.att_embed = nn.Sequential(*sequential) if len(sequential) > 0 else None
        else:
            if cfg.MODEL.ATT_FEATS_EMBED_DIM > 0:
                sequential.append(nn.Linear(cfg.MODEL.ATT_FEATS_DIM, cfg.MODEL.ATT_FEATS_EMBED_DIM))
            sequential.append(utils.activation(cfg.MODEL.ATT_FEATS_EMBED_ACT))
            if cfg.MODEL.DROPOUT_ATT_EMBED > 0:
                sequential.append(nn.Dropout(cfg.MODEL.DROPOUT_ATT_EMBED))
            if cfg.MODEL.ATT_FEATS_NORM == True:
                sequential.append(torch.nn.LayerNorm(cfg.MODEL.ATT_FEATS_EMBED_DIM))
            self.att_embed = nn.Sequential(*sequential) if len(sequential) > 0 else None
        #####################
        if cfg.MODEL.FUSION:
            sequential = []
            if cfg.MODEL.ATT_FEATS_EMBED_DIM > 0:
                sequential.append(nn.Linear(1024, cfg.MODEL.ATT_FEATS_EMBED_DIM))
            sequential.append(utils.activation(cfg.MODEL.ATT_FEATS_EMBED_ACT))
            if cfg.MODEL.DROPOUT_ATT_EMBED > 0:
                sequential.append(nn.Dropout(cfg.MODEL.DROPOUT_ATT_EMBED))
            if cfg.MODEL.ATT_FEATS_NORM == True:
                sequential.append(torch.nn.LayerNorm(cfg.MODEL.ATT_FEATS_EMBED_DIM))
            self.att_embed_s3d = nn.Sequential(*sequential) if len(sequential) > 0 else None

            sequential = []
            if cfg.MODEL.ATT_FEATS_EMBED_DIM > 0:
                sequential.append(nn.Linear(2048, cfg.MODEL.ATT_FEATS_EMBED_DIM))
            sequential.append(utils.activation(cfg.MODEL.ATT_FEATS_EMBED_ACT))
            if cfg.MODEL.DROPOUT_ATT_EMBED > 0:
                sequential.append(nn.Dropout(cfg.MODEL.DROPOUT_ATT_EMBED))
            if cfg.MODEL.ATT_FEATS_NORM == True:
                sequential.append(torch.nn.LayerNorm(cfg.MODEL.ATT_FEATS_EMBED_DIM))
            self.att_embed_resnet = nn.Sequential(*sequential) if len(sequential) > 0 else None

            sequential = []
            if cfg.MODEL.ATT_FEATS_EMBED_DIM > 0:
                sequential.append(nn.Linear(2304, cfg.MODEL.ATT_FEATS_EMBED_DIM))
            sequential.append(utils.activation(cfg.MODEL.ATT_FEATS_EMBED_ACT))
            if cfg.MODEL.DROPOUT_ATT_EMBED > 0:
                sequential.append(nn.Dropout(cfg.MODEL.DROPOUT_ATT_EMBED))
            if cfg.MODEL.ATT_FEATS_NORM == True:
                sequential.append(torch.nn.LayerNorm(cfg.MODEL.ATT_FEATS_EMBED_DIM))
            self.att_embed_slowfast = nn.Sequential(*sequential) if len(sequential) > 0 else None
        ###################

        self.dropout_lm  = nn.Dropout(cfg.MODEL.DROPOUT_LM) if cfg.MODEL.DROPOUT_LM > 0 else None
        self.logit = nn.Linear(cfg.MODEL.RNN_SIZE, self.vocab_size)
        self.p_att_feats = nn.Linear(self.att_dim, cfg.MODEL.ATT_HIDDEN_SIZE) \
            if cfg.MODEL.ATT_HIDDEN_SIZE > 0 else None

        # bilinear
        if cfg.MODEL.BILINEAR.DIM > 0:
            self.p_att_feats = None
            self.encoder_layers = blocks.create(
                cfg.MODEL.BILINEAR.ENCODE_BLOCK, 
                embed_dim = cfg.MODEL.BILINEAR.DIM, 
                att_type = cfg.MODEL.BILINEAR.ATTTYPE,
                att_heads = cfg.MODEL.BILINEAR.HEAD,
                att_mid_dim = cfg.MODEL.BILINEAR.ENCODE_ATT_MID_DIM,
                att_mid_drop = cfg.MODEL.BILINEAR.ENCODE_ATT_MID_DROPOUT,
                dropout = cfg.MODEL.BILINEAR.ENCODE_DROPOUT, 
                layer_num = cfg.MODEL.BILINEAR.ENCODE_LAYERS
            )
        self.loss_fct = CrossEn()

    def _get_feat_extract_output_lengths(self, input_lengths):
        """
        Computes the output length of the convolutional layers
        """

        def _conv_out_length(input_length, kernel_size, stride):
            # 1D convolutional layer output length formula taken
            # from https://pytorch.org/docs/stable/generated/torch.nn.Conv1d.html
            return (input_length - kernel_size) // stride + 1

        for kernel_size, stride in zip([2 for i in range(cfg.MODEL.CNN_LAYER)], [2 for i in range(cfg.MODEL.CNN_LAYER)]):
            input_lengths = _conv_out_length(input_lengths, kernel_size, stride)

        return input_lengths

    def _get_feature_vector_attention_mask(self, feature_vector_length, attention_mask):
        output_lengths = self._get_feat_extract_output_lengths(attention_mask.sum(-1)).to(torch.long)
        batch_size = attention_mask.shape[0]

        attention_mask = torch.zeros(
            (batch_size, feature_vector_length), dtype=attention_mask.dtype, device=attention_mask.device
        )
        # these two operations makes sure that all values before the output lengths idxs are attended to
        attention_mask[(torch.arange(attention_mask.shape[0], device=attention_mask.device), output_lengths - 1)] = 1
        attention_mask = attention_mask.flip([-1]).cumsum(-1).flip([-1])
        return attention_mask        

    def init_hidden(self, batch_size):
        return [Variable(torch.zeros(self.num_layers, batch_size, cfg.MODEL.RNN_SIZE).cuda()),
                Variable(torch.zeros(self.num_layers, batch_size, cfg.MODEL.RNN_SIZE).cuda())]

    def make_kwargs(self, wt, gv_feat, att_feats, att_mask, p_att_feats, state, **kgs):
        kwargs = kgs
        kwargs[cfg.PARAM.WT] = wt
        kwargs[cfg.PARAM.GLOBAL_FEAT] = gv_feat
        kwargs[cfg.PARAM.ATT_FEATS] = att_feats
        kwargs[cfg.PARAM.ATT_FEATS_MASK] = att_mask
        kwargs[cfg.PARAM.P_ATT_FEATS] = p_att_feats
        kwargs[cfg.PARAM.STATE] = state
        return kwargs

    def preprocess(self, retrieval=False, **kwargs):
        gv_feat = kwargs[cfg.PARAM.GLOBAL_FEAT]
        att_feats = kwargs[cfg.PARAM.ATT_FEATS]
        att_mask = kwargs[cfg.PARAM.ATT_FEATS_MASK]

        # embed gv_feat
        if self.gv_feat_embed is not None:
            gv_feat = self.gv_feat_embed(gv_feat)
        
        # embed att_feats
        if self.att_embed is not None:
            # fuse feature
            if cfg.MODEL.FUSION:
                queries = [att_feats[:, :, :512], att_feats[:, :, 512:1536], att_feats[:, :, 1536:3584],
                           att_feats[:, :, 3584:]]
                att_feat_clip = self.att_embed(queries[0])
                att_feat_s3d = self.att_embed_s3d(queries[1])
                att_feat_resnet = self.att_embed_resnet(queries[2])
                att_feat_slowfast = self.att_embed_slowfast(queries[3])
                att_feats = torch.stack([att_feat_clip, att_feat_slowfast, att_feat_s3d, att_feat_resnet])
                att_feats = torch.mean(att_feats, 0)
            elif cfg.DATA_LOADER.DATA_TYPE == 'speech' and cfg.MODEL.SPEECH_TYPE == 'CNN':
                att_feats = att_feats.transpose(-1, -2)
                att_feats = self.att_embed(att_feats)
                att_feats = att_feats.transpose(-1, -2)
                att_mask = self._get_feature_vector_attention_mask(att_feats.shape[1], att_mask)
            else:
                att_feats = self.att_embed(att_feats)

        if self.use_sub:
            sub_seq = kwargs[cfg.PARAM.SUB_SEQ]
            token_type_ids = kwargs[cfg.PARAM.TOKEN_TYPE_IDS]
            sub_embeddings = self.sub_embed(sub_seq)
            if len(sub_embeddings.shape)!=3:
                sub_embeddings = sub_embeddings.unsqueeze(1)
            if retrieval:
                video_mask, seq_mask = att_mask
                return att_feats, sub_embeddings, video_mask, seq_mask
            token_type_embeddings = self.token_type_embeddings(token_type_ids)
            att_feats = torch.cat((att_feats,sub_embeddings),-2) + token_type_embeddings
            att_feats = self.cross_layer(att_feats)
            
        p_att_feats = self.p_att_feats(att_feats) if self.p_att_feats is not None else None
        # bilinear
        if cfg.MODEL.BILINEAR.DIM > 0:
            gv_feat, att_feats = self.encoder_layers(gv_feat, att_feats, att_mask)
            keys, value2s = self.attention.precompute(att_feats, att_feats)
            p_att_feats = torch.cat([keys, value2s], dim=-1)

        return gv_feat, att_feats, att_mask, p_att_feats

    # gv_feat -- batch_size * cfg.MODEL.GVFEAT_DIM
    # att_feats -- batch_size * att_num * att_feats_dim
    def forward(self, **kwargs): 
        seq = kwargs[cfg.PARAM.INPUT_SENT] 
        gv_feat, att_feats, att_mask, p_att_feats = self.preprocess(**kwargs)
        gv_feat = utils.expand_tensor(gv_feat, cfg.DATA_LOADER.SEQ_PER_IMG)
        att_feats = utils.expand_tensor(att_feats, cfg.DATA_LOADER.SEQ_PER_IMG)
        att_mask = utils.expand_tensor(att_mask, cfg.DATA_LOADER.SEQ_PER_IMG)
        p_att_feats = utils.expand_tensor(p_att_feats, cfg.DATA_LOADER.SEQ_PER_IMG)

        batch_size = gv_feat.size(0)
        state = self.init_hidden(batch_size)

        outputs = Variable(torch.zeros(batch_size, seq.size(1), self.vocab_size).cuda())
        for t in range(seq.size(1)):
            if self.training and t >=1 and self.ss_prob > 0:
                prob = torch.empty(batch_size).cuda().uniform_(0, 1)
                mask = prob < self.ss_prob
                if mask.sum() == 0:
                    wt = seq[:,t].clone()
                else:
                    ind = mask.nonzero().view(-1)
                    wt = seq[:, t].data.clone()
                    prob_prev = torch.exp(outputs[:, t-1].detach())
                    wt.index_copy_(0, ind, torch.multinomial(prob_prev, 1).view(-1).index_select(0, ind))
            else:
                wt = seq[:,t].clone() # 0

            if t >= 1 and seq[:, t].max() == 0:
                break
            
            kwargs = self.make_kwargs(wt, gv_feat, att_feats, att_mask, p_att_feats, state)
            output, state = self.Forward(**kwargs)
            if self.dropout_lm is not None:
                output = self.dropout_lm(output)

            logit = self.logit(output)
            outputs[:, t] = logit

        return outputs

    def get_logprobs_state(self, **kwargs):
        output, state = self.Forward(**kwargs)
        logprobs = F.log_softmax(self.logit(output), dim=1)
        return logprobs, state

    def _expand_state(self, batch_size, beam_size, cur_beam_size, state, selected_beam):
        shape = [int(sh) for sh in state.shape]
        beam = selected_beam
        for _ in shape[2:]:
            beam = beam.unsqueeze(-1)
        beam = beam.unsqueeze(0)
        
        state = torch.gather(
            state.view(*([shape[0], batch_size, cur_beam_size] + shape[2:])), 2,
            beam.expand(*([shape[0], batch_size, beam_size] + shape[2:]))
        )
        state = state.view(*([shape[0], -1, ] + shape[2:]))
        return state

    # the beam search code is inspired by https://github.com/aimagelab/meshed-memory-transformer
    def decode_beam(self, **kwargs):
        gv_feat, att_feats, att_mask, p_att_feats = self.preprocess(**kwargs)
        
        beam_size = kwargs['BEAM_SIZE']
        batch_size = att_feats.size(0)
        seq_logprob = torch.zeros((batch_size, 1, 1)).cuda()
        log_probs = []
        selected_words = None
        seq_mask = torch.ones((batch_size, beam_size, 1)).cuda()

        state = self.init_hidden(batch_size)
        wt = Variable(torch.zeros(batch_size, dtype=torch.long).cuda())

        kwargs[cfg.PARAM.ATT_FEATS] = att_feats
        kwargs[cfg.PARAM.GLOBAL_FEAT] = gv_feat
        kwargs[cfg.PARAM.P_ATT_FEATS] = p_att_feats
        kwargs[cfg.PARAM.ATT_FEATS_MASK] = att_mask

        outputs = []
        for t in range(cfg.MODEL.SEQ_LEN):
            cur_beam_size = 1 if t == 0 else beam_size

            kwargs[cfg.PARAM.WT] = wt
            kwargs[cfg.PARAM.STATE] = state
            word_logprob, state = self.get_logprobs_state(**kwargs)
            word_logprob = word_logprob.view(batch_size, cur_beam_size, -1)
            candidate_logprob = seq_logprob + word_logprob

            # Mask sequence if it reaches EOS
            if t > 0:
                mask = (selected_words.view(batch_size, cur_beam_size) != 0).float().unsqueeze(-1)
                seq_mask = seq_mask * mask
                word_logprob = word_logprob * seq_mask.expand_as(word_logprob)
                old_seq_logprob = seq_logprob.expand_as(candidate_logprob).contiguous()
                old_seq_logprob[:, :, 1:] = -999
                candidate_logprob = seq_mask * candidate_logprob + old_seq_logprob * (1 - seq_mask)

            selected_idx, selected_logprob = self.select(batch_size, beam_size, t, candidate_logprob)
            selected_beam = selected_idx // candidate_logprob.shape[-1]
            selected_words = selected_idx % candidate_logprob.shape[-1]

            for s in range(len(state)):
                state[s] = self._expand_state(batch_size, beam_size, cur_beam_size, state[s], selected_beam)

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
            wt = selected_words.squeeze(-1)

            if t == 0:
                att_feats = utils.expand_tensor(att_feats, beam_size)
                gv_feat = utils.expand_tensor(gv_feat, beam_size)
                att_mask = utils.expand_tensor(att_mask, beam_size)
                p_att_feats = utils.expand_tensor(p_att_feats, beam_size)

                kwargs[cfg.PARAM.ATT_FEATS] = att_feats
                kwargs[cfg.PARAM.GLOBAL_FEAT] = gv_feat
                kwargs[cfg.PARAM.ATT_FEATS_MASK] = att_mask
                kwargs[cfg.PARAM.P_ATT_FEATS] = p_att_feats
 
        seq_logprob, sort_idxs = torch.sort(seq_logprob, 1, descending=True)
        outputs = torch.cat(outputs, -1)
        outputs = torch.gather(outputs, 1, sort_idxs.expand(batch_size, beam_size, cfg.MODEL.SEQ_LEN))
        log_probs = torch.cat(log_probs, -1)
        log_probs = torch.gather(log_probs, 1, sort_idxs.expand(batch_size, beam_size, cfg.MODEL.SEQ_LEN))

        outputs = outputs.contiguous()[:, 0]
        log_probs = log_probs.contiguous()[:, 0]

        return outputs, log_probs

    def decode(self, **kwargs):
        greedy_decode = kwargs['GREEDY_DECODE']
 
        gv_feat, att_feats, att_mask, p_att_feats = self.preprocess(**kwargs)
        batch_size = gv_feat.size(0)
        state = self.init_hidden(batch_size)

        sents = Variable(torch.zeros((batch_size, cfg.MODEL.SEQ_LEN), dtype=torch.long).cuda())
        logprobs = Variable(torch.zeros(batch_size, cfg.MODEL.SEQ_LEN).cuda())
        wt = Variable(torch.zeros(batch_size, dtype=torch.long).cuda())
        unfinished = wt.eq(wt)
        for t in range(cfg.MODEL.SEQ_LEN):
            kwargs = self.make_kwargs(wt, gv_feat, att_feats, att_mask, p_att_feats, state)
            logprobs_t, state = self.get_logprobs_state(**kwargs)
            
            if greedy_decode:
                logP_t, wt = torch.max(logprobs_t, 1)
            else:
                probs_t = torch.exp(logprobs_t)
                wt = torch.multinomial(probs_t, 1)
                logP_t = logprobs_t.gather(1, wt)
            wt = wt.view(-1).long()
            unfinished = unfinished * (wt > 0)
            wt = wt * unfinished.type_as(wt)
            sents[:,t] = wt
            logprobs[:,t] = logP_t.view(-1)

            if unfinished.sum() == 0:
                break
        return sents, logprobs

    def retrieval(self, get_logits=False, **kwargs):
        if get_logits:
            att_feats = kwargs['att_feats']
            sub_embeddings = kwargs['sub_embeddings']
            video_mask = kwargs['video_mask']
            seq_mask = kwargs['seq_mask']
        else:
            att_feats, sub_embeddings, video_mask, seq_mask = self.preprocess(retrieval=True, **kwargs)
        text_out, video_out = self._mean_pooling_for_similarity(sub_embeddings, att_feats, seq_mask, video_mask)
        text_out = F.normalize(text_out, dim=-1)
        video_out = F.normalize(video_out, dim=-1)
        retrieve_logits = torch.matmul(text_out, video_out.t())
        if get_logits:
            return retrieve_logits
        sim_loss1 = self.loss_fct(retrieve_logits)
        sim_loss2 = self.loss_fct(retrieve_logits.T)
        sim_loss = (sim_loss1 + sim_loss2) / 2
        return sim_loss

    def _mean_pooling_for_similarity(self, sequence_output, visual_output, attention_mask, video_mask):
        attention_mask_un = attention_mask.to(dtype=torch.float).unsqueeze(-1)
        attention_mask_un[:, 0, :] = 0.
        sequence_output = sequence_output * attention_mask_un
        text_out = torch.sum(sequence_output, dim=1) / torch.sum(attention_mask_un, dim=1, dtype=torch.float)

        video_mask_un = video_mask.to(dtype=torch.float).unsqueeze(-1)
        visual_output = visual_output * video_mask_un
        video_mask_un_sum = torch.sum(video_mask_un, dim=1, dtype=torch.float)
        video_mask_un_sum[video_mask_un_sum == 0.] = 1.
        video_out = torch.sum(visual_output, dim=1) / video_mask_un_sum

        return text_out, video_out