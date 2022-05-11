import random
import numpy as np
import json
import torch.utils.data as data
import pickle
from lib.config import cfg


class MimicDataset(data.Dataset):
    def __init__(
        self, 
        json_path,
        input_seq,
        target_seq,
        seq_per_img,
        max_feat_num,
        training=True,
        tsn=False,
        gv_load_method='none'
    ):
        self.training = training
        self.max_feat_num = max_feat_num
        self.seq_per_img = seq_per_img
        self.gv_load_method = gv_load_method
        self.tsn = tsn
        self.id2path = json.load(open(json_path, 'r'))
        self.image_ids = list(self.id2path.keys())

        if input_seq is not None and target_seq is not None:
            self.input_seq = pickle.load(open(input_seq, 'rb'), encoding='bytes')
            self.target_seq = pickle.load(open(target_seq, 'rb'), encoding='bytes')
            self.seq_len = len(self.input_seq[self.image_ids[0]][0, :])
        else:
            self.seq_len = -1
            self.input_seq = None
            self.target_seq = None

        if cfg.MODEL.USE_MLC:
            self.mlc_label = pickle.load(open(cfg.DATA_LOADER.MLC_LABEL_PATH, 'rb'))

    def set_seq_per_img(self, seq_per_img):
        self.seq_per_img = seq_per_img

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, index):
        image_id = self.image_ids[index]
        att_feats = np.load(self.id2path[image_id])['feature']
        att_feats = np.array(att_feats).astype('float32')

        if self.gv_load_method == 'first':
            gv_feat = np.zeros((1, att_feats.shape[-1]))
            gv_feat[0] = att_feats[0]
            att_feats = att_feats[1:]
        elif self.gv_load_method == 'nofirst':
            gv_feat = np.zeros((1, 1))
            att_feats = att_feats[1:]
        else:
            gv_feat = np.zeros((1, 1))
        gv_feat = gv_feat.astype('float32')

        if 0 < self.max_feat_num < att_feats.shape[0]:
            indices = np.linspace(0, len(att_feats), self.max_feat_num, endpoint=False).astype(np.int).tolist()
            att_feats = att_feats[indices]

        if cfg.MODEL.USE_MLC:
            if image_id in self.mlc_label.keys():
                mlc_label = self.mlc_label[image_id]
            else:
                print(image_id)
                mlc_label = np.ones((1, 768), dtype='int')
        else:
            mlc_label = None

        if self.seq_len < 0:
            return image_id, gv_feat, att_feats

        input_seq = np.zeros((self.seq_per_img, self.seq_len), dtype='int')
        target_seq = np.zeros((self.seq_per_img, self.seq_len), dtype='int')
           
        n = len(self.input_seq[image_id])   
        if n >= self.seq_per_img:
            sid = 0
            ixs = random.sample(range(n), self.seq_per_img)                
        else:
            sid = n
            ixs = random.sample(range(n), self.seq_per_img - n)
            input_seq[0:n, :] = self.input_seq[image_id]
            target_seq[0:n, :] = self.target_seq[image_id]
           
        for i, ix in enumerate(ixs):
            input_seq[sid + i] = self.input_seq[image_id][ix, :]
            target_seq[sid + i] = self.target_seq[image_id][ix, :]
        return image_id, input_seq, target_seq, gv_feat, att_feats, mlc_label
