import torch
from lib.config import cfg
import samplers.distributed
import numpy as np
from datasets.mimic_dataset import MimicDataset


def sample_collate(batch):
    image_id, input_seq, target_seq, gv_feat, att_feats, mlc_label = zip(*batch)
    image_id = np.stack(image_id, axis=0).reshape(-1)
    input_seq = torch.cat([torch.from_numpy(b) for b in input_seq], 0).long()
    target_seq = torch.cat([torch.from_numpy(b) for b in target_seq], 0).long()
    gv_feat = torch.cat([torch.from_numpy(b) for b in gv_feat], 0)
    if cfg.MODEL.USE_MLC:
        mlc_seq = torch.cat([torch.from_numpy(b) for b in mlc_label], 0)
    else:
        mlc_seq = None

    atts_num = [x.shape[0] for x in att_feats]
    max_att_num = np.max(atts_num)

    feat_arr = []
    mask_arr = []
    for i, num in enumerate(atts_num):
        tmp_feat = np.zeros((1, max_att_num, att_feats[i].shape[1]), dtype=np.float32)
        tmp_feat[:, 0:att_feats[i].shape[0], :] = att_feats[i]
        feat_arr.append(torch.from_numpy(tmp_feat))

        tmp_mask = np.zeros((1, max_att_num), dtype=np.float32)
        tmp_mask[:, 0:num] = 1
        mask_arr.append(torch.from_numpy(tmp_mask))

    att_feats = torch.cat(feat_arr, 0)
    att_mask = torch.cat(mask_arr, 0)

    return image_id, input_seq, target_seq, gv_feat, att_feats, att_mask, mlc_seq


def load_train(distributed, epoch, coco_set, n_gpu):
    sampler = samplers.distributed.DistributedSampler(coco_set, epoch=epoch) \
        if distributed else None
    shuffle = cfg.DATA_LOADER.SHUFFLE if sampler is None else False
    
    loader = torch.utils.data.DataLoader(
        coco_set, 
        batch_size=cfg.TRAIN.BATCH_SIZE // n_gpu,
        shuffle=shuffle,
        num_workers=cfg.DATA_LOADER.NUM_WORKERS,
        drop_last=cfg.DATA_LOADER.DROP_LAST,
        pin_memory=cfg.DATA_LOADER.PIN_MEMORY,
        sampler=sampler,
        collate_fn=sample_collate
    )
    return loader


def load_val(json_path):
    coco_set = MimicDataset(
        json_path=json_path,
        input_seq=cfg.DATA_LOADER.INPUT_SEQ_PATH,
        target_seq=cfg.DATA_LOADER.TARGET_SEQ_PATH,
        seq_per_img=cfg.DATA_LOADER.SEQ_PER_IMG,
        max_feat_num=cfg.DATA_LOADER.MAX_FEAT,
        training=False,
        tsn=cfg.DATA_LOADER.TSN,
        gv_load_method=cfg.DATA_LOADER.GV_LOAD
    )

    loader = torch.utils.data.DataLoader(
        coco_set,
        batch_size=cfg.TEST.BATCH_SIZE,
        shuffle=False,
        num_workers=cfg.DATA_LOADER.NUM_WORKERS,
        drop_last=False,
        pin_memory=cfg.DATA_LOADER.PIN_MEMORY,
        collate_fn=sample_collate
    )
    return loader

