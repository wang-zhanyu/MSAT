import os
import sys
from pprint import pprint
import random
import time
import logging
import argparse
import torch
import queue
import torch.distributed as dist
import losses
import models
import lib.utils as utils
from lib.utils import AverageMeter
from optimizer.optimizer import Optimizer
from evaluation.evaler import Evaler
from scorer.scorer import Scorer
from lib.config import cfg, cfg_from_file
from datasets.mimic_dataset import MimicDataset
from datasets.data_loader import load_train
import numpy as np

time_cn = []


class Trainer(object):
    def __init__(self, args):
        super(Trainer, self).__init__()
        self.args = args
        if cfg.SEED > 0:
            random.seed(cfg.SEED)
            torch.manual_seed(cfg.SEED)
            torch.cuda.manual_seed_all(cfg.SEED)

        self.num_gpus = torch.cuda.device_count()
        self.distributed = self.num_gpus > 1
        if self.distributed:
            torch.cuda.set_device(args.local_rank)
            torch.distributed.init_process_group(backend="nccl")
        self.device = torch.device("cuda")

        self.train_dataset = MimicDataset(
            json_path=cfg.DATA_LOADER.TRAIN_JSON_PATH,
            input_seq=cfg.DATA_LOADER.INPUT_SEQ_PATH,
            target_seq=cfg.DATA_LOADER.TARGET_SEQ_PATH,
            seq_per_img=cfg.DATA_LOADER.SEQ_PER_IMG,
            max_feat_num=cfg.DATA_LOADER.MAX_FEAT,
            training=True,
            tsn=cfg.DATA_LOADER.TSN,
            gv_load_method=cfg.DATA_LOADER.GV_LOAD
        )

        self.test_evaler = Evaler(
            json_path=cfg.DATA_LOADER.TEST_JSON_PATH,
            vocab=cfg.INFERENCE.VOCAB,
            debug=self.args.debug
        )

        self.rl_stage = False
        self.setup_logging()
        self.setup_network()
        self.scorer = Scorer()

    def setup_logging(self):
        self.logger = logging.getLogger(cfg.LOGGER_NAME)
        self.logger.setLevel(logging.INFO)
        if self.distributed and dist.get_rank() > 0:
            return

        ch = logging.StreamHandler(stream=sys.stdout)
        ch.setLevel(logging.INFO)
        formatter = logging.Formatter("[%(levelname)s: %(asctime)s] %(message)s")
        ch.setFormatter(formatter)
        self.logger.addHandler(ch)

        if not os.path.exists(cfg.ROOT_DIR):
            os.makedirs(cfg.ROOT_DIR)

        fh = logging.FileHandler(os.path.join(cfg.ROOT_DIR, cfg.LOGGER_NAME + '.txt'))
        fh.setLevel(logging.INFO)
        fh.setFormatter(formatter)
        self.logger.addHandler(fh)

        self.logger.info('Training with config:')
        self.logger.info(pprint.pformat(cfg))

    def setup_network(self):
        model = models.create(cfg.MODEL.TYPE)
        if self.distributed:
            self.model = torch.nn.parallel.DistributedDataParallel(
                model.to(self.device),
                device_ids=[self.args.local_rank],
                output_device=self.args.local_rank,
                broadcast_buffers=False,
                find_unused_parameters=True
            )
        else:
            self.model = torch.nn.DataParallel(model).cuda()

        if self.args.resume > 0:
            self.model.load_state_dict(
                torch.load(self.args.resume,
                           map_location=lambda storage, loc: storage)
            )

        self.optim = Optimizer(self.model)
        self.xe_criterion = losses.create(cfg.LOSSES.XE_TYPE).cuda()
        self.rl_criterion = losses.create(cfg.LOSSES.RL_TYPE).cuda()
        self.bce_criterion = losses.create(cfg.LOSSES.BCE_TYPE).cuda()

    def setup_loader(self, epoch):
        self.training_loader = load_train(
            self.distributed, epoch, self.train_dataset, self.num_gpus)

    def eval(self, epoch):
        if (epoch + 1) % cfg.SOLVER.TEST_INTERVAL != 0:
            return None
        if self.distributed and dist.get_rank() > 0:
            return None

        scores = self.test_evaler(self.model, 'test_' + str(epoch + 1))
        self.logger.info('######## Epoch (TEST)' + str(epoch + 1) + ' ########')
        self.logger.info(str(scores))

        val = 0
        for score_type, weight in zip(cfg.SCORER.TYPES, cfg.SCORER.WEIGHTS):
            val -= scores[score_type] * weight
        return val, scores

    def snapshot_path(self, name, epoch):
        snapshot_folder = os.path.join(cfg.ROOT_DIR, 'snapshot')
        return os.path.join(snapshot_folder, name + "_" + str(epoch) + ".pth")

    def save_model(self, epoch):
        if (epoch + 1) % cfg.SOLVER.SNAPSHOT_ITERS != 0:
            return
        if self.distributed and dist.get_rank() > 0:
            return
        snapshot_folder = os.path.join(cfg.ROOT_DIR, 'snapshot')
        if not os.path.exists(snapshot_folder):
            os.mkdir(snapshot_folder)
        torch.save(self.model.state_dict(), self.snapshot_path("model", epoch + 1))
        if save_file_queue.full():
            remove_file = save_file_queue.get_nowait()
            os.remove(remove_file)
        save_file_queue.put_nowait(self.snapshot_path("model", epoch + 1))

    def make_kwargs(self, indices, input_seq, target_seq, gv_feat, att_feats, att_mask, mlc_label):
        seq_mask = (input_seq > 0).type(torch.cuda.LongTensor)
        seq_mask[:, 0] += 1
        seq_mask_sum = seq_mask.sum(-1)
        max_len = int(seq_mask_sum.max())
        input_seq = input_seq[:, 0:max_len].contiguous()
        target_seq = target_seq[:, 0:max_len].contiguous()

        kwargs = {
            cfg.PARAM.INDICES: indices,
            cfg.PARAM.INPUT_SENT: input_seq,
            cfg.PARAM.TARGET_SENT: target_seq,
            cfg.PARAM.GLOBAL_FEAT: gv_feat,
            cfg.PARAM.ATT_FEATS: att_feats,
            cfg.PARAM.ATT_FEATS_MASK: att_mask,
            cfg.PARAM.MLC_LABEL: mlc_label
        }
        return kwargs

    def scheduled_sampling(self, epoch):
        if epoch > cfg.TRAIN.SCHEDULED_SAMPLING.START:
            frac = (epoch - cfg.TRAIN.SCHEDULED_SAMPLING.START) // cfg.TRAIN.SCHEDULED_SAMPLING.INC_EVERY
            ss_prob = min(cfg.TRAIN.SCHEDULED_SAMPLING.INC_PROB * frac, cfg.TRAIN.SCHEDULED_SAMPLING.MAX_PROB)
            self.model.module.ss_prob = ss_prob

    def display(self, epoch, iteration, data_time, batch_time, losses, loss_info, totol_iter):
        global time_cn
        if iteration % cfg.SOLVER.DISPLAY != 0:
            return
        if self.distributed and dist.get_rank() > 0:
            return
        info_str = ' (DataTime/BatchTime: {:.3}/{:.3}) losses = {:.5}'.format(data_time.avg, batch_time.avg, losses.avg)
        time_cn.append(batch_time.avg)
        if len(time_cn) == 25:
            print(f"mean_time:{np.mean(time_cn)}")
        else:
            print(len(time_cn))
        self.logger.info(
            f"Epoch:{epoch}/{cfg.SOLVER.MAX_EPOCH} [{iteration % totol_iter}/{totol_iter}] Totol_iter:{iteration}{info_str} lr={str(self.optim.get_lr())}")
        for name in sorted(loss_info):
            self.logger.info('  ' + name + ' = ' + str(loss_info[name]))
        data_time.reset()
        batch_time.reset()
        losses.reset()

    def forward(self, kwargs):
        if self.rl_stage == False:
            logit, mlc_logit = self.model(**kwargs)
            loss, loss_info = self.xe_criterion(logit, kwargs[cfg.PARAM.TARGET_SENT])
            if mlc_logit is not None:
                mlc_loss, mlc_loss_info = self.bce_criterion(mlc_logit, kwargs[cfg.PARAM.MLC_LABEL])
                loss += cfg.MODEL.MLC_WEIGHT * mlc_loss
                loss_info.update(mlc_loss_info)
        else:
            ids = kwargs[cfg.PARAM.INDICES]
            gv_feat = kwargs[cfg.PARAM.GLOBAL_FEAT]
            att_feats = kwargs[cfg.PARAM.ATT_FEATS]
            att_mask = kwargs[cfg.PARAM.ATT_FEATS_MASK]

            # max
            kwargs['BEAM_SIZE'] = 1
            kwargs['GREEDY_DECODE'] = True
            kwargs[cfg.PARAM.GLOBAL_FEAT] = gv_feat
            kwargs[cfg.PARAM.ATT_FEATS] = att_feats
            kwargs[cfg.PARAM.ATT_FEATS_MASK] = att_mask
            kwargs[cfg.PARAM.SUB_SEQ] = sub_seq
            kwargs[cfg.PARAM.TOKEN_TYPE_IDS] = token_type_ids

            self.model.eval()
            with torch.no_grad():
                seq_max, logP_max = self.model.module.decode(**kwargs)
            self.model.train()
            rewards_max, rewards_info_max = self.scorer(ids, seq_max.data.cpu().numpy().tolist())
            rewards_max = utils.expand_numpy(rewards_max)

            ids = utils.expand_numpy(ids)
            gv_feat = utils.expand_tensor(gv_feat, cfg.DATA_LOADER.SEQ_PER_IMG)
            att_feats = utils.expand_tensor(att_feats, cfg.DATA_LOADER.SEQ_PER_IMG)
            att_mask = utils.expand_tensor(att_mask, cfg.DATA_LOADER.SEQ_PER_IMG)
            sub_seq = utils.expand_tensor(sub_seq, cfg.DATA_LOADER.SEQ_PER_IMG)
            token_type_ids = utils.expand_tensor(token_type_ids, cfg.DATA_LOADER.SEQ_PER_IMG)
            # sample
            kwargs['BEAM_SIZE'] = 1
            kwargs['GREEDY_DECODE'] = False
            kwargs[cfg.PARAM.GLOBAL_FEAT] = gv_feat
            kwargs[cfg.PARAM.ATT_FEATS] = att_feats
            kwargs[cfg.PARAM.ATT_FEATS_MASK] = att_mask
            kwargs[cfg.PARAM.SUB_SEQ] = sub_seq
            kwargs[cfg.PARAM.TOKEN_TYPE_IDS] = token_type_ids

            seq_sample, logP_sample = self.model.module.decode(**kwargs)
            rewards_sample, rewards_info_sample = self.scorer(ids, seq_sample.data.cpu().numpy().tolist())

            rewards = rewards_sample - rewards_max
            rewards = torch.from_numpy(rewards).float().cuda()
            loss = self.rl_criterion(seq_sample, logP_sample, rewards)

            loss_info = {}
            for key in rewards_info_sample:
                loss_info[key + '_sample'] = rewards_info_sample[key]
            for key in rewards_info_max:
                loss_info[key + '_max'] = rewards_info_max[key]

        return loss, loss_info

    def train(self):
        self.model.train()
        self.optim.zero_grad()
        last_score = 0
        iteration = 0
        best_scores, best_epoch = None, None

        for epoch in range(cfg.SOLVER.MAX_EPOCH):
            if epoch == cfg.TRAIN.REINFORCEMENT.START:
                self.rl_stage = True
            self.setup_loader(epoch)

            start = time.time()
            data_time = AverageMeter()
            batch_time = AverageMeter()
            losses = AverageMeter()
            for step, (indices, input_seq, target_seq, gv_feat, att_feats, att_mask,
                       mlc_label) in enumerate(self.training_loader):
                data_time.update(time.time() - start)
                input_seq, target_seq, gv_feat = input_seq.cuda(), target_seq.cuda(), gv_feat.cuda()
                att_feats, att_mask, mlc_label = att_feats.cuda(), mlc_label.cuda(), att_mask.cuda()

                kwargs = self.make_kwargs(indices, input_seq, target_seq, gv_feat, att_feats, att_mask, mlc_label)
                loss, loss_info = self.forward(kwargs)
                loss.backward()
                utils.clip_gradient(self.optim.optimizer, self.model,
                                    cfg.SOLVER.GRAD_CLIP_TYPE, cfg.SOLVER.GRAD_CLIP)
                self.optim.step()
                self.optim.zero_grad()
                self.optim.scheduler_step('Iter')

                batch_time.update(time.time() - start)
                start = time.time()
                losses.update(loss.item())
                self.display(epoch, iteration, data_time, batch_time, losses, loss_info, len(self.training_loader))
                iteration += 1

                if self.distributed:
                    dist.barrier()

                if self.args.debug and step > 5:
                    break

            if self.distributed:
                if dist.get_rank() == 0:
                    val, scores = self.eval(epoch)
                    CIDEr = scores['CIDEr']
                    if CIDEr > last_score:
                        self.save_model(epoch)
                        last_score = CIDEr
                        best_scores = scores
                        best_epoch = epoch
                        self.logger.info(f"Best model saved in epoch{epoch}")
                    else:
                        self.logger.info(f"BEST model is epoch{best_epoch} {best_scores}")
                else:
                    val = None
            else:
                val, scores = self.eval(epoch)
                CIDEr = scores['CIDEr']
                if CIDEr > last_score:
                    self.save_model(epoch)
                    last_score = CIDEr
                    best_scores = scores
                    best_epoch = epoch
                    self.logger.info(f"Best model saved in epoch{epoch}")
                else:
                    self.logger.info(f"BEST model is epoch{best_epoch} {best_scores}")
            self.optim.scheduler_step('Epoch', val)
            self.scheduled_sampling(epoch)

            if self.distributed:
                dist.barrier()


def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description='Medical Report Generation')
    parser.add_argument('--folder', dest='folder', type=str, default='experiments/V1')
    parser.add_argument("--local_rank", type=int, default=0)
    parser.add_argument("--resume", type=int, default=-1)
    parser.add_argument("--top_k", type=int, default=5)
    parser.add_argument("--debug", action='store_true', default=False)
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    pprint(args)
    save_file_queue = queue.Queue(maxsize=args.top_k)
    if args.folder is not None:
        cfg_from_file(os.path.join(args.folder, 'config.yml'))
    cfg.ROOT_DIR = args.folder
    trainer = Trainer(args)
    trainer.train()
