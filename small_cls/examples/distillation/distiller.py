# coding=utf-8
# Copyright 2019-present, the HuggingFace Inc. team and Facebook, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" The distiller to distil DistilBERT
    adapted in part from Facebook, Inc XLM model (https://github.com/facebookresearch/XLM)
"""
import os
import math
import psutil
from tensorboardX import SummaryWriter
from tqdm import trange, tqdm
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from pytorch_transformers import AdamW, WarmupLinearSchedule

from utils import logger
from dataset import Dataset

class Distiller:
    def __init__(self,
                 params: dict,
                 dataloader: Dataset,
                 token_probs: torch.tensor,
                 student: nn.Module,
                 teacher: nn.Module):
        logger.info('Initializing Distiller')
        self.params = params
        self.dump_path = params.dump_path
        self.multi_gpu = params.multi_gpu
        self.fp16 = params.fp16

        self.student = student
        self.teacher = teacher

        self.dataloader = dataloader
        if self.params.n_gpu > 1:
            self.dataloader.split()
        self.get_iterator(seed=params.seed)

        self.temperature = params.temperature
        assert self.temperature > 0.

        self.alpha_ce = params.alpha_ce
        self.alpha_mlm = params.alpha_mlm
        self.alpha_mse = params.alpha_mse
        assert self.alpha_ce >= 0.
        assert self.alpha_mlm >= 0.
        assert self.alpha_mse >= 0.
        assert self.alpha_ce + self.alpha_mlm + self.alpha_mse > 0.

        self.mlm_mask_prop = params.mlm_mask_prop
        assert 0.0 <= self.mlm_mask_prop <= 1.0
        assert params.word_mask + params.word_keep + params.word_rand == 1.0
        self.pred_probs = torch.FloatTensor([params.word_mask, params.word_keep, params.word_rand])
        self.pred_probs = self.pred_probs.to(f'cuda:{params.local_rank}') if params.n_gpu > 0 else self.pred_probs
        self.token_probs = token_probs.to(f'cuda:{params.local_rank}') if params.n_gpu > 0 else token_probs
        if self.fp16:
            self.pred_probs = self.pred_probs.half()
            self.token_probs = self.token_probs.half()

        self.epoch = 0
        self.n_iter = 0
        self.n_total_iter = 0
        self.n_sequences_epoch = 0
        self.total_loss_epoch = 0
        self.last_loss = 0
        self.last_loss_ce = 0
        self.last_loss_mlm = 0
        self.last_loss_mse = 0

        self.ce_loss_fct = nn.KLDivLoss(reduction='batchmean')
        self.mlm_loss_fct = nn.CrossEntropyLoss(ignore_index=-1)
        self.mse_loss_fct = nn.MSELoss(reduction='sum')

        logger.info('--- Initializing model optimizer')
        assert params.gradient_accumulation_steps >= 1
        self.num_steps_epoch = int(len(self.dataloader) / params.batch_size) + 1
        num_train_optimization_steps = int(self.num_steps_epoch / params.gradient_accumulation_steps * params.n_epoch) + 1
        warmup_steps = math.ceil(num_train_optimization_steps * params.warmup_prop)

        no_decay = ['bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in student.named_parameters() if not any(nd in n for nd in no_decay) and p.requires_grad], 'weight_decay': params.weight_decay},
            {'params': [p for n, p in student.named_parameters() if any(nd in n for nd in no_decay) and p.requires_grad], 'weight_decay': 0.0}
        ]
        logger.info("------ Number of trainable parameters (student): %i" % sum([p.numel() for p in self.student.parameters() if p.requires_grad]))
        logger.info("------ Number of parameters (student): %i" % sum([p.numel() for p in self.student.parameters()]))
        self.optimizer = AdamW(optimizer_grouped_parameters,
                               lr=params.learning_rate,
                               eps=params.adam_epsilon,
                               betas=(0.9, 0.98))
        self.scheduler = WarmupLinearSchedule(self.optimizer,
                                              warmup_steps=warmup_steps,
                                              t_total=num_train_optimization_steps)

        if self.fp16:
            try:
                from apex import amp
            except ImportError:
                raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")
            logger.info(f"Using fp16 training: {self.params.fp16_opt_level} level")
            self.student, self.optimizer = amp.initialize(self.student,
                                                          self.optimizer,
                                                          opt_level=self.params.fp16_opt_level)
            self.teacher = self.teacher.half()

        if self.multi_gpu:
            if self.fp16:
                from apex.parallel import DistributedDataParallel
                logger.info("Using apex.parallel.DistributedDataParallel for distributed training.")
                self.student = DistributedDataParallel(self.student)
            else:
                from torch.nn.parallel import DistributedDataParallel
                logger.info("Using nn.parallel.DistributedDataParallel for distributed training.")
                self.student = DistributedDataParallel(self.student,
                                                       device_ids=[params.local_rank],
                                                       output_device=params.local_rank)

        self.is_master = params.is_master
        if self.is_master:
            logger.info('--- Initializing Tensorboard')
            self.tensorboard = SummaryWriter(log_dir=os.path.join(self.dump_path, 'log', 'train'))
            self.tensorboard.add_text(tag='config', text_string=str(self.params), global_step=0)

    def get_iterator(self,
                     seed: int = None):
        """
        Initialize the data iterator.
        Each process has its own data iterator (iterating on his own random portion of the dataset).

        Input:
        ------
            seed: `int` - The random seed.
        """
        logger.info('--- Initializing Data Iterator')
        self.data_iterator = self.dataloader.get_iterator(seed=seed)

    def get_batch(self):
        """
        Call the data iterator to output a new batch.
        If the data iterator went through the whole dataset, create a new iterator.
        """
        assert hasattr(self, 'data_iterator')
        try:
            x = next(self.data_iterator)
        except StopIteration:
            logger.warning('--- Went through the whole dataset. Creating new data iterator.')
            self.data_iterator = self.dataloader.get_iterator()
            x = next(self.data_iterator)
        return x

    def prepare_batch(self,
                      batch):
        """
        Prepare the batch: from the token_ids and the lenghts, compute the attention mask and the masked label for MLM.

        Input:
        ------
            batch: `Tuple`
                token_ids: `torch.tensor(bs, seq_length)` - The token ids for each of the sequence. It is padded.
                lengths: `torch.tensor(bs)` - The lengths of each of the sequences in the batch.

        Output:
        -------
            token_ids: `torch.tensor(bs, seq_length)` - The token ids after the modifications for MLM.
            attn_mask: `torch.tensor(bs, seq_length)` - The attention mask for the self-attention.
            mlm_labels: `torch.tensor(bs, seq_length)` - The masked languge modeling labels. There is a -1 where there is nothing to predict.
        """
        token_ids, lengths = batch
        token_ids, lengths = self.round_batch(x=token_ids, lengths=lengths)
        assert token_ids.size(0) == lengths.size(0)

        attn_mask = (torch.arange(token_ids.size(1), dtype=torch.long, device=lengths.device) < lengths[:, None])

        bs, max_seq_len = token_ids.size()
        mlm_labels = token_ids.new(token_ids.size()).copy_(token_ids)

        x_prob = self.token_probs[token_ids.flatten()]
        n_tgt = math.ceil(self.mlm_mask_prop * lengths.sum().item())
        tgt_ids = torch.multinomial(x_prob / x_prob.sum(), n_tgt, replacement=False)
        pred_mask = torch.zeros(bs * max_seq_len, dtype=torch.bool, device=token_ids.device) # previously `dtype=torch.uint8`, cf pytorch 1.2.0 compatibility
        pred_mask[tgt_ids] = 1
        pred_mask = pred_mask.view(bs, max_seq_len)

        pred_mask[token_ids == self.params.special_tok_ids['pad_token']] = 0

        # mask a number of words == 0 [8] (faster with fp16)
        if self.fp16:
            n1 = pred_mask.sum().item()
            if n1 > 8:
                pred_mask = pred_mask.view(-1)
                n2 = max(n1 % 8, 8 * (n1 // 8))
                if n2 != n1:
                    pred_mask[torch.nonzero(pred_mask).view(-1)[:n1-n2]] = 0
                pred_mask = pred_mask.view(bs, max_seq_len)
                assert pred_mask.sum().item() % 8 == 0, pred_mask.sum().item()

        _token_ids_real = token_ids[pred_mask]
        _token_ids_rand = _token_ids_real.clone().random_(self.params.vocab_size)
        _token_ids_mask = _token_ids_real.clone().fill_(self.params.special_tok_ids['mask_token'])
        probs = torch.multinomial(self.pred_probs, len(_token_ids_real), replacement=True)
        _token_ids = _token_ids_mask * (probs == 0).long() + _token_ids_real * (probs == 1).long() + _token_ids_rand * (probs == 2).long()
        token_ids = token_ids.masked_scatter(pred_mask, _token_ids)

        mlm_labels[~pred_mask] = -1 # previously `mlm_labels[1-pred_mask] = -1`, cf pytorch 1.2.0 compatibility

        return token_ids, attn_mask, mlm_labels

    def round_batch(self,
                    x: torch.tensor,
                    lengths: torch.tensor):
        """
        For float16 only.
        Sub-sample sentences in a batch, and add padding, so that each dimension is a multiple of 8.

        Input:
        ------
            x: `torch.tensor(bs, seq_length)` - The token ids.
            lengths: `torch.tensor(bs, seq_length)` - The lengths of each of the sequence in the batch.

        Output:
        -------
            x:  `torch.tensor(new_bs, new_seq_length)` - The updated token ids.
            lengths: `torch.tensor(new_bs, new_seq_length)` - The updated lengths.
        """
        if not self.fp16 or len(lengths) < 8:
            return x, lengths

        # number of sentences == 0 [8]
        bs1 = len(lengths)
        bs2 = 8 * (bs1 // 8)
        assert bs2 > 0 and bs2 % 8 == 0
        if bs1 != bs2:
            idx = torch.randperm(bs1)[:bs2]
            lengths = lengths[idx]
            slen = lengths.max().item()
            x = x[idx, :slen]
        else:
            idx = None

        # sequence length == 0 [8]
        ml1 = x.size(1)
        if ml1 % 8 != 0:
            pad = 8 - (ml1 % 8)
            ml2 = ml1 + pad
            pad_id = self.params.special_tok_ids['pad_token']
            padding_tensor = torch.zeros(bs2, pad, dtype=torch.long, device=x.device).fill_(pad_id)
            x = torch.cat([x, padding_tensor], 1)
            assert x.size() == (bs2, ml2)

        assert x.size(0) % 8 == 0
        assert x.size(1) % 8 == 0
        return x, lengths

    def train(self):
        """
        The real training loop.
        """
        if self.is_master: logger.info('Starting training')
        self.student.train()
        self.teacher.eval()

        for _ in range(self.params.n_epoch):
            if self.is_master: logger.info(f'--- Starting epoch {self.epoch}/{self.params.n_epoch-1}')

            iter_bar = trange(self.num_steps_epoch, desc="-Iter", disable=self.params.local_rank not in [-1, 0])
            for __ in range(self.num_steps_epoch):
                batch = self.get_batch()
                if self.params.n_gpu > 0:
                    batch = tuple(t.to(f'cuda:{self.params.local_rank}') for t in batch)
                token_ids, attn_mask, mlm_labels = self.prepare_batch(batch=batch)

                self.step(input_ids=token_ids, attention_mask=attn_mask, mlm_labels=mlm_labels)

                iter_bar.update()
                iter_bar.set_postfix({'Last_loss': f'{self.last_loss:.2f}',
                                      'Avg_cum_loss': f'{self.total_loss_epoch/self.n_iter:.2f}'})
            iter_bar.close()

            if self.is_master: logger.info(f'--- Ending epoch {self.epoch}/{self.params.n_epoch-1}')
            self.end_epoch()

        if self.is_master:
            logger.info(f'Save very last checkpoint as `pytorch_model.bin`.')
            self.save_checkpoint(checkpoint_name=f'pytorch_model.bin')
            logger.info('Training is finished')

    def step(self,
             input_ids: torch.tensor,
             attention_mask: torch.tensor,
             mlm_labels: torch.tensor):
        """
        One optimization step: forward of student AND teacher, backward on the loss (for gradient accumulation),
        and possibly a parameter update (depending on the gradient accumulation).

        Input:
        ------
        input_ids: `torch.tensor(bs, seq_length)` - The token ids.
        attention_mask: `torch.tensor(bs, seq_length)` - The attention mask for self attention.
        mlm_labels: `torch.tensor(bs, seq_length)` - The masked language modeling labels.
        """
        s_logits = self.student(input_ids=input_ids, attention_mask=attention_mask)[0]     # (bs, seq_length, voc_size)
        with torch.no_grad():
            t_logits = self.teacher(input_ids=input_ids, attention_mask=attention_mask)[0] # (bs, seq_length, voc_size)
        assert s_logits.size() == t_logits.size()

        #https://github.com/peterliht/knowledge-distillation-pytorch/blob/master/model/net.py#L100
        #https://github.com/peterliht/knowledge-distillation-pytorch/issues/2
        if self.params.restrict_ce_to_mask:
            mask = (mlm_labels>-1).unsqueeze(-1).expand_as(s_logits)   # (bs, seq_lenth, voc_size)
        else:
            mask = attention_mask.unsqueeze(-1).expand_as(s_logits)    # (bs, seq_lenth, voc_size)
        s_logits_slct = torch.masked_select(s_logits, mask)            # (bs * seq_length * voc_size) modulo the 1s in mask
        s_logits_slct = s_logits_slct.view(-1, s_logits.size(-1))      # (bs * seq_length, voc_size) modulo the 1s in mask
        t_logits_slct = torch.masked_select(t_logits, mask)            # (bs * seq_length * voc_size) modulo the 1s in mask
        t_logits_slct = t_logits_slct.view(-1, s_logits.size(-1))      # (bs * seq_length, voc_size) modulo the 1s in mask
        assert t_logits_slct.size() == s_logits_slct.size()

        loss_ce = self.ce_loss_fct(F.log_softmax(s_logits_slct/self.temperature, dim=-1),
                                   F.softmax(t_logits_slct/self.temperature, dim=-1)) * (self.temperature)**2
        loss = self.alpha_ce*loss_ce
        if self.alpha_mlm > 0.:
            loss_mlm = self.mlm_loss_fct(s_logits.view(-1, s_logits.size(-1)), mlm_labels.view(-1))
            loss += self.alpha_mlm * loss_mlm
        if self.alpha_mse > 0.:
            loss_mse = self.mse_loss_fct(s_logits_slct, t_logits_slct)/s_logits_slct.size(0) # Reproducing batchmean reduction
            loss += self.alpha_mse * loss_mse

        self.total_loss_epoch += loss.item()
        self.last_loss = loss.item()
        self.last_loss_ce = loss_ce.item()
        if self.alpha_mlm > 0.:
            self.last_loss_mlm = loss_mlm.item()
        if self.alpha_mse > 0.:
            self.last_loss_mse = loss_mse.item()

        self.optimize(loss)

        self.n_sequences_epoch += input_ids.size(0)

    def optimize(self,
                 loss):
        """
        Normalization on the loss (gradient accumulation or distributed training), followed by
        backward pass on the loss, possibly followed by a parameter update (depending on the gradient accumulation).
        Also update the metrics for tensorboard.
        """
        # Check for NaN
        if (loss != loss).data.any():
            logger.error('NaN detected')
            exit()

        if self.multi_gpu:
            loss = loss.mean()
        if self.params.gradient_accumulation_steps > 1:
            loss = loss / self.params.gradient_accumulation_steps

        if self.fp16:
            from apex import amp
            with amp.scale_loss(loss, self.optimizer) as scaled_loss:
                scaled_loss.backward()
        else:
            loss.backward()

        self.iter()
        if self.n_iter % self.params.gradient_accumulation_steps == 0:
            if self.fp16:
                torch.nn.utils.clip_grad_norm_(amp.master_params(self.optimizer), self.params.max_grad_norm)
            else:
                torch.nn.utils.clip_grad_norm_(self.student.parameters(), self.params.max_grad_norm)
            self.optimizer.step()
            self.optimizer.zero_grad()
            self.scheduler.step()

    def iter(self):
        """
        Update global counts, write to tensorboard and save checkpoint.
        """
        self.n_iter += 1
        self.n_total_iter += 1

        if self.n_total_iter % self.params.log_interval == 0:
            self.log_tensorboard()
        if self.n_total_iter % self.params.checkpoint_interval == 0:
            self.save_checkpoint()

    def log_tensorboard(self):
        """
        Log into tensorboard. Only by the master process.
        """
        if not self.is_master:
            return

        for param_name, param in self.student.named_parameters():
            self.tensorboard.add_scalar(tag='parameter_mean/' + param_name, scalar_value=param.data.mean(), global_step=self.n_total_iter)
            self.tensorboard.add_scalar(tag='parameter_std/' + param_name, scalar_value=param.data.std(), global_step=self.n_total_iter)
            if param.grad is None:
                continue
            self.tensorboard.add_scalar(tag="grad_mean/" + param_name, scalar_value=param.grad.data.mean(),global_step=self.n_total_iter)
            self.tensorboard.add_scalar(tag="grad_std/" + param_name, scalar_value=param.grad.data.std(), global_step=self.n_total_iter)

        self.tensorboard.add_scalar(tag="losses/cum_avg_loss_epoch", scalar_value=self.total_loss_epoch/self.n_iter, global_step=self.n_total_iter)
        self.tensorboard.add_scalar(tag="losses/loss", scalar_value=self.last_loss, global_step=self.n_total_iter)
        self.tensorboard.add_scalar(tag="losses/loss_ce", scalar_value=self.last_loss_ce, global_step=self.n_total_iter)
        if self.alpha_mlm > 0.:
            self.tensorboard.add_scalar(tag="losses/loss_mlm", scalar_value=self.last_loss_mlm, global_step=self.n_total_iter)
        if self.alpha_mse > 0.:
            self.tensorboard.add_scalar(tag="losses/loss_mse", scalar_value=self.last_loss_mse, global_step=self.n_total_iter)
        self.tensorboard.add_scalar(tag="learning_rate/lr", scalar_value=self.scheduler.get_lr()[0], global_step=self.n_total_iter)
        
        self.tensorboard.add_scalar(tag="global/memory_usage", scalar_value=psutil.virtual_memory()._asdict()['used']/1_000_000, global_step=self.n_total_iter)

    def end_epoch(self):
        """
        Finally arrived at the end of epoch (full pass on dataset).
        Do some tensorboard logging and checkpoint saving.
        """
        logger.info(f'{self.n_sequences_epoch} sequences have been trained during this epoch.')

        if self.is_master:
            self.save_checkpoint(checkpoint_name=f'model_epoch_{self.epoch}.pth')
            self.tensorboard.add_scalar(tag='epoch/loss', scalar_value=self.total_loss_epoch/self.n_iter, global_step=self.epoch)

        self.epoch += 1
        self.n_sequences_epoch = 0
        self.n_iter = 0
        self.total_loss_epoch = 0

    def save_checkpoint(self,
                        checkpoint_name: str = 'checkpoint.pth'):
        """
        Save the current state. Only by the master process.
        """
        if not self.is_master:
            return
        mdl_to_save = self.student.module if hasattr(self.student, 'module') else self.student
        mdl_to_save.config.save_pretrained(self.dump_path)
        state_dict = mdl_to_save.state_dict()
        torch.save(state_dict, os.path.join(self.dump_path, checkpoint_name))
