import os
import time
import csv
import warnings

import numpy as np
import torch
import torch.nn as nn
from torch import optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, Subset

import models.FreSH
from data_provider.data_factory import data_provider
from data_provider.uea import collate_fn
from exp.exp_basic import Exp_Basic
from utils.tools import EarlyStopping, cal_accuracy
from layers.PLoss import PolyLoss
from layers.MixUP import random_mixup_with_extra


warnings.filterwarnings('ignore')


def count_parameters(model):
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total params: {total_params:,}")
    print(f"Trainable params: {trainable_params:,}")
    return total_params, trainable_params


class Exp_FreSH(Exp_Basic):
    def __init__(self, args):
        super(Exp_FreSH, self).__init__(args)

    def _build_model(self):
        train_data, train_loader = self._get_data(flag='TRAIN')
        test_data, test_loader = self._get_data(flag='TEST')

        self.args.seq_len = max(train_data.max_seq_len, test_data.max_seq_len)
        self.args.pred_len = 0
        self.args.enc_in = train_data.feature_df.shape[1]
        self.args.num_class = len(train_data.class_names)
        self.args.mixup_type = 'standard'
        self.alpha = 0.2
        self.mix_ratio = 0.5

        base_model = self.model_dict[self.args.model].Model(self.args).float()
        count_parameters(base_model)

        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(base_model, device_ids=self.args.device_ids)
        else:
            model = base_model

        return model

    def _get_data(self, flag):
        data_set, data_loader = data_provider(self.args, flag)
        return data_set, data_loader

    def _build_train_val_loaders(self, train_data):
        """Hold out 20% of the training set for validation.

        UEA only provides fixed TRAIN/TEST files. To avoid using the test set
        for model selection (data leakage), a deterministic validation split is
        created from TRAIN. TEST is reserved exclusively for final evaluation.
        """
        n = len(train_data)
        indices = np.arange(n)
        rng = np.random.RandomState(2021)
        rng.shuffle(indices)

        val_size = max(1, int(round(n * 0.2)))
        val_indices = indices[:val_size]
        train_indices = indices[val_size:]

        collate = lambda x: collate_fn(x, max_len=self.args.seq_len)
        train_loader = DataLoader(
            Subset(train_data, train_indices),
            batch_size=self.args.batch_size,
            shuffle=True,
            num_workers=self.args.num_workers,
            drop_last=False,
            collate_fn=collate,
        )
        vali_loader = DataLoader(
            Subset(train_data, val_indices),
            batch_size=self.args.batch_size,
            shuffle=False,
            num_workers=self.args.num_workers,
            drop_last=False,
            collate_fn=collate,
        )
        return train_loader, vali_loader

    def _select_optimizer(self):
        model_optim = optim.RAdam(self.model.parameters(), lr=self.args.learning_rate)
        scheduler = CosineAnnealingLR(
            optimizer=model_optim,
            T_max=self.args.train_epochs,
            eta_min=1e-6,
        )
        return model_optim, scheduler

    def _select_criterion(self):
        return PolyLoss(lambda_p=1.0, reduction='mean')

    def vali(self, vali_loader, criterion):
        total_loss = []
        preds, trues = [], []
        self.model.eval()

        with torch.no_grad():
            for batch_x, label, padding_mask in vali_loader:
                batch_x = batch_x.float().to(self.device)
                label = label.to(self.device)
                padding_mask = padding_mask.float().to(self.device)

                outputs = self.model(batch_x, padding_mask, None)
                if isinstance(outputs, tuple):
                    outputs = outputs[0]

                loss = criterion(outputs, label.long().squeeze())
                total_loss.append(loss.item())
                preds.append(outputs.detach())
                trues.append(label)

        total_loss = np.average(total_loss)
        preds = torch.cat(preds, 0)
        trues = torch.cat(trues, 0)
        predictions = torch.argmax(torch.nn.functional.softmax(preds, dim=1), dim=1).cpu().numpy()
        trues = trues.flatten().cpu().numpy()
        accuracy = cal_accuracy(predictions, trues)
        self.model.train()
        return total_loss, accuracy

    def train(self, setting):
        train_data, _ = self._get_data(flag='TRAIN')
        train_loader, vali_loader = self._build_train_val_loaders(train_data)

        path = os.path.join(self.args.checkpoints, setting)
        os.makedirs(path, exist_ok=True)

        alpha = self.alpha
        mix_ratio = self.mix_ratio
        prev_val_loss = float('inf')
        stagnation_count = 0
        delta_threshold = 1e-3
        patience_mixup = 2

        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)
        model_optim, scheduler = self._select_optimizer()
        criterion = self._select_criterion()
        time_now = time.time()
        train_steps = len(train_loader)

        for epoch in range(self.args.train_epochs):
            self.model.train()
            train_loss = []
            iter_count = 0
            epoch_time = time.time()

            for i, (batch_x, label, padding_mask) in enumerate(train_loader):
                iter_count += 1
                model_optim.zero_grad()
                batch_x = batch_x.float().to(self.device)
                label = label.to(self.device)
                padding_mask = padding_mask.float().to(self.device)

                if self.args.mixup_type == 'standard':
                    inputs, mixed_labels = random_mixup_with_extra(
                        batch_x, label, alpha=alpha, mix_ratio=mix_ratio
                    )
                    outputs = self.model(inputs, padding_mask, None)
                    loss = sum(
                        lam * criterion(outputs[j].unsqueeze(0), y_a.unsqueeze(0)) +
                        (1 - lam) * criterion(outputs[j].unsqueeze(0), y_b.unsqueeze(0))
                        for j, (lam, y_a, y_b) in enumerate(mixed_labels)
                    ) / len(mixed_labels)
                else:
                    outputs = self.model(batch_x, padding_mask, None)
                    loss = criterion(outputs, label.long().squeeze(-1))

                train_loss.append(loss.item())
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=4.0)
                model_optim.step()

                if (i + 1) % 100 == 0:
                    print(f"\titers: {i + 1}, epoch: {epoch + 1} | loss: {loss.item():.7f}")
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                    print(f'\tspeed: {speed:.4f}s/iter; left time: {left_time:.4f}s')
                    iter_count = 0
                    time_now = time.time()

            scheduler.step()
            train_loss_avg = np.average(train_loss)
            vali_loss, val_acc = self.vali(vali_loader, criterion)

            print(f"Epoch: {epoch + 1}, Steps: {train_steps} | Train Loss: {train_loss_avg:.3f} "
                  f"Vali Loss: {vali_loss:.3f} Vali Acc: {val_acc:.3f}")

            if vali_loss > prev_val_loss - delta_threshold:
                stagnation_count += 1
            else:
                stagnation_count = 0
            prev_val_loss = vali_loss

            if stagnation_count >= patience_mixup:
                mix_ratio = max(mix_ratio * 0.5, 0.05)
                print(f"[MixUp] validation loss stagnated, reducing mix_ratio -> {mix_ratio:.4f}")
                stagnation_count = 0

            early_stopping(-val_acc, self.model, path)
            if early_stopping.early_stop:
                print("Early stopping")
                break

        best_model_path = os.path.join(path, 'checkpoint.pth')
        self.model.load_state_dict(torch.load(best_model_path))
        return self.model

    def test(self, setting, test=0):
        test_data, test_loader = self._get_data(flag='TEST')

        if test:
            print('Loading model...')
            self.model.load_state_dict(
                torch.load(os.path.join('./checkpoints/' + setting, 'checkpoint.pth'))
            )

        self.model.eval()
        preds, trues = [], []

        with torch.no_grad():
            for batch_x, label, padding_mask in test_loader:
                batch_x = batch_x.float().to(self.device)
                label = label.to(self.device)
                padding_mask = padding_mask.float().to(self.device)

                outputs = self.model(batch_x, padding_mask, None)
                if isinstance(outputs, tuple):
                    outputs = outputs[0]

                preds.append(outputs.detach())
                trues.append(label)

        preds = torch.cat(preds, 0)
        trues = torch.cat(trues, 0)
        predictions = torch.argmax(torch.nn.functional.softmax(preds, dim=1), dim=1).cpu().numpy()
        true_labels = trues.flatten().cpu().numpy()
        accuracy = cal_accuracy(predictions, true_labels)

        result_path = './results/test_accuracy.csv'
        os.makedirs(os.path.dirname(result_path), exist_ok=True)
        with open(result_path, 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['Dataset', 'Test Accuracy'])
            writer.writerow([self.args.model_id, accuracy])

        print(f"Overall test accuracy: {accuracy:.4f}")
        return accuracy
