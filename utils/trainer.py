# @title Trainer with Multi-Loss Support
from __future__ import annotations
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass

import csv
import pandas as pd

import random
import numpy as np
import os
import logging
from datetime import datetime
import time
import copy


# >>> TRAINER - Updated with Multi-Loss Support <<<
class Trainer(object):
    """Minimal yet full featured Trainer for MAMBA_BGNN with multi-loss support."""

    # ================= init =================
    def __init__(self, model, loss_fn, optimizer, train_loader, val_loader, test_loader,
                 args, lr_scheduler=None, loss_type='bayesian'):
        """
        Args:
            loss_type: 'bayesian', 'mse', or 'smoothl1'
        """
        self.model = model
        self.loss_fn = loss_fn
        self.opt = optimizer
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.args = args
        self.lr_scheduler = lr_scheduler
        self.loss_type = loss_type.lower()

        os.makedirs(self.args['log_dir'], exist_ok=True)
        self.logger = self._get_logger()

        self.best_state, self.best_loss = None, float('inf')
        self.not_improved = 0
        self.best_path = os.path.join(args.get('log_dir'), 'best_model.pth')

        # Training time tracking
        self.epoch_times = []
        self.total_training_time = 0.0

        # --- CSV paths & headers (MAIN TABLE) ---
        self.val_csv  = os.path.join(self.args['log_dir'], 'val_metrics.csv')
        self.test_csv = os.path.join(self.args['log_dir'], 'test_metrics.csv')
        self.test_pred_csv = os.path.join(self.args['log_dir'], 'test_predictions.csv')
        self.best_val_pred_csv = os.path.join(self.args['log_dir'], 'val_predictions_best.csv')

        # Adaptive headers based on loss type
        if self.loss_type == 'bayesian':
            self.val_header  = ['epoch','nll','rmse','mae','ic','ric','crps','sharp',
                                'picp90','gap90','picp95','gap95','aurc']
            self.test_header = ['nll','rmse','mae','ic','ric','crps','sharp',
                                'picp90','gap90','picp95','gap95','aurc']
        else:
            self.val_header  = ['epoch','loss','rmse','mae','ic','ric']
            self.test_header = ['loss','rmse','mae','ic','ric']

        self._init_csv(self.val_csv,  self.val_header)
        self._init_csv(self.test_csv, self.test_header)

    # ================= logging & csv utils =================
    def _get_logger(self):
        model_name = self.args['model_name'] + ' ' + datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        logger = logging.getLogger(model_name)
        logger.setLevel(logging.DEBUG)
        fh = logging.FileHandler(os.path.join(self.args['log_dir'], model_name + '.log'))
        sh = logging.StreamHandler()
        fmt = logging.Formatter('%(asctime)s - %(levelname)s: %(message)s', '%Y-%m-%d %H:%M')
        sh.setFormatter(fmt)
        logger.addHandler(sh); logger.addHandler(fh)
        return logger

    def _init_csv(self, path, header):
        if not os.path.exists(path):
            with open(path, 'w', newline='') as f:
                csv.writer(f).writerow(header)

    def _append_csv(self, path, header, row_dict):
        row = []
        for k in header:
            v = row_dict[k]
            if isinstance(v, torch.Tensor):
                v = v.detach().cpu().item()
            if isinstance(v, (np.generic,)):
                v = float(v)
            row.append(v)
        with open(path, 'a', newline='') as f:
            csv.writer(f).writerow(row)

    def _dump_predictions(self, path, mu, sigma, y):
        df = pd.DataFrame({
            'y': y.detach().cpu().numpy().astype(float),
            'mu': mu.detach().cpu().numpy().astype(float),
            'sigma': sigma.detach().cpu().numpy().astype(float),
        })
        df.to_csv(path, index=False)

    # ================= probabilistic helpers =================
    @staticmethod
    def _to_sigma(log_var: torch.Tensor) -> torch.Tensor:
        return torch.exp(0.5 * log_var).clamp_min(1e-8)

    @staticmethod
    def _crps_gaussian(mu: torch.Tensor, sigma: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        # Closed-form CRPS for N(mu, sigma^2):
        # CRPS = sigma * [ z*(2Φ(z)-1) + 2φ(z) - 1/√π ],  z=(y-mu)/sigma
        sigma = sigma.clamp_min(1e-8)
        z = (y - mu) / sigma
        try:
            Phi = torch.special.ndtr(z)
        except AttributeError:
            Phi = 0.5 * (1.0 + torch.erf(z / math.sqrt(2.0)))
        phi = torch.exp(-0.5 * z**2) / math.sqrt(2*math.pi)
        crps = sigma * (z * (2*Phi - 1) + 2*phi - 1.0/math.sqrt(math.pi))
        return torch.clamp(crps, min=0.0)

    @staticmethod
    def _picp_and_gap(mu: torch.Tensor, sigma: torch.Tensor, y: torch.Tensor, q: float):
        # central interval coverage at nominal q (e.g., q=0.90)
        p = (1.0 + q)/2.0
        try:
            z = math.sqrt(2.0) * torch.erfinv(torch.tensor(2.0*p - 1.0, device=mu.device, dtype=mu.dtype))
        except AttributeError:
            z = math.sqrt(2.0) * torch.special.erfinv(torch.tensor(2.0*p - 1.0, device=mu.device, dtype=mu.dtype))
        lo, hi = mu - z*sigma, mu + z*sigma
        obs = ((y >= lo) & (y <= hi)).float().mean().item()
        gap = abs(obs - q)
        return obs, gap

    @staticmethod
    def _aurc_rmse(y: torch.Tensor, mu: torch.Tensor, sigma: torch.Tensor, points: int = 10) -> float:
        # Risk–Coverage AUC for RMSE (lower better)
        idx = torch.argsort(sigma)  # most certain first
        y, mu = y[idx], mu[idx]
        covs = torch.linspace(0.1, 1.0, points, device=y.device)
        prev = None; auc = 0.0
        for i, c in enumerate(covs):
            k = max(1, int(c.item()*y.numel()))
            rmse_c = torch.sqrt(torch.mean((y[:k]-mu[:k])**2)).item()
            if i > 0:
                h = (covs[i] - covs[i-1]).item()
                auc += 0.5 * h * (prev + rmse_c)
            prev = rmse_c
        return auc

    # ================= training loops =================
    def _run_epoch(self, epoch):
        epoch_start_time = time.time()
        self.model.train(); total = 0.0
        for step, (x, y) in enumerate(self.train_loader):
            self.opt.zero_grad()

            if self.loss_type == 'bayesian':
                mu, log_var = self.model(x)                           # (B,), (B,)
                loss = self.loss_fn(mu, y.squeeze(), log_var.exp())   # NLL
                loss_name = 'NLL'
            else:
                mu, _ = self.model(x)                                 # (B,), ignore log_var
                loss = self.loss_fn(mu, y.squeeze())                  # MSE or SmoothL1
                loss_name = 'MSE' if self.loss_type == 'mse' else 'SmoothL1'

            loss.backward()
            if self.args['grad_norm']:
                nn.utils.clip_grad_norm_(self.model.parameters(), self.args['max_grad_norm'])
            self.opt.step()
            total += loss.item()
            if step % self.args['log_step'] == 0:
                self.logger.info(f"Epoch {epoch} [{step}/{len(self.train_loader)}] Loss({loss_name}): {loss.item():.6f}")
        if self.lr_scheduler: self.lr_scheduler.step()
        train_loss = total / len(self.train_loader)
        epoch_time = time.time() - epoch_start_time
        self.epoch_times.append(epoch_time)
        self.logger.info(f"Epoch {epoch} Train {loss_name}: {train_loss:.6f} (Time: {epoch_time:.2f}s)")
        return train_loss

    def _validate(self, epoch):
        self.model.eval(); total = 0.0
        preds, trues = [], []

        if self.loss_type == 'bayesian':
            logvars = []
            with torch.no_grad():
                for x, y in self.val_loader:
                    mu, log_var = self.model(x)
                    total += self.loss_fn(mu, y.squeeze(), log_var.exp()).item()
                    preds.append(mu); trues.append(y.squeeze()); logvars.append(log_var.squeeze())

            preds   = torch.cat(preds, 0).squeeze(-1)
            trues   = torch.cat(trues, 0).squeeze(-1)
            logvars = torch.cat(logvars, 0).squeeze(-1)
            sigmas  = self._to_sigma(logvars)

            nll = total / len(self.val_loader)
            rmse = torch.sqrt(torch.mean((trues - preds)**2))
            mae  = torch.mean(torch.abs(trues - preds))
            ic   = self.pearson(trues, preds)
            ric  = self.ric(trues, preds)
            crps = self._crps_gaussian(preds, sigmas, trues).mean()
            sharp = sigmas.mean()
            picp90, gap90 = self._picp_and_gap(preds, sigmas, trues, q=0.90)
            picp95, gap95 = self._picp_and_gap(preds, sigmas, trues, q=0.95)
            aurc = self._aurc_rmse(trues, preds, sigmas, points=10)

            self.logger.info(
                f"VAL  RMSE:{rmse:.4f}  MAE:{mae:.4f}  IC:{ic:.4f}  RIC:{ric:.4f}  "
                f"NLL:{nll:.5f}  CRPS:{crps:.5f}  Sharp(σ):{sharp:.5f}  "
                f"PICP90:{picp90:.3f}|Gap:{gap90:.3f}  PICP95:{picp95:.3f}|Gap:{gap95:.3f}  "
                f"AURC:{aurc:.5f}"
            )

            row = {'epoch': int(epoch), 'nll': nll, 'rmse': rmse, 'mae': mae,
                   'ic': ic, 'ric': ric, 'crps': crps, 'sharp': sharp,
                   'picp90': picp90, 'gap90': gap90, 'picp95': picp95, 'gap95': gap95, 'aurc': aurc}
            self._append_csv(self.val_csv, self.val_header, row)

            return nll, (preds, sigmas, trues)

        else:
            # Deterministic case (MSE or SmoothL1)
            with torch.no_grad():
                for x, y in self.val_loader:
                    mu, _ = self.model(x)  # ignore log_var
                    total += self.loss_fn(mu, y.squeeze()).item()
                    preds.append(mu); trues.append(y.squeeze())

            preds = torch.cat(preds, 0).squeeze(-1)
            trues = torch.cat(trues, 0).squeeze(-1)

            loss = total / len(self.val_loader)
            rmse = torch.sqrt(torch.mean((trues - preds)**2))
            mae  = torch.mean(torch.abs(trues - preds))
            ic   = self.pearson(trues, preds)
            ric  = self.ric(trues, preds)

            loss_name = 'MSE' if self.loss_type == 'mse' else 'SmoothL1'
            self.logger.info(
                f"VAL  RMSE:{rmse:.4f}  MAE:{mae:.4f}  IC:{ic:.4f}  RIC:{ric:.4f}  {loss_name}:{loss:.5f}"
            )

            row = {'epoch': int(epoch), 'loss': loss, 'rmse': rmse, 'mae': mae, 'ic': ic, 'ric': ric}
            self._append_csv(self.val_csv, self.val_header, row)

            return loss, (preds, None, trues)

    def train(self):
        training_start_time = time.time()
        best_bundle = None
        metric_name = 'NLL' if self.loss_type == 'bayesian' else ('MSE' if self.loss_type == 'mse' else 'SmoothL1')
        for epoch in range(1, self.args['epochs'] + 1):
            _ = self._run_epoch(epoch)
            val_loss, bundle = self._validate(epoch)
            if val_loss < self.best_loss:
                self.best_loss = val_loss
                self.best_state = copy.deepcopy(self.model.state_dict())
                self.not_improved = 0
                self.logger.info(f'--- New best model (by VAL {metric_name}) ---')
                torch.save(self.best_state, self.best_path)
                best_bundle = bundle
            else:
                self.not_improved += 1
            if self.args['early_stop'] and self.not_improved >= self.args['early_stop_patience']:
                self.logger.info('Early stopping triggered.')
                break

        self.total_training_time = time.time() - training_start_time
        avg_epoch_time = np.mean(self.epoch_times) if self.epoch_times else 0.0
        self.logger.info(f"Training completed in {self.total_training_time:.2f}s (Avg: {avg_epoch_time:.2f}s/epoch)")

        if self.best_state is not None:
            self.model.load_state_dict(self.best_state)
        if best_bundle is not None:
            mu_b, sigma_b, y_b = best_bundle
            if sigma_b is not None:
                self._dump_predictions(self.best_val_pred_csv, mu_b, sigma_b, y_b)
            else:
                # Deterministic case: save without sigma
                df = pd.DataFrame({
                    'y': y_b.detach().cpu().numpy().astype(float),
                    'mu': mu_b.detach().cpu().numpy().astype(float),
                })
                df.to_csv(self.best_val_pred_csv, index=False)

    # ================ metrics & test ================
    @staticmethod
    def pearson(x, y):
        vx, vy = x - x.mean(), y - y.mean()
        return (vx * vy).sum() / (torch.sqrt((vx ** 2).sum()) * torch.sqrt((vy ** 2).sum()) + 1e-12)

    @staticmethod
    def rank_tensor(x):
        flat = x.view(-1)
        idx = torch.argsort(flat)
        ranks = torch.empty_like(flat, dtype=torch.float32)
        ranks[idx] = torch.arange(1, flat.numel() + 1, dtype=torch.float32, device=x.device)
        return ranks.view_as(x)

    @staticmethod
    def ric(x, y):
        rx, ry = Trainer.rank_tensor(x), Trainer.rank_tensor(y)
        return Trainer.pearson(rx, ry)

    @staticmethod
    def directional_accuracy(y_pred, y_true):
        """
        Tính directional accuracy dựa trên hướng của giá trị dự đoán và thực tế.
        """
        pred_dir = torch.sign(y_pred)
        true_dir = torch.sign(y_true)
        return float((pred_dir == true_dir).float().mean())

    def calculate_portfolio_metrics(self, preds, returns, transaction_cost=0.001):
        """
        Tính các chỉ số hiệu suất danh mục đầu tư toàn diện sau khi dự đoán.

        Args:
            preds: Giá trị dự đoán (tensor)
            returns: Giá trị thực tế (tensor)
            transaction_cost: Chi phí giao dịch (mặc định 0.1%)

        Returns:
            Dictionary chứa: sharpe, max_drawdown, calmar, hit_rate, turnover, profit_factor.
        """
        preds_np = preds.cpu().numpy()
        returns_np = returns.cpu().numpy()
        # Xác định vị thế giao dịch dựa trên hướng của dự đoán
        positions = np.sign(preds_np)
        # Tính turnover dựa trên sự khác biệt giữa các vị thế liên tiếp
        turnover = np.abs(np.diff(positions, axis=0)).mean()
        costs = turnover * transaction_cost
        # Tính lợi nhuận danh mục sau chi phí
        portfolio_returns = positions[1:] * returns_np[1:] - costs
        annual_factor = np.sqrt(252)
        std_ret = portfolio_returns.std()
        sharpe = portfolio_returns.mean() / std_ret * annual_factor if std_ret != 0 else np.inf
        cum_returns = np.cumprod(1 + portfolio_returns)
        peak = np.maximum.accumulate(cum_returns)
        drawdown = 1 - cum_returns / peak
        max_drawdown = drawdown.max()
        calmar = (portfolio_returns.mean() * 252) / max_drawdown if max_drawdown > 0 else np.inf
        hit_rate = float(np.mean(np.sign(portfolio_returns) == np.sign(returns_np[1:])))
        profit_factor = float(np.abs(portfolio_returns[portfolio_returns > 0].sum() /
                            portfolio_returns[portfolio_returns < 0].sum())) if np.any(portfolio_returns < 0) else np.inf
        return {
            'sharpe': sharpe,
            'max_drawdown': max_drawdown,
            'calmar': calmar,
            'hit_rate': hit_rate,
            'turnover': turnover,
            'profit_factor': profit_factor
        }

    def rolling_window_eval(self, data_loader, window_size=63, step_size=21):
        """
        Thực hiện đánh giá theo rolling-window (walk-forward evaluation).

        Args:
            data_loader: DataLoader chứa dữ liệu test.
            window_size: Kích thước của mỗi cửa sổ đánh giá (mặc định 63, tương đương khoảng 3 tháng giao dịch)
            step_size: Bước nhảy giữa các cửa sổ (mặc định 10, tương đương khoảng 0.5 tháng giao dịch)

        Returns:
            Một tuple (df, stats) với df là DataFrame chứa metrics theo từng cửa sổ và stats là thống kê (mean±std) của các metrics.
        """
        self.model.eval()
        windows_metrics = []
        all_x, all_y = [], []
        with torch.no_grad():
            for x, y in data_loader:
                all_x.append(x)
                all_y.append(y)
        X = torch.cat(all_x, 0)
        Y = torch.cat(all_y, 0)
        for start_idx in range(0, len(X) - window_size, step_size):
            end_idx = start_idx + window_size
            x_window = X[start_idx:end_idx]
            y_window = Y[start_idx:end_idx].reshape(-1)
            with torch.no_grad():
                preds, _ = self.model(x_window)  # ignore log_var for rolling window eval
            window_rmse = torch.sqrt(torch.mean((y_window - preds)**2)).item()
            window_mae = torch.mean(torch.abs(y_window - preds)).item()
            window_ic = self.pearson(y_window, preds).item()
            window_ric = self.ric(y_window, preds).item()
            window_dir_acc = self.directional_accuracy(preds, y_window)
            port_met = self.calculate_portfolio_metrics(preds, y_window)
            metrics = {
                'window_start': start_idx,
                'rmse': window_rmse,
                'mae': window_mae,
                'ic': window_ic,
                'ric': window_ric,
                'dir_acc': window_dir_acc
            }
            metrics.update(port_met)
            windows_metrics.append(metrics)
        df = pd.DataFrame(windows_metrics)
        stats = df.drop(['window_start'], axis=1).agg(['mean', 'std']).round(4)
        return df, stats

    def test(self):
        self.model.eval()
        preds, trues = [], []

        if self.loss_type == 'bayesian':
            logvars = []
            loss_total = 0.0
            with torch.no_grad():
                for x, y in self.test_loader:
                    mu, log_var = self.model(x)
                    loss_total += self.loss_fn(mu, y.squeeze(), log_var.exp()).item()
                    preds.append(mu); trues.append(y.squeeze()); logvars.append(log_var.squeeze())

            preds   = torch.cat(preds, 0).squeeze(-1)
            trues   = torch.cat(trues, 0).squeeze(-1)
            logvars = torch.cat(logvars, 0).squeeze(-1)
            sigmas  = self._to_sigma(logvars)
            nll = loss_total / max(1, len(self.test_loader))

            rmse = torch.sqrt(torch.mean((trues - preds)**2))
            mae  = torch.mean(torch.abs(trues - preds))
            ic   = self.pearson(trues, preds)
            ric  = self.ric(trues, preds)
            crps = self._crps_gaussian(preds, sigmas, trues).mean()
            sharp = sigmas.mean()
            picp90, gap90 = self._picp_and_gap(preds, sigmas, trues, q=0.90)
            picp95, gap95 = self._picp_and_gap(preds, sigmas, trues, q=0.95)
            aurc = self._aurc_rmse(trues, preds, sigmas, points=10)

            self.logger.info(
                f"TEST RMSE:{rmse:.4f}  MAE:{mae:.4f}  IC:{ic:.4f}  RIC:{ric:.4f}  "
                f"NLL:{nll:.5f}  CRPS:{crps:.5f}  Sharp(σ):{sharp:.5f}  "
                f"PICP90:{picp90:.3f}|Gap:{gap90:.3f}  PICP95:{picp95:.3f}|Gap:{gap95:.3f}  "
                f"AURC:{aurc:.5f}"
            )
            metrics = { 'nll': nll, 'rmse': rmse, 'mae': mae,
                   'ic': ic, 'ric': ric, 'crps': crps, 'sharp': sharp,
                   'picp90': picp90, 'gap90': gap90, 'picp95': picp95, 'gap95': gap95, 'aurc': aurc}
            self._append_csv(self.test_csv, self.test_header, metrics)

            self._dump_predictions(self.test_pred_csv, preds, sigmas, trues)

        else:
            # Deterministic case (MSE or SmoothL1)
            loss_total = 0.0
            with torch.no_grad():
                for x, y in self.test_loader:
                    mu, _ = self.model(x)  # ignore log_var
                    loss_total += self.loss_fn(mu, y.squeeze()).item()
                    preds.append(mu); trues.append(y.squeeze())

            preds = torch.cat(preds, 0).squeeze(-1)
            trues = torch.cat(trues, 0).squeeze(-1)
            loss = loss_total / max(1, len(self.test_loader))

            rmse = torch.sqrt(torch.mean((trues - preds)**2))
            mae  = torch.mean(torch.abs(trues - preds))
            ic   = self.pearson(trues, preds)
            ric  = self.ric(trues, preds)

            loss_name = 'MSE' if self.loss_type == 'mse' else 'SmoothL1'
            self.logger.info(
                f"TEST RMSE:{rmse:.4f}  MAE:{mae:.4f}  IC:{ic:.4f}  RIC:{ric:.4f}  {loss_name}:{loss:.5f}"
            )
            metrics = {'loss': loss, 'rmse': rmse, 'mae': mae, 'ic': ic, 'ric': ric}
            self._append_csv(self.test_csv, self.test_header, metrics)

            # Save predictions without sigma
            df = pd.DataFrame({
                'y': trues.detach().cpu().numpy().astype(float),
                'mu': preds.detach().cpu().numpy().astype(float),
            })
            df.to_csv(self.test_pred_csv, index=False)

        # Thêm đoạn gọi hàm đánh giá rolling-window và lưu kết quả
        window_results, window_stats = self.rolling_window_eval(self.test_loader
                                                                , self.args.get('rolling_window_size', 63)
                                                                , self.args.get('rolling_step_size', 21))
        window_results.to_csv(os.path.join(self.args['log_dir'], 'rolling_window_results.csv'))
        window_stats.to_csv(os.path.join(self.args['log_dir'], 'rolling_window_stats.csv'))
        self.logger.info("\nRolling Window Evaluation Statistics:")
        self.logger.info(window_stats)

        # Generate summary report automatically
        self.generate_summary_report()

        return metrics

    # ================ summary report generation =================
    def generate_summary_report(self):
        """
        Generate a comprehensive summary report similar to baseline_notebook.py
        Reads test metrics and creates a formatted summary with training time info
        """
        summary_txt = os.path.join(self.args['log_dir'], 'training_summary.txt')

        # Read test metrics
        if os.path.exists(self.test_csv):
            test_df = pd.read_csv(self.test_csv)
            if len(test_df) > 0:
                test_metrics = test_df.iloc[-1].to_dict()  # Get last row (latest test results)
            else:
                self.logger.warning("No test metrics found")
                return
        else:
            self.logger.warning(f"Test CSV not found: {self.test_csv}")
            return

        # Read validation metrics for best epoch info
        best_epoch = 1
        if os.path.exists(self.val_csv):
            val_df = pd.read_csv(self.val_csv)
            if len(val_df) > 0:
                loss_col = 'nll' if self.loss_type == 'bayesian' else 'loss'
                best_epoch = val_df[loss_col].idxmin() + 1

        # Calculate time statistics
        avg_epoch_time = np.mean(self.epoch_times) if self.epoch_times else 0.0
        total_epochs = len(self.epoch_times)

        # Create summary report
        with open(summary_txt, 'w') as f:
            f.write("="*80 + "\n")
            f.write("TRAINING SUMMARY REPORT\n")
            f.write("="*80 + "\n")
            f.write(f"Model: {self.args.get('model_name', 'MAMBA_BGNN')}\n")
            f.write(f"Dataset: {self.args.get('dataset', 'N/A')}\n")
            f.write(f"Loss Type: {self.loss_type.upper()}\n")
            f.write(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Log Directory: {self.args['log_dir']}\n")
            f.write("="*80 + "\n\n")

            # Training configuration
            f.write("TRAINING CONFIGURATION:\n")
            f.write("-"*80 + "\n")
            f.write(f"Total Epochs: {total_epochs}\n")
            f.write(f"Best Epoch: {best_epoch}\n")
            f.write(f"Early Stop: {self.args.get('early_stop', False)}\n")
            if self.args.get('early_stop'):
                f.write(f"Early Stop Patience: {self.args.get('early_stop_patience', 'N/A')}\n")
            f.write(f"Batch Size: {self.args.get('batch_size', 'N/A')}\n")
            f.write(f"Learning Rate: {self.args.get('lr', 'N/A')}\n")
            f.write("\n")

            # Training time summary
            f.write("TRAINING TIME:\n")
            f.write("-"*80 + "\n")
            f.write(f"Total Training Time: {self.total_training_time:.2f}s ({self.total_training_time/60:.2f} min)\n")
            f.write(f"Average Epoch Time: {avg_epoch_time:.2f}s/epoch\n")
            if self.epoch_times:
                f.write(f"Min Epoch Time: {np.min(self.epoch_times):.2f}s\n")
                f.write(f"Max Epoch Time: {np.max(self.epoch_times):.2f}s\n")
            f.write("\n")

            # Test metrics
            f.write("TEST METRICS:\n")
            f.write("-"*80 + "\n")

            if self.loss_type == 'bayesian':
                f.write(f"NLL:                {test_metrics.get('nll', 0):.6f}\n")
                f.write(f"RMSE:               {test_metrics.get('rmse', 0):.6f}\n")
                f.write(f"MAE:                {test_metrics.get('mae', 0):.6f}\n")
                f.write(f"IC:                 {test_metrics.get('ic', 0):.6f}\n")
                f.write(f"RIC:                {test_metrics.get('ric', 0):.6f}\n")
                f.write(f"CRPS:               {test_metrics.get('crps', 0):.6f}\n")
                f.write(f"Sharpness (σ):      {test_metrics.get('sharp', 0):.6f}\n")
                f.write(f"PICP 90%:           {test_metrics.get('picp90', 0):.4f}\n")
                f.write(f"Gap 90%:            {test_metrics.get('gap90', 0):.4f}\n")
                f.write(f"PICP 95%:           {test_metrics.get('picp95', 0):.4f}\n")
                f.write(f"Gap 95%:            {test_metrics.get('gap95', 0):.4f}\n")
                f.write(f"AURC (RMSE):        {test_metrics.get('aurc', 0):.6f}\n")
            else:
                loss_name = 'MSE' if self.loss_type == 'mse' else 'SmoothL1'
                f.write(f"{loss_name}:         {test_metrics.get('loss', 0):.6f}\n")
                f.write(f"RMSE:               {test_metrics.get('rmse', 0):.6f}\n")
                f.write(f"MAE:                {test_metrics.get('mae', 0):.6f}\n")
                f.write(f"IC:                 {test_metrics.get('ic', 0):.6f}\n")
                f.write(f"RIC:                {test_metrics.get('ric', 0):.6f}\n")

            f.write("\n")

            # Rolling window summary (if available)
            rolling_stats_path = os.path.join(self.args['log_dir'], 'rolling_window_stats.csv')
            if os.path.exists(rolling_stats_path):
                f.write("ROLLING WINDOW EVALUATION:\n")
                f.write("-"*80 + "\n")
                rolling_stats = pd.read_csv(rolling_stats_path, index_col=0)
                f.write(rolling_stats.to_string())
                f.write("\n\n")

            # File paths
            f.write("OUTPUT FILES:\n")
            f.write("-"*80 + "\n")
            f.write(f"Best Model:              {self.best_path}\n")
            f.write(f"Validation Metrics:      {self.val_csv}\n")
            f.write(f"Test Metrics:            {self.test_csv}\n")
            f.write(f"Test Predictions:        {self.test_pred_csv}\n")
            f.write(f"Best Val Predictions:    {self.best_val_pred_csv}\n")
            f.write(f"Rolling Window Results:  {os.path.join(self.args['log_dir'], 'rolling_window_results.csv')}\n")
            f.write(f"Rolling Window Stats:    {rolling_stats_path}\n")
            f.write("\n")

            f.write("="*80 + "\n")
            f.write("END OF REPORT\n")
            f.write("="*80 + "\n")

        self.logger.info(f"Summary report saved to: {summary_txt}")

        # Also print key metrics to console
        print("\n" + "="*80)
        print("TRAINING COMPLETED")
        print("="*80)
        print(f"Model: {self.args.get('model_name', 'MAMBA_BGNN')}")
        print(f"Total Time: {self.total_training_time:.2f}s ({self.total_training_time/60:.2f} min)")
        print(f"Avg Epoch Time: {avg_epoch_time:.2f}s")
        print("-"*80)
        if self.loss_type == 'bayesian':
            print(f"Test RMSE:  {test_metrics.get('rmse', 0):.6f}")
            print(f"Test IC:    {test_metrics.get('ic', 0):.6f}")
            print(f"Test RIC:   {test_metrics.get('ric', 0):.6f}")
            print(f"Test NLL:   {test_metrics.get('nll', 0):.6f}")
            print(f"Test CRPS:  {test_metrics.get('crps', 0):.6f}")
        else:
            print(f"Test RMSE:  {test_metrics.get('rmse', 0):.6f}")
            print(f"Test IC:    {test_metrics.get('ic', 0):.6f}")
            print(f"Test RIC:   {test_metrics.get('ric', 0):.6f}")
        print("="*80)
        print(f"Summary saved to: {summary_txt}")
        print("="*80 + "\n")

        return summary_txt
