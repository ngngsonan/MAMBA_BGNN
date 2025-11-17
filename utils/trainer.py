# @title Import libraries
from __future__ import annotations
import math
import torch
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

import matplotlib.pyplot as plt
import math


# >>> TRAINER - NEW 2 <<<
class Trainer(object):
    """Minimal yet full featured Trainer for MAMBA_BGNN (probabilistic)."""

    # ================= init =================
    def __init__(self, model, loss_fn, optimizer, train_loader, val_loader, test_loader,
                 args, lr_scheduler=None):
        self.model = model
        self.loss_fn = loss_fn
        self.opt = optimizer
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.args = args
        self.lr_scheduler = lr_scheduler

        os.makedirs(self.args['log_dir'], exist_ok=True)
        self.logger = self._get_logger()

        self.best_state, self.best_loss = None, float('inf')
        self.not_improved = 0
        self.best_path = os.path.join(args.get('log_dir'), 'best_model.pth')

        # --- CSV paths & headers (MAIN TABLE) ---
        self.val_csv  = os.path.join(self.args['log_dir'], 'val_metrics.csv')
        self.test_csv = os.path.join(self.args['log_dir'], 'test_metrics.csv')
        self.test_pred_csv = os.path.join(self.args['log_dir'], 'test_predictions.csv')
        self.best_val_pred_csv = os.path.join(self.args['log_dir'], 'val_predictions_best.csv')

        # --- Comprehensive metrics CSV ---
        self.comprehensive_csv = os.path.join(self.args['log_dir'], 'comprehensive_metrics.csv')
        self.regime_csv = os.path.join(self.args['log_dir'], 'regime_analysis.csv')
        self.stress_csv = os.path.join(self.args['log_dir'], 'stress_test.csv')

        self.val_header  = ['epoch','nll','rmse','mae','ic','ric','crps','sharp',
                            'picp90','gap90','picp95','gap95','aurc']
        self.test_header = ['nll','rmse','mae','ic','ric','crps','sharp',
                            'picp90','gap90','picp95','gap95','aurc']

        # Comprehensive header with all financial metrics
        self.comprehensive_header = [
            'nll', 'rmse', 'mae', 'ic', 'ric', 'crps', 'sharp',
            'picp90', 'gap90', 'picp95', 'gap95', 'aurc',
            'dir_acc', 'sharpe_ratio', 'max_drawdown', 'calmar_ratio',
            'info_ratio', 'hit_rate', 'tail_ratio',
            'total_return', 'net_return', 'transaction_costs',
            'strategy_volatility', 'strategy_max_drawdown'
        ]

        self._init_csv(self.val_csv,  self.val_header)
        self._init_csv(self.test_csv, self.test_header)
        self._init_csv(self.comprehensive_csv, self.comprehensive_header)

    # ================= Dataset & temporal info =================
    def set_dataset_info(self, dataset_info: dict):
        """Store dataset temporal information for comprehensive evaluation"""
        self.dataset_info = dataset_info
        # Save to JSON
        import json
        info_path = os.path.join(self.args['log_dir'], 'dataset_temporal_info.json')
        with open(info_path, 'w') as f:
            json.dump(dataset_info, f, indent=2)
        self.logger.info(f"Dataset temporal info saved to {info_path}")
        if 'train_period' in dataset_info:
            self.logger.info(f"Train period: {dataset_info['train_period']}")
        if 'test_period' in dataset_info:
            self.logger.info(f"Test period: {dataset_info['test_period']}")

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

    # ================= Advanced Financial Metrics =================
    @staticmethod
    def _information_ratio(y_pred: torch.Tensor, y_true: torch.Tensor,
                          benchmark_returns: torch.Tensor = None) -> float:
        """Information ratio - active return divided by tracking error"""
        if benchmark_returns is None:
            benchmark_returns = torch.zeros_like(y_true)
        active_returns = y_pred - benchmark_returns
        tracking_error = torch.std(active_returns)
        if tracking_error == 0:
            return 0.0
        return torch.mean(active_returns).item() / tracking_error.item()

    @staticmethod
    def _tail_ratio(returns: torch.Tensor, threshold: float = 0.05) -> float:
        """Tail ratio - ratio of average positive tail to average negative tail"""
        returns_np = returns.detach().cpu().numpy()
        upper_threshold = np.percentile(returns_np, (1 - threshold) * 100)
        lower_threshold = np.percentile(returns_np, threshold * 100)
        upper_tail = returns_np[returns_np >= upper_threshold]
        lower_tail = returns_np[returns_np <= lower_threshold]
        if len(upper_tail) == 0 or len(lower_tail) == 0:
            return 1.0
        upper_avg = np.mean(upper_tail)
        lower_avg = np.mean(lower_tail)
        if lower_avg >= 0:
            return float('inf') if upper_avg > 0 else 1.0
        return abs(upper_avg / lower_avg)

    @staticmethod
    def _profit_and_loss(y_pred: torch.Tensor, y_true: torch.Tensor,
                        transaction_cost: float = 0.001) -> dict:
        """Simulated P&L from trading strategy based on predictions"""
        positions = torch.sign(y_pred).detach().cpu().numpy()
        returns = y_true.detach().cpu().numpy()
        strategy_returns = positions * returns
        position_changes = np.abs(np.diff(np.concatenate([[0], positions])))
        total_costs = np.sum(position_changes) * transaction_cost
        cumulative_returns = np.cumprod(1 + strategy_returns) - 1
        total_return = cumulative_returns[-1] if len(cumulative_returns) > 0 else 0
        net_return = total_return - total_costs

        # Calculate max drawdown for strategy
        cumulative = np.cumprod(1 + strategy_returns)
        running_max = np.maximum.accumulate(cumulative)
        drawdown = (cumulative - running_max) / running_max
        strategy_max_dd = float(np.min(drawdown))

        return {
            'total_return': float(total_return),
            'net_return': float(net_return),
            'transaction_costs': float(total_costs),
            'strategy_volatility': float(np.std(strategy_returns)) if len(strategy_returns) > 0 else 0.0,
            'strategy_max_drawdown': strategy_max_dd
        }

    # ================= Market Regime Analysis =================
    @staticmethod
    def _classify_market_regime(returns: torch.Tensor, window: int = 20) -> torch.Tensor:
        """
        Classify market regime based on rolling volatility
        0: Low volatility (stable), 1: High volatility (volatile)
        """
        returns_np = returns.detach().cpu().numpy()
        rolling_vol = pd.Series(returns_np).rolling(window=window, min_periods=window//2).std()
        vol_median = rolling_vol.median()
        regime = (rolling_vol > vol_median).astype(int).values
        return torch.tensor(regime, dtype=torch.int, device=returns.device)

    @staticmethod
    def _regime_performance_analysis(y_pred: torch.Tensor, y_true: torch.Tensor,
                                     regimes: torch.Tensor) -> dict:
        """Analyze performance metrics across different market regimes"""
        results = {}
        for regime_id in [0, 1]:  # 0: stable, 1: volatile
            regime_name = "stable" if regime_id == 0 else "volatile"
            mask = (regimes == regime_id)
            if mask.sum() == 0:
                continue
            pred_regime = y_pred[mask]
            true_regime = y_true[mask]

            results[f'regime_{regime_name}_samples'] = int(mask.sum().item())
            results[f'regime_{regime_name}_rmse'] = float(torch.sqrt(torch.mean((pred_regime - true_regime)**2)))
            results[f'regime_{regime_name}_mae'] = float(torch.mean(torch.abs(pred_regime - true_regime)))
            results[f'regime_{regime_name}_dir_acc'] = float(Trainer.directional_accuracy(pred_regime, true_regime))

            if len(pred_regime) > 1:
                corr_matrix = torch.corrcoef(torch.stack([pred_regime, true_regime]))
                results[f'regime_{regime_name}_correlation'] = float(corr_matrix[0, 1])
            else:
                results[f'regime_{regime_name}_correlation'] = 0.0
        return results

    @staticmethod
    def _market_stress_test(y_pred: torch.Tensor, y_true: torch.Tensor,
                           stress_percentile: float = 5.0) -> dict:
        """Test model performance during market stress periods (extreme negative returns)"""
        returns_np = y_true.detach().cpu().numpy()
        stress_threshold = np.percentile(returns_np, stress_percentile)
        stress_mask = y_true <= stress_threshold

        if stress_mask.sum() == 0:
            return {'stress_samples': 0}

        pred_stress = y_pred[stress_mask]
        true_stress = y_true[stress_mask]

        results = {
            'stress_samples': int(stress_mask.sum().item()),
            'stress_rmse': float(torch.sqrt(torch.mean((pred_stress - true_stress)**2))),
            'stress_mae': float(torch.mean(torch.abs(pred_stress - true_stress))),
            'stress_dir_acc': float(Trainer.directional_accuracy(pred_stress, true_stress)),
            'stress_avg_return': float(torch.mean(true_stress))
        }

        if len(pred_stress) > 1:
            corr_matrix = torch.corrcoef(torch.stack([pred_stress, true_stress]))
            results['stress_correlation'] = float(corr_matrix[0, 1])
        else:
            results['stress_correlation'] = 0.0

        return results

    # ================= training loops =================
    def _run_epoch(self, epoch):
        self.model.train(); total = 0.0
        for step, (x, y) in enumerate(self.train_loader):
            self.opt.zero_grad()
            mu, log_var = self.model(x)                               # (B,), (B,)
            loss = self.loss_fn(mu, y.squeeze(), log_var.exp())       # NLL
            loss.backward()
            if self.args['grad_norm']:
                nn.utils.clip_grad_norm_(self.model.parameters(), self.args['max_grad_norm'])
            self.opt.step()
            total += loss.item()
            if step % self.args['log_step'] == 0:
                self.logger.info(f"Epoch {epoch} [{step}/{len(self.train_loader)}] Loss(NLL): {loss.item():.6f}")
        if self.lr_scheduler: self.lr_scheduler.step()
        train_loss = total / len(self.train_loader)
        self.logger.info(f"Epoch {epoch} Train NLL: {train_loss:.6f}")
        return train_loss

    def _validate(self, epoch):
        self.model.eval(); total = 0.0
        preds, trues, logvars = [], [], []
        with torch.no_grad():
            for x, y in self.val_loader:
                mu, log_var = self.model(x)
                total += self.loss_fn(mu, y.squeeze(), log_var.exp()).item()
                preds.append(mu); trues.append(y.squeeze()); logvars.append(log_var.squeeze())
        nll = total / len(self.val_loader)

        preds   = torch.cat(preds, 0).squeeze(-1)
        trues   = torch.cat(trues, 0).squeeze(-1)
        logvars = torch.cat(logvars, 0).squeeze(-1)
        sigmas  = self._to_sigma(logvars)

        # ---- MAIN metrics ----
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

        # write CSV row
        # row = {'epoch': int(epoch), 'nll': nll.item(), 'rmse': rmse.item(), 'mae': mae.item(),
        #        'ic': ic.item(), 'ric': ric.item(), 'crps': crps.item(), 'sharp': sharp.item(),
        #        'picp90': picp90, 'gap90': gap90, 'picp95': picp95, 'gap95': gap95, 'aurc': aurc}
        row = {'epoch': int(epoch), 'nll': nll, 'rmse': rmse, 'mae': mae,
               'ic': ic, 'ric': ric, 'crps': crps, 'sharp': sharp,
               'picp90': picp90, 'gap90': gap90, 'picp95': picp95, 'gap95': gap95, 'aurc': aurc}
        self._append_csv(self.val_csv, self.val_header, row)

        # return bundle to dump best-val predictions later
        return nll, (preds, sigmas, trues)

    def train(self):
        best_bundle = None
        for epoch in range(1, self.args['epochs'] + 1):
            _ = self._run_epoch(epoch)
            nll, bundle = self._validate(epoch)
            if nll < self.best_loss:
                self.best_loss = nll
                self.best_state = copy.deepcopy(self.model.state_dict())
                self.not_improved = 0
                self.logger.info('--- New best model (by VAL NLL) ---')
                torch.save(self.best_state, self.best_path)
                best_bundle = bundle
            else:
                self.not_improved += 1
            if self.args['early_stop'] and self.not_improved >= self.args['early_stop_patience']:
                self.logger.info('Early stopping triggered.')
                break
        if self.best_state is not None:
            self.model.load_state_dict(self.best_state)
        if best_bundle is not None:
            mu_b, sigma_b, y_b = best_bundle
            self._dump_predictions(self.best_val_pred_csv, mu_b, sigma_b, y_b)

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
                preds, logvars = self.model(x_window)
                sigmas = self._to_sigma(logvars)
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
        self.model.eval(); preds, trues, logvars = [], [], []
        nll_total = 0.0
        with torch.no_grad():
            for x, y in self.test_loader:
                mu, log_var = self.model(x)
                nll_total += self.loss_fn(mu, y.squeeze(), log_var.exp()).item()
                preds.append(mu); trues.append(y.squeeze()); logvars.append(log_var.squeeze())

        preds   = torch.cat(preds, 0).squeeze(-1)
        trues   = torch.cat(trues, 0).squeeze(-1)
        logvars = torch.cat(logvars, 0).squeeze(-1)
        sigmas  = self._to_sigma(logvars)
        nll = nll_total / max(1, len(self.test_loader))

        # ---- Basic & Probabilistic metrics ----
        rmse = torch.sqrt(torch.mean((trues - preds)**2))
        mae  = torch.mean(torch.abs(trues - preds))
        ic   = self.pearson(trues, preds)
        ric  = self.ric(trues, preds)
        crps = self._crps_gaussian(preds, sigmas, trues).mean()
        sharp = sigmas.mean()
        picp90, gap90 = self._picp_and_gap(preds, sigmas, trues, q=0.90)
        picp95, gap95 = self._picp_and_gap(preds, sigmas, trues, q=0.95)
        aurc = self._aurc_rmse(trues, preds, sigmas, points=10)

        # ---- Financial metrics ----
        dir_acc = self.directional_accuracy(preds, trues)
        port_metrics = self.calculate_portfolio_metrics(preds, trues, transaction_cost=0.001)
        sharpe_ratio = port_metrics['sharpe']
        max_drawdown = port_metrics['max_drawdown']
        calmar = port_metrics['calmar']
        hit_rate = port_metrics['hit_rate']

        # ---- Advanced Financial metrics ----
        info_ratio = self._information_ratio(preds, trues)
        tail_ratio = self._tail_ratio(preds)
        pnl_results = self._profit_and_loss(preds, trues, transaction_cost=0.001)

        # ---- Market Regime Analysis ----
        regimes = self._classify_market_regime(trues, window=20)
        regime_analysis = self._regime_performance_analysis(preds, trues, regimes)

        # ---- Stress Test ----
        stress_results = self._market_stress_test(preds, trues, stress_percentile=5.0)

        # ---- Log results ----
        self.logger.info("="*80)
        self.logger.info("COMPREHENSIVE TEST RESULTS")
        self.logger.info("="*80)
        self.logger.info(
            f"Probabilistic: RMSE:{rmse:.4f}  MAE:{mae:.4f}  IC:{ic:.4f}  RIC:{ric:.4f}  "
            f"NLL:{nll:.5f}  CRPS:{crps:.5f}  Sharp(σ):{sharp:.5f}  "
            f"PICP90:{picp90:.3f}|Gap:{gap90:.3f}  PICP95:{picp95:.3f}|Gap:{gap95:.3f}  "
            f"AURC:{aurc:.5f}"
        )
        self.logger.info(
            f"Financial:     DirAcc:{dir_acc:.4f}  Sharpe:{sharpe_ratio:.4f}  "
            f"MaxDD:{max_drawdown:.4f}  Calmar:{calmar:.4f}  HitRate:{hit_rate:.4f}  "
            f"InfoRatio:{info_ratio:.4f}  TailRatio:{tail_ratio:.4f}"
        )
        self.logger.info(
            f"P&L:           TotalReturn:{pnl_results['total_return']:.4f}  "
            f"NetReturn:{pnl_results['net_return']:.4f}  TxnCost:{pnl_results['transaction_costs']:.4f}  "
            f"StratVol:{pnl_results['strategy_volatility']:.4f}  StratMaxDD:{pnl_results['strategy_max_drawdown']:.4f}"
        )

        # Log regime analysis
        self.logger.info("\nMarket Regime Analysis:")
        for key, value in regime_analysis.items():
            if 'samples' in key:
                self.logger.info(f"  {key}: {value}")
            else:
                self.logger.info(f"  {key}: {value:.4f}")

        # Log stress test
        self.logger.info("\nMarket Stress Test (Bottom 5%):")
        for key, value in stress_results.items():
            if 'samples' in key:
                self.logger.info(f"  {key}: {value}")
            else:
                self.logger.info(f"  {key}: {value:.4f}")

        # ---- Compile all metrics ----
        metrics = {
            'nll': nll, 'rmse': rmse, 'mae': mae,
            'ic': ic, 'ric': ric, 'crps': crps, 'sharp': sharp,
            'picp90': picp90, 'gap90': gap90, 'picp95': picp95, 'gap95': gap95, 'aurc': aurc
        }

        comprehensive_metrics = {
            'nll': nll, 'rmse': rmse, 'mae': mae,
            'ic': ic, 'ric': ric, 'crps': crps, 'sharp': sharp,
            'picp90': picp90, 'gap90': gap90, 'picp95': picp95, 'gap95': gap95, 'aurc': aurc,
            'dir_acc': dir_acc, 'sharpe_ratio': sharpe_ratio, 'max_drawdown': max_drawdown,
            'calmar_ratio': calmar, 'info_ratio': info_ratio, 'hit_rate': hit_rate,
            'tail_ratio': tail_ratio,
            'total_return': pnl_results['total_return'],
            'net_return': pnl_results['net_return'],
            'transaction_costs': pnl_results['transaction_costs'],
            'strategy_volatility': pnl_results['strategy_volatility'],
            'strategy_max_drawdown': pnl_results['strategy_max_drawdown']
        }

        # ---- Save to CSVs ----
        self._append_csv(self.test_csv, self.test_header, metrics)
        self._append_csv(self.comprehensive_csv, self.comprehensive_header, comprehensive_metrics)

        # Save regime analysis
        regime_df = pd.DataFrame([regime_analysis])
        regime_df.to_csv(self.regime_csv, index=False)

        # Save stress test
        stress_df = pd.DataFrame([stress_results])
        stress_df.to_csv(self.stress_csv, index=False)

        # Save predictions
        self._dump_predictions(self.test_pred_csv, preds, sigmas, trues)

        # ---- Rolling window evaluation ----
        window_results, window_stats = self.rolling_window_eval(
            self.test_loader,
            self.args.get('rolling_window_size', 63),
            self.args.get('rolling_step_size', 21)
        )
        window_results.to_csv(os.path.join(self.args['log_dir'], 'rolling_window_results.csv'), index=False)
        window_stats.to_csv(os.path.join(self.args['log_dir'], 'rolling_window_stats.csv'))
        self.logger.info("\nRolling Window Evaluation Statistics:")
        self.logger.info(window_stats)

        # ---- Save evaluation summary ----
        summary_path = os.path.join(self.args['log_dir'], 'evaluation_summary.txt')
        with open(summary_path, 'w') as f:
            f.write(f"COMPREHENSIVE EVALUATION SUMMARY\n")
            f.write(f"{'='*80}\n\n")

            f.write("PROBABILISTIC METRICS:\n")
            f.write(f"  NLL:      {comprehensive_metrics['nll']:.6f}\n")
            f.write(f"  RMSE:     {comprehensive_metrics['rmse']:.6f}\n")
            f.write(f"  MAE:      {comprehensive_metrics['mae']:.6f}\n")
            f.write(f"  IC:       {comprehensive_metrics['ic']:.6f}\n")
            f.write(f"  RIC:      {comprehensive_metrics['ric']:.6f}\n")
            f.write(f"  CRPS:     {comprehensive_metrics['crps']:.6f}\n")
            f.write(f"  AURC:     {comprehensive_metrics['aurc']:.6f}\n\n")

            f.write("FINANCIAL METRICS:\n")
            f.write(f"  Directional Accuracy: {comprehensive_metrics['dir_acc']:.4f}\n")
            f.write(f"  Sharpe Ratio:         {comprehensive_metrics['sharpe_ratio']:.4f}\n")
            f.write(f"  Max Drawdown:         {comprehensive_metrics['max_drawdown']:.4f}\n")
            f.write(f"  Calmar Ratio:         {comprehensive_metrics['calmar_ratio']:.4f}\n")
            f.write(f"  Information Ratio:    {comprehensive_metrics['info_ratio']:.4f}\n")
            f.write(f"  Hit Rate:             {comprehensive_metrics['hit_rate']:.4f}\n")
            f.write(f"  Tail Ratio:           {comprehensive_metrics['tail_ratio']:.4f}\n\n")

            f.write("P&L ANALYSIS:\n")
            f.write(f"  Total Return:         {comprehensive_metrics['total_return']:.4f}\n")
            f.write(f"  Net Return:           {comprehensive_metrics['net_return']:.4f}\n")
            f.write(f"  Transaction Costs:    {comprehensive_metrics['transaction_costs']:.6f}\n")
            f.write(f"  Strategy Volatility:  {comprehensive_metrics['strategy_volatility']:.6f}\n")
            f.write(f"  Strategy Max DD:      {comprehensive_metrics['strategy_max_drawdown']:.4f}\n\n")

            f.write("MARKET REGIME ANALYSIS:\n")
            for key, value in regime_analysis.items():
                f.write(f"  {key}: {value}\n")
            f.write("\n")

            f.write("STRESS TEST RESULTS:\n")
            for key, value in stress_results.items():
                f.write(f"  {key}: {value}\n")

        self.logger.info(f"Evaluation summary saved to: {summary_path}")
        self.logger.info("="*80)

        return comprehensive_metrics
