"""Risk-aware training utilities for the ICML 2026 preparation pipeline."""

from __future__ import annotations

import csv
import logging
import math
import os
from dataclasses import dataclass
from datetime import datetime
from statistics import mean
from typing import Dict, Iterable, Optional, Tuple

import torch
import torch.nn.functional as F
from torch import nn


@dataclass
class RiskAwareLossConfig:
    kl_weight: float = 1e-3
    risk_weight: float = 5e-2
    alpha: float = 0.1  # conformal target


class RiskAwareLoss(nn.Module):
    """Combine Gaussian NLL with KL regularisation and CVaR-style penalties."""

    def __init__(self, config: RiskAwareLossConfig) -> None:
        super().__init__()
        self.config = config
        self.gaussian = nn.GaussianNLLLoss(full=True, reduction='mean')

    def forward(
        self,
        mean: torch.Tensor,
        log_var: torch.Tensor,
        targets: torch.Tensor,
        aux: Dict[str, torch.Tensor],
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        base_loss = self.gaussian(mean, targets, log_var.exp())
        regimes = aux['regimes']
        kl_total = regimes['kl_global'] + regimes['kl_sector'] + regimes['kl_asset']

        sigma = torch.exp(0.5 * log_var)
        cvar_multiplier = aux['decoder']['cvar_multiplier']
        predicted_tail = mean - cvar_multiplier * sigma
        risk_penalty = F.relu(predicted_tail - targets).mean()

        total = base_loss + self.config.kl_weight * kl_total + self.config.risk_weight * risk_penalty
        components = {
            'nll': float(base_loss.detach().cpu()),
            'kl': float(kl_total.detach().cpu()),
            'risk': float(risk_penalty.detach().cpu()),
            'total': float(total.detach().cpu()),
        }
        return total, components


class ConformalCalibrator:
    """Simple conformal calibration using absolute residual statistics."""

    def __init__(self, alpha: float = 0.1, auto_scale: bool = True) -> None:
        self.alpha = alpha
        self.target = 1.0 - alpha
        self.auto_scale = auto_scale
        self.quantile: Optional[torch.Tensor] = None
        self.scale_factor: float = 1.0
        self.latest_coverage: float = 0.0

    def fit(self, mu: torch.Tensor, sigma: torch.Tensor, targets: torch.Tensor) -> None:
        sigma_clamped = sigma.clamp_min(1e-6)
        residual = torch.abs(targets - mu) / sigma_clamped
        quantile = torch.quantile(residual, 1 - self.alpha)
        coverage = float(((torch.abs(targets - mu) <= quantile * sigma_clamped).float().mean()).item())
        scale_factor = 1.0
        if self.auto_scale and coverage > 1e-6:
            desired = self.target
            scale_factor = desired / coverage
            scale_factor = float(torch.clamp(torch.tensor(scale_factor), 0.5, 5.0).item())
            quantile = quantile * scale_factor
            coverage = float(((torch.abs(targets - mu) <= quantile * sigma_clamped).float().mean()).item())

        self.quantile = quantile.detach()
        self.scale_factor = scale_factor
        self.latest_coverage = coverage

    def calibrate(self, mu: torch.Tensor, sigma: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.quantile is None:
            raise RuntimeError("Calibrator not fitted")
        width = self.quantile.to(mu.device) * sigma.clamp_min(1e-6)
        return mu - width, mu + width


class RiskAwareTrainer:
    """Trainer integrating variational regularisation and conformal calibration."""

    def __init__(
        self,
        model: nn.Module,
        loss_fn: RiskAwareLoss,
        optimizer: torch.optim.Optimizer,
        train_loader: Iterable,
        val_loader: Iterable,
        test_loader: Iterable,
        args: Dict,
        lr_scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
        calibrator: Optional[ConformalCalibrator] = None,
    ) -> None:
        self.model = model
        self.loss_fn = loss_fn
        self.opt = optimizer
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.args = args
        self.lr_scheduler = lr_scheduler
        auto_scale = args.get('conformal_auto_scale', True)
        self.calibrator = calibrator or ConformalCalibrator(alpha=loss_fn.config.alpha, auto_scale=auto_scale)

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)

        os.makedirs(self.args['log_dir'], exist_ok=True)
        self.logger = self._get_logger()

        self.val_history = []

        self.best_state: Optional[Dict[str, torch.Tensor]] = None
        self.best_loss: float = float('inf')
        self.not_improved = 0
        self.best_path = os.path.join(self.args['log_dir'], 'best_model_icml2026.pth')

        self.val_csv = os.path.join(self.args['log_dir'], 'val_metrics_icml2026.csv')
        self.test_csv = os.path.join(self.args['log_dir'], 'test_metrics_icml2026.csv')
        self._init_csv(self.val_csv, ['epoch', 'val_nll', 'val_kl', 'val_risk', 'val_total'])
        self._init_csv(self.test_csv, [
            'nll', 'rmse', 'mae', 'ic', 'ric', 'crps', 'sharp',
            'picp90', 'gap90', 'picp95', 'gap95', 'aurc',
            'conformal_coverage', 'conformal_target', 'conformal_scale',
            'conformal_val_coverage'
        ])

    # ------------------------------------------------------------------
    def _get_logger(self) -> logging.Logger:
        model_name = self.args['model_name'] + ' ' + datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        logger = logging.getLogger(model_name)
        logger.setLevel(logging.DEBUG)
        fh = logging.FileHandler(os.path.join(self.args['log_dir'], model_name + '.log'))
        sh = logging.StreamHandler()
        fmt = logging.Formatter('%(asctime)s - %(levelname)s: %(message)s', '%Y-%m-%d %H:%M')
        sh.setFormatter(fmt)
        logger.addHandler(sh)
        logger.addHandler(fh)
        return logger

    @staticmethod
    def _init_csv(path: str, header) -> None:
        if not os.path.exists(path):
            with open(path, 'w', newline='') as f:
                csv.writer(f).writerow(header)

    @staticmethod
    def _append_csv(path: str, row) -> None:
        with open(path, 'a', newline='') as f:
            csv.writer(f).writerow(row)

    @staticmethod
    def _to_sigma(log_var: torch.Tensor) -> torch.Tensor:
        return torch.exp(0.5 * log_var).clamp_min(1e-8)

    @staticmethod
    def pearson(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        vx, vy = x - x.mean(), y - y.mean()
        return (vx * vy).sum() / (torch.sqrt((vx ** 2).sum()) * torch.sqrt((vy ** 2).sum()) + 1e-12)

    @staticmethod
    def rank_tensor(x: torch.Tensor) -> torch.Tensor:
        flat = x.view(-1)
        idx = torch.argsort(flat)
        ranks = torch.empty_like(flat, dtype=torch.float32)
        ranks[idx] = torch.arange(1, flat.numel() + 1, dtype=torch.float32, device=x.device)
        return ranks.view_as(x)

    @classmethod
    def ric(cls, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        rx, ry = cls.rank_tensor(x), cls.rank_tensor(y)
        return cls.pearson(rx, ry)

    @staticmethod
    def _crps_gaussian(mu: torch.Tensor, sigma: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        sigma = sigma.clamp_min(1e-8)
        z = (y - mu) / sigma
        Phi = 0.5 * (1.0 + torch.erf(z / math.sqrt(2.0)))
        phi = torch.exp(-0.5 * z**2) / math.sqrt(2 * math.pi)
        return sigma * (z * (2 * Phi - 1) + 2 * phi - 1.0 / math.sqrt(math.pi))

    @staticmethod
    def _picp_and_gap(mu: torch.Tensor, sigma: torch.Tensor, y: torch.Tensor, q: float):
        p = (1.0 + q) / 2.0
        z = math.sqrt(2.0) * torch.erfinv(torch.tensor(2.0 * p - 1.0, device=mu.device, dtype=mu.dtype))
        lo, hi = mu - z * sigma, mu + z * sigma
        obs = ((y >= lo) & (y <= hi)).float().mean().item()
        gap = abs(obs - q)
        return obs, gap

    @staticmethod
    def _aurc_rmse(y: torch.Tensor, mu: torch.Tensor, sigma: torch.Tensor, points: int = 10) -> float:
        idx = torch.argsort(sigma)
        y, mu = y[idx], mu[idx]
        covs = torch.linspace(0.1, 1.0, points, device=y.device)
        prev = None
        auc = 0.0
        for i, c in enumerate(covs):
            k = max(1, int(c.item() * y.numel()))
            rmse_c = torch.sqrt(torch.mean((y[:k] - mu[:k]) ** 2)).item()
            if i > 0:
                h = (covs[i] - covs[i - 1]).item()
                auc += 0.5 * h * (prev + rmse_c)
            prev = rmse_c
        return auc

    # ------------------------------------------------------------------
    def _run_epoch(self, epoch: int) -> Dict[str, float]:
        self.model.train()
        total_loss = 0.0
        components_acc = {'nll': 0.0, 'kl': 0.0, 'risk': 0.0}
        steps = 0
        for x, y in self.train_loader:
            x = x.to(self.device)
            y = y.squeeze().to(self.device)

            self.opt.zero_grad()
            mean, log_var, aux = self.model(x)
            loss, components = self.loss_fn(mean, log_var, y, aux)
            loss.backward()
            if self.args.get('grad_norm', False):
                nn.utils.clip_grad_norm_(self.model.parameters(), self.args['max_grad_norm'])
            self.opt.step()

            total_loss += components['total']
            for k in components_acc:
                components_acc[k] += components[k]
            steps += 1

        if self.lr_scheduler is not None:
            self.lr_scheduler.step()

        avg = {k: v / max(1, steps) for k, v in components_acc.items()}
        avg['total'] = total_loss / max(1, steps)
        self.logger.info(
            f"Epoch {epoch} Train Total: {avg['total']:.4f} | "
            f"NLL: {avg['nll']:.4f} | KL: {avg['kl']:.4f} | Risk: {avg['risk']:.4f}"
        )
        return avg

    def _validate(self, epoch: int) -> Tuple[float, Dict[str, torch.Tensor]]:
        self.model.eval()
        losses = []
        comp_acc = {'nll': 0.0, 'kl': 0.0, 'risk': 0.0}
        preds, trues, sigmas = [], [], []
        with torch.no_grad():
            for x, y in self.val_loader:
                x = x.to(self.device)
                y = y.squeeze().to(self.device)
                mean, log_var, aux = self.model(x)
                loss, components = self.loss_fn(mean, log_var, y, aux)
                losses.append(loss.item())
                for k in comp_acc:
                    comp_acc[k] += components[k]
                preds.append(mean)
                trues.append(y)
                sigmas.append(self._to_sigma(log_var))
        steps = max(1, len(self.val_loader))
        avg_loss = sum(losses) / steps
        avg_components = {k: v / steps for k, v in comp_acc.items()}

        mu = torch.cat(preds)
        sigma = torch.cat(sigmas)
        targets = torch.cat(trues)
        self._append_csv(self.val_csv, [epoch, avg_components['nll'], avg_components['kl'], avg_components['risk'], avg_loss])

        self.logger.info(
            f"Epoch {epoch} Val Total: {avg_loss:.4f} | NLL: {avg_components['nll']:.4f} | "
            f"KL: {avg_components['kl']:.4f} | Risk: {avg_components['risk']:.4f}"
        )
        self.val_history.append({
            'epoch': epoch,
            'nll': avg_components['nll'],
            'kl': avg_components['kl'],
            'risk': avg_components['risk'],
            'total': avg_loss,
        })
        bundle = {'mu': mu, 'sigma': sigma, 'targets': targets}
        return avg_loss, bundle

    def train(self) -> None:
        best_bundle: Optional[Dict[str, torch.Tensor]] = None
        for epoch in range(1, self.args['epochs'] + 1):
            self._run_epoch(epoch)
            val_loss, bundle = self._validate(epoch)
            if val_loss < self.best_loss:
                self.best_loss = val_loss
                self.best_state = {k: v.detach().cpu() for k, v in self.model.state_dict().items()}
                self.not_improved = 0
                best_bundle = bundle
                torch.save(self.best_state, self.best_path)
                self.logger.info('--- New best model (val total loss) ---')
            else:
                self.not_improved += 1
            if self.args.get('early_stop', False) and self.not_improved >= self.args['early_stop_patience']:
                self.logger.info('Early stopping triggered.')
                break
        if self.best_state is not None:
            self.model.load_state_dict(self.best_state)
        if best_bundle is not None:
            self.calibrator.fit(best_bundle['mu'].cpu(), best_bundle['sigma'].cpu(), best_bundle['targets'].cpu())
        if self.val_history:
            kl_values = [entry['kl'] for entry in self.val_history]
            self.logger.info(
                "Validation KL | mean: %.4f min: %.4f max: %.4f"
                % (mean(kl_values), min(kl_values), max(kl_values))
            )

    def test(self) -> Dict[str, float]:
        self.model.eval()
        preds, trues, logvars = [], [], []
        with torch.no_grad():
            for x, y in self.test_loader:
                x = x.to(self.device)
                y = y.squeeze().to(self.device)
                mean, log_var, _ = self.model(x)
                preds.append(mean)
                trues.append(y)
                logvars.append(log_var)

        mu = torch.cat(preds)
        y = torch.cat(trues)
        log_var = torch.cat(logvars)
        sigma = self._to_sigma(log_var)

        nll = float(self.loss_fn.gaussian(mu, y, log_var.exp()).item())
        rmse = torch.sqrt(torch.mean((y - mu) ** 2))
        mae = torch.mean(torch.abs(y - mu))
        ic = self.pearson(y, mu)
        ric = self.ric(y, mu)
        crps = self._crps_gaussian(mu, sigma, y).mean()
        sharp = sigma.mean()
        picp90, gap90 = self._picp_and_gap(mu, sigma, y, q=0.90)
        picp95, gap95 = self._picp_and_gap(mu, sigma, y, q=0.95)
        aurc = self._aurc_rmse(y, mu, sigma, points=10)

        conformal_lo, conformal_hi = self.calibrator.calibrate(mu.cpu(), sigma.cpu())
        coverage = ((y.cpu() >= conformal_lo) & (y.cpu() <= conformal_hi)).float().mean().item()

        row = [nll, rmse.item(), mae.item(), ic.item(), ric.item(), crps.item(), sharp.item(),
               picp90, gap90, picp95, gap95, aurc,
               coverage, self.calibrator.target, self.calibrator.scale_factor,
               self.calibrator.latest_coverage]
        self._append_csv(self.test_csv, row)

        results = {
            'nll': nll,
            'rmse': rmse.item(),
            'mae': mae.item(),
            'ic': ic.item(),
            'ric': ric.item(),
            'crps': crps.item(),
            'sharp': sharp.item(),
            'picp90': picp90,
            'gap90': gap90,
            'picp95': picp95,
            'gap95': gap95,
            'aurc': aurc,
            'coverage': coverage,
            'target_coverage': self.calibrator.target,
            'calibration_scale': self.calibrator.scale_factor,
            'val_coverage': self.calibrator.latest_coverage,
        }
        results['conformal_coverage'] = coverage

        self.logger.info(
            "TEST | RMSE: %.4f MAE: %.4f IC: %.4f RIC: %.4f "
            "NLL: %.4f CRPS: %.4f Sharp: %.4f "
            "PICP90: %.3f Gap90: %.3f PICP95: %.3f Gap95: %.3f "
            "AURC: %.4f ConformalCoverage: %.3f (target %.3f, scale %.2f, val %.3f)"
            % (
                results['rmse'],
                results['mae'],
                results['ic'],
                results['ric'],
                results['nll'],
                results['crps'],
                results['sharp'],
                results['picp90'],
                results['gap90'],
                results['picp95'],
                results['gap95'],
                results['aurc'],
                results['coverage'],
                results['target_coverage'],
                results['calibration_scale'],
                results['val_coverage'],
            )
        )
        return results
