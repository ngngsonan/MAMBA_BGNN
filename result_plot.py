import os, math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def load_csv(path):
    return pd.read_csv(path)

def load_preds(path):
    return pd.read_csv(path)   # y, mu, sigma

def plot_learning_curves(val_df, out_dir):
    # NLL & CRPS (twin y-axis)
    fig, ax1 = plt.subplots()
    ax1.plot(val_df['epoch'], val_df['nll'], label='Val NLL')
    ax1.set_xlabel('Epoch'); ax1.set_ylabel('NLL')
    ax2 = ax1.twinx()
    ax2.plot(val_df['epoch'], val_df['crps'], '--', label='Val CRPS')
    ax2.set_ylabel('CRPS')
    fig.suptitle('Learning Curves (Probabilistic)')
    lines, labels = [], []
    for ax in (ax1, ax2):
        h, l = ax.get_legend_handles_labels(); lines += h; labels += l
    ax1.legend(lines, labels, loc='best')
    fig.tight_layout(); fig.savefig(os.path.join(out_dir, 'curve_nll_crps.png')); plt.close(fig)

    # RMSE / MAE
    plt.figure()
    plt.plot(val_df['epoch'], val_df['rmse'], label='Val RMSE')
    plt.plot(val_df['epoch'], val_df['mae'],  label='Val MAE')
    plt.xlabel('Epoch'); plt.ylabel('Error'); plt.legend()
    plt.title('Learning Curves (Point)')
    plt.tight_layout(); plt.savefig(os.path.join(out_dir, 'curve_rmse_mae.png')); plt.close()

    # Coverage gap (90/95)
    plt.figure()
    plt.plot(val_df['epoch'], val_df['gap90'], label='Gap@90%')
    plt.plot(val_df['epoch'], val_df['gap95'], label='Gap@95%')
    plt.xlabel('Epoch'); plt.ylabel('|Observed - Nominal|')
    plt.legend(); plt.title('Coverage Gap (Calibration)')
    plt.tight_layout(); plt.savefig(os.path.join(out_dir, 'curve_coverage_gap.png')); plt.close()

def plot_picp_from_preds(test_preds_csv, out_dir):
    d = load_preds(test_preds_csv)
    y, mu, sigma = d['y'].values, d['mu'].values, np.maximum(d['sigma'].values, 1e-12)
    qs = [0.5, 0.8, 0.9, 0.95]
    nom, obs = [], []
    for q in qs:
        p = (1.0 + q)/2.0
        from scipy.special import erfinv
        z = np.sqrt(2.0)*erfinv(2.0*p - 1.0)
        lo, hi = mu - z*sigma, mu + z*sigma
        oc = np.mean((y >= lo) & (y <= hi))
        nom.append(q); obs.append(oc)
    plt.figure()
    plt.plot(nom, nom, 'k--', label='Ideal')
    plt.plot(nom, obs, 'o-', label='Observed')
    plt.xlabel('Nominal coverage'); plt.ylabel('Observed coverage')
    plt.legend(); plt.title('Calibration: PICP (Test)')
    plt.tight_layout(); plt.savefig(os.path.join(out_dir, 'calib_picp.png')); plt.close()

def plot_risk_coverage(test_preds_csv, out_dir):
    d = load_preds(test_preds_csv)
    y, mu, sigma = d['y'].values, d['mu'].values, np.maximum(d['sigma'].values, 1e-12)
    idx = np.argsort(sigma)  # most certain first
    y, mu = y[idx], mu[idx]
    covs = np.linspace(0.1, 1.0, 10)
    rmses = []
    for c in covs:
        k = max(1, int(c*len(y)))
        rmses.append(np.sqrt(np.mean((y[:k]-mu[:k])**2)))
    # AURC (trapezoid)
    aurc = 0.0
    for i in range(1, len(covs)):
        h = covs[i] - covs[i-1]
        aurc += 0.5*h*(rmses[i] + rmses[i-1])
    plt.figure()
    plt.plot(covs, rmses, 'o-')
    plt.xlabel('Coverage'); plt.ylabel('RMSE')
    plt.title(f'Risk–Coverage (Test)  AURC={aurc:.5f}')
    plt.tight_layout(); plt.savefig(os.path.join(out_dir, 'risk_coverage.png')); plt.close()

def plot_ic_by_sigma_decile(test_preds_csv, out_dir):
    d = load_preds(test_preds_csv)
    y, mu, sigma = d['y'].values, d['mu'].values, np.maximum(d['sigma'].values, 1e-12)
    dec = pd.qcut(sigma, 10, labels=False, duplicates='drop')
    ic_vals = []
    for g in np.unique(dec):
        m = (dec == g)
        yy, mm = y[m], mu[m]
        if len(yy) > 2:
            vx, vy = yy - yy.mean(), mm - mm.mean()
            ic = (vx*vy).sum() / (np.sqrt((vx**2).sum())*np.sqrt((vy**2).sum()) + 1e-12)
            ic_vals.append(ic)
        else:
            ic_vals.append(np.nan)
    plt.figure()
    plt.plot(range(1, len(ic_vals)+1), ic_vals, 'o-')
    plt.xlabel('Sigma decile (1=lowest uncertainty)'); plt.ylabel('IC')
    plt.title('IC by sigma decile (Test)')
    plt.tight_layout(); plt.savefig(os.path.join(out_dir, 'ic_by_sigma_decile.png')); plt.close()

def plot_analytics(log_dir):
    os.makedirs(log_dir, exist_ok=True)
    val_df = load_csv(os.path.join(log_dir, 'val_metrics.csv'))
    plot_learning_curves(val_df, log_dir)

    test_preds_csv = os.path.join(log_dir, 'test_predictions.csv')
    if os.path.exists(test_preds_csv):
        plot_picp_from_preds(test_preds_csv, log_dir)
        plot_risk_coverage(test_preds_csv, log_dir)
        plot_ic_by_sigma_decile(test_preds_csv, log_dir)

    # In hàng TEST tổng hợp nếu có
    test_csv = os.path.join(log_dir, 'test_metrics.csv')
    if os.path.exists(test_csv):
        df_test = load_csv(test_csv)
        print('TEST summary:\n', df_test.tail(1).to_string(index=False))

# if __name__ == '__main__':
#     import argparse
#     ap = argparse.ArgumentParser()
#     ap.add_argument('--log_dir', type=str, required=True)
#     args = ap.parse_args()
#     main(args.log_dir)
