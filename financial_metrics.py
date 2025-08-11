"""
Comprehensive financial evaluation metrics for stock prediction models.
Addresses reviewer concerns about financial relevance and practical applicability.
"""

import torch
import numpy as np
import pandas as pd
from typing import Dict, Tuple, List
import warnings
from scipy import stats


class FinancialMetrics:
    """Comprehensive financial evaluation metrics"""
    
    @staticmethod
    def directional_accuracy(y_pred: torch.Tensor, y_true: torch.Tensor) -> float:
        """
        Directional accuracy - correctly predicted sign of returns
        Critical metric for trading applications
        """
        pred_direction = torch.sign(y_pred)
        true_direction = torch.sign(y_true)
        
        # Handle zero returns (no direction change)
        mask = (true_direction != 0) & (pred_direction != 0)
        if mask.sum() == 0:
            return 0.5  # Random baseline
            
        correct = (pred_direction[mask] == true_direction[mask]).float()
        return correct.mean().item()
    
    @staticmethod
    def sharpe_ratio(returns: torch.Tensor, risk_free_rate: float = 0.02) -> float:
        """
        Sharpe ratio - risk-adjusted return measure
        Higher is better
        """
        returns_np = returns.detach().cpu().numpy()
        
        if len(returns_np) == 0:
            return 0.0
            
        daily_rf = risk_free_rate / 252  # Assuming daily returns
        excess_returns = returns_np - daily_rf
        
        if np.std(excess_returns) == 0:
            return 0.0
            
        return np.mean(excess_returns) / np.std(excess_returns) * np.sqrt(252)
    
    @staticmethod
    def maximum_drawdown(returns: torch.Tensor) -> float:
        """
        Maximum drawdown - worst peak-to-trough decline
        Lower (more negative) is worse
        """
        returns_np = returns.detach().cpu().numpy()
        cumulative = np.cumprod(1 + returns_np)
        
        # Calculate running maximum
        running_max = np.maximum.accumulate(cumulative)
        
        # Calculate drawdown
        drawdown = (cumulative - running_max) / running_max
        
        return float(np.min(drawdown))
    
    @staticmethod
    def calmar_ratio(returns: torch.Tensor) -> float:
        """
        Calmar ratio - annual return divided by maximum drawdown
        Higher is better
        """
        annual_return = FinancialMetrics.sharpe_ratio(returns, 0.0) * np.sqrt(252) * np.std(returns.detach().cpu().numpy()) / np.sqrt(252)
        max_dd = abs(FinancialMetrics.maximum_drawdown(returns))
        
        if max_dd == 0:
            return float('inf') if annual_return > 0 else 0.0
            
        return annual_return / max_dd
    
    @staticmethod
    def information_ratio(y_pred: torch.Tensor, y_true: torch.Tensor, 
                         benchmark_returns: torch.Tensor = None) -> float:
        """
        Information ratio - active return divided by tracking error
        """
        if benchmark_returns is None:
            benchmark_returns = torch.zeros_like(y_true)
            
        active_returns = y_pred - benchmark_returns
        tracking_error = torch.std(active_returns)
        
        if tracking_error == 0:
            return 0.0
            
        return torch.mean(active_returns).item() / tracking_error.item()
    
    @staticmethod
    def hit_rate(y_pred: torch.Tensor, y_true: torch.Tensor) -> float:
        """
        Hit rate - percentage of predictions with same sign as actual
        Same as directional accuracy but clearer name
        """
        return FinancialMetrics.directional_accuracy(y_pred, y_true)
    
    @staticmethod
    def profit_and_loss(y_pred: torch.Tensor, y_true: torch.Tensor, 
                       transaction_cost: float = 0.001) -> Dict[str, float]:
        """
        Simulated P&L from trading strategy based on predictions
        """
        # Simple strategy: long if predicted positive, short if negative
        positions = torch.sign(y_pred).detach().cpu().numpy()
        returns = y_true.detach().cpu().numpy()
        
        # Calculate strategy returns
        strategy_returns = positions * returns
        
        # Apply transaction costs (assuming position changes incur costs)
        position_changes = np.abs(np.diff(np.concatenate([[0], positions])))
        total_costs = np.sum(position_changes) * transaction_cost
        
        # Calculate cumulative P&L
        cumulative_returns = np.cumprod(1 + strategy_returns) - 1
        total_return = cumulative_returns[-1] if len(cumulative_returns) > 0 else 0
        
        # Adjust for transaction costs
        net_return = total_return - total_costs
        
        return {
            'total_return': float(total_return),
            'net_return': float(net_return),
            'transaction_costs': float(total_costs),
            'volatility': float(np.std(strategy_returns)) if len(strategy_returns) > 0 else 0.0,
            'max_drawdown': FinancialMetrics.maximum_drawdown(torch.tensor(strategy_returns))
        }
    
    @staticmethod
    def tail_ratio(returns: torch.Tensor, threshold: float = 0.05) -> float:
        """
        Tail ratio - ratio of average positive tail to average negative tail
        Higher is better
        """
        returns_np = returns.detach().cpu().numpy()
        
        # Calculate quantiles
        upper_threshold = np.percentile(returns_np, (1 - threshold) * 100)
        lower_threshold = np.percentile(returns_np, threshold * 100)
        
        # Calculate tail averages
        upper_tail = returns_np[returns_np >= upper_threshold]
        lower_tail = returns_np[returns_np <= lower_threshold]
        
        if len(upper_tail) == 0 or len(lower_tail) == 0:
            return 1.0
            
        upper_avg = np.mean(upper_tail)
        lower_avg = np.mean(lower_tail)
        
        if lower_avg >= 0:  # Handle edge case
            return float('inf') if upper_avg > 0 else 1.0
            
        return abs(upper_avg / lower_avg)


class MarketRegimeAnalysis:
    """Analyze model performance across different market conditions"""
    
    @staticmethod
    def classify_market_regime(returns: torch.Tensor, window: int = 20) -> torch.Tensor:
        """
        Classify market regime based on rolling volatility
        0: Low volatility (stable)
        1: High volatility (volatile)
        """
        returns_np = returns.detach().cpu().numpy()
        
        # Calculate rolling volatility
        rolling_vol = pd.Series(returns_np).rolling(window=window, min_periods=window//2).std()
        vol_median = rolling_vol.median()
        
        # Classify: above median = volatile (1), below = stable (0)
        regime = (rolling_vol > vol_median).astype(int).values
        
        return torch.tensor(regime, dtype=torch.int)
    
    @staticmethod
    def regime_performance_analysis(y_pred: torch.Tensor, y_true: torch.Tensor, 
                                   regimes: torch.Tensor) -> Dict[str, Dict[str, float]]:
        """
        Analyze performance metrics across different market regimes
        """
        results = {}
        
        for regime_id in [0, 1]:  # 0: stable, 1: volatile
            regime_name = "Stable" if regime_id == 0 else "Volatile"
            mask = (regimes == regime_id)
            
            if mask.sum() == 0:
                continue
                
            pred_regime = y_pred[mask]
            true_regime = y_true[mask]
            
            # Calculate metrics for this regime
            metrics = FinancialMetrics()
            results[regime_name] = {
                'samples': int(mask.sum().item()),
                'rmse': float(torch.sqrt(torch.mean((pred_regime - true_regime)**2))),
                'mae': float(torch.mean(torch.abs(pred_regime - true_regime))),
                'directional_accuracy': metrics.directional_accuracy(pred_regime, true_regime),
                'sharpe_ratio': metrics.sharpe_ratio(pred_regime),
                'correlation': float(torch.corrcoef(torch.stack([pred_regime, true_regime]))[0, 1]) if len(pred_regime) > 1 else 0.0
            }
            
        return results
    
    @staticmethod
    def market_stress_test(y_pred: torch.Tensor, y_true: torch.Tensor,
                          stress_percentile: float = 5.0) -> Dict[str, float]:
        """
        Test model performance during market stress periods
        """
        returns_np = y_true.detach().cpu().numpy()
        
        # Identify stress periods (extreme negative returns)
        stress_threshold = np.percentile(returns_np, stress_percentile)
        stress_mask = y_true <= stress_threshold
        
        if stress_mask.sum() == 0:
            return {'stress_samples': 0}
            
        pred_stress = y_pred[stress_mask]
        true_stress = y_true[stress_mask]
        
        metrics = FinancialMetrics()
        
        return {
            'stress_samples': int(stress_mask.sum().item()),
            'stress_rmse': float(torch.sqrt(torch.mean((pred_stress - true_stress)**2))),
            'stress_directional_accuracy': metrics.directional_accuracy(pred_stress, true_stress),
            'stress_correlation': float(torch.corrcoef(torch.stack([pred_stress, true_stress]))[0, 1]) if len(pred_stress) > 1 else 0.0,
            'average_stress_return': float(torch.mean(true_stress))
        }


def comprehensive_evaluation(y_pred: torch.Tensor, y_true: torch.Tensor, 
                           model_name: str = "Model", 
                           save_path: str = None) -> Dict[str, float]:
    """
    Comprehensive evaluation including all financial metrics
    """
    metrics = FinancialMetrics()
    
    # Basic metrics
    rmse = float(torch.sqrt(torch.mean((y_pred - y_true)**2)))
    mae = float(torch.mean(torch.abs(y_pred - y_true)))
    correlation = float(torch.corrcoef(torch.stack([y_pred, y_true]))[0, 1]) if len(y_pred) > 1 else 0.0
    
    # Financial metrics
    directional_acc = metrics.directional_accuracy(y_pred, y_true)
    sharpe = metrics.sharpe_ratio(y_pred)
    max_dd = metrics.maximum_drawdown(y_pred)
    calmar = metrics.calmar_ratio(y_pred)
    info_ratio = metrics.information_ratio(y_pred, y_true)
    hit_rate = metrics.hit_rate(y_pred, y_true)
    tail_ratio = metrics.tail_ratio(y_pred)
    
    # P&L analysis
    pnl_results = metrics.profit_and_loss(y_pred, y_true)
    
    # Market regime analysis
    regimes = MarketRegimeAnalysis.classify_market_regime(y_true)
    regime_analysis = MarketRegimeAnalysis.regime_performance_analysis(y_pred, y_true, regimes)
    
    # Stress test
    stress_results = MarketRegimeAnalysis.market_stress_test(y_pred, y_true)
    
    # Compile all results
    results = {
        'model_name': model_name,
        'samples': len(y_pred),
        
        # Basic metrics
        'rmse': rmse,
        'mae': mae,
        'correlation': correlation,
        
        # Financial metrics
        'directional_accuracy': directional_acc,
        'sharpe_ratio': sharpe,
        'maximum_drawdown': max_dd,
        'calmar_ratio': calmar,
        'information_ratio': info_ratio,
        'hit_rate': hit_rate,
        'tail_ratio': tail_ratio,
        
        # P&L metrics
        'total_return': pnl_results['total_return'],
        'net_return': pnl_results['net_return'],
        'transaction_costs': pnl_results['transaction_costs'],
        'strategy_volatility': pnl_results['volatility'],
        'strategy_max_drawdown': pnl_results['max_drawdown'],
        
        # Regime analysis
        **{f"regime_{k.lower()}_{metric}": v for k, regime_metrics in regime_analysis.items() 
           for metric, v in regime_metrics.items()},
        
        # Stress test
        **{f"stress_{k}": v for k, v in stress_results.items()}
    }
    
    # Save results if path provided
    if save_path:
        pd.DataFrame([results]).to_csv(save_path, index=False)
    
    return results


if __name__ == "__main__":
    # Test financial metrics
    torch.manual_seed(42)
    
    # Generate sample data
    y_true = torch.randn(1000) * 0.02  # Daily returns ~2% volatility
    y_pred = y_true + torch.randn(1000) * 0.01  # Add some prediction error
    
    # Run comprehensive evaluation
    results = comprehensive_evaluation(y_pred, y_true, "Test Model")
    
    print("Comprehensive Financial Evaluation Results:")
    print("=" * 50)
    
    for key, value in results.items():
        if isinstance(value, (int, float)):
            if 'ratio' in key or 'accuracy' in key or 'correlation' in key:
                print(f"{key:25}: {value:.4f}")
            elif 'return' in key or 'drawdown' in key:
                print(f"{key:25}: {value:.2%}")
            else:
                print(f"{key:25}: {value:.6f}")
        else:
            print(f"{key:25}: {value}")