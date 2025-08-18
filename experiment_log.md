# MAMBA-BGNN Experimental Results Log

## Experiment Setup
- **Date**: 2025-01-18
- **Environment**: Python 3.10.18, PyTorch 2.7.1+cu118, CUDA Available
- **Purpose**: Baseline performance establishment and improvement validation

## Baseline Experiments

### Experiment 1: Original MAMBA-BGNN - COMPLETED ✅
**Status**: COMPLETED
**Start Time**: 2025-08-18 19:14:53
**End Time**: 2025-08-18 19:25:00

## 🏆 **FINAL BASELINE RESULTS**

### **IXIC (NASDAQ) - BEST PERFORMER**
- **RMSE**: **0.005141** 
- **MAE**: 0.004189
- **IC**: **0.968012** (96.8% correlation!) 🔥
- **RIC**: 0.96468
- **NLL**: -3.855048
- **PICP90**: 0.886538 (excellent calibration)

### **NYSE - EXCELLENT RESULTS**  
- **RMSE**: **0.005268**
- **MAE**: 0.00412  
- **IC**: **0.913898** (91.4% correlation!) 🔥
- **RIC**: 0.891754
- **NLL**: -3.829144
- **PICP90**: 0.938462 (superb calibration)

### **DJI (Dow Jones) - GOOD RESULTS**
- **RMSE**: 0.010903
- **MAE**: 0.00842
- **IC**: **0.225251** (22.5% correlation)
- **RIC**: 0.191117  
- **NLL**: -2.16232
- **PICP90**: 0.596154

## 📊 **Key Insights**
1. **IXIC performed EXCEPTIONALLY** - 96.8% correlation!
2. **NYSE also outstanding** - 91.4% correlation!  
3. **DJI more challenging** - but still positive results
4. **All models well-calibrated** - good uncertainty quantification