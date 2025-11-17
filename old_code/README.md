# Old Code Archive

Thư mục này chứa các file phân tích và validation cũ đã không còn sử dụng trong pipeline chính.

## Files trong thư mục này:

### Data Analysis & Validation:
- `clean_data_processing.py` - Phiên bản cũ của data processing
- `data_leakage_analysis.py` - Công cụ phân tích data leakage
- `feature_analysis.py` - Phân tích features và correlations
- `temporal_alignment_analysis.py` - Kiểm tra temporal alignment
- `test_framework.py` - Framework test cũ

### Evaluation Tools:
- `demo_evaluation.py` - Demo evaluation
- `proper_ic_ric_calculation.py` - Tính toán IC/RIC
- `run_comprehensive_evaluation.py` - Comprehensive evaluation runner

### Validation Scripts:
- `feature_analysis_cell.py` - Cell phân tích features cho notebook
- `validate_all_datasets.py` - Script validate tất cả datasets
- `validation_cell.py` - Cell validation cho notebook

## Files đang sử dụng (trong utils/):

### Core Files:
- `data_processing.py` - **FILE CHÍNH** - Data processing pipeline
- `comprehensive_evaluation.py` - Comprehensive evaluation
- `financial_metrics.py` - Financial metrics calculation
- `data_validation.py` - Data validation utilities

## Cách sử dụng:

**File chính để sử dụng**: `utils/data_processing.py`

**File test**: `test_data_processing.py` (trong root directory)

Các file trong thư mục này chỉ nên dùng để tham khảo hoặc debugging.
