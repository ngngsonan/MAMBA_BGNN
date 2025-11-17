#!/usr/bin/env python3
"""
Validate all 3 datasets (IXIC, DJI, NYSE) for MAMBA_BGNN

This script runs comprehensive validation on all datasets and generates a summary report.
"""

import sys
sys.path.append('.')

from utils.data_processing import data_processing, validate_processed_data
import json
from datetime import datetime


def validate_single_dataset(dataset_name, window=5, batch_size=32):
    """Validate a single dataset"""
    print(f"\n{'='*60}")
    print(f"VALIDATING: {dataset_name}")
    print(f"{'='*60}")

    data_path = f'Dataset/combined_dataframe_{dataset_name}.csv'

    try:
        # Step 1: Process data
        print(f"\n📊 Processing {dataset_name} data...")
        num_features, train_loader, val_loader, test_loader = data_processing(
            data_path=data_path,
            window=window,
            batch_size=batch_size
        )
        print(f"   ✅ Features: {num_features}")

        # Step 2: Validate
        print(f"\n🔍 Running validation checks...")
        report = validate_processed_data(
            train_loader=train_loader,
            val_loader=val_loader,
            test_loader=test_loader,
            verbose=True
        )

        # Add metadata
        report['dataset'] = dataset_name
        report['num_features'] = num_features
        report['window'] = window
        report['batch_size'] = batch_size

        return report

    except FileNotFoundError:
        print(f"\n❌ Dataset file not found: {data_path}")
        return {
            'dataset': dataset_name,
            'all_passed': False,
            'error': 'File not found',
            'results': {}
        }
    except Exception as e:
        print(f"\n❌ Error validating {dataset_name}: {e}")
        import traceback
        traceback.print_exc()
        return {
            'dataset': dataset_name,
            'all_passed': False,
            'error': str(e),
            'results': {}
        }


def print_summary(all_reports):
    """Print summary of all validations"""
    print("\n" + "="*60)
    print("VALIDATION SUMMARY - ALL DATASETS")
    print("="*60)

    for report in all_reports:
        dataset = report.get('dataset', 'Unknown')
        passed = report.get('all_passed', False)

        if 'error' in report:
            status = f"❌ ERROR: {report['error']}"
        elif passed:
            status = "✅ PASSED"
        else:
            status = "❌ FAILED"

        print(f"\n{dataset:10s}: {status}")

        if not report.get('error') and 'results' in report:
            for check_name, check_passed in report['results'].items():
                check_status = "✅" if check_passed else "❌"
                print(f"   {check_status} {check_name.replace('_', ' ').title()}")

    # Overall summary
    print("\n" + "="*60)
    passed_count = sum(1 for r in all_reports if r.get('all_passed', False))
    total_count = len(all_reports)

    print(f"Overall: {passed_count}/{total_count} datasets passed all checks")

    if passed_count == total_count:
        print("\n🎉 SUCCESS: All datasets are ready for training!")
        return True
    else:
        print("\n⚠️  WARNING: Some datasets have issues")
        print("Please review the validation results above before training")
        return False


def save_validation_report(all_reports, filename='validation_report.json'):
    """Save validation report to JSON file"""
    report_data = {
        'timestamp': datetime.now().isoformat(),
        'datasets': all_reports,
        'summary': {
            'total_datasets': len(all_reports),
            'passed_datasets': sum(1 for r in all_reports if r.get('all_passed', False)),
            'failed_datasets': sum(1 for r in all_reports if not r.get('all_passed', False))
        }
    }

    with open(filename, 'w') as f:
        json.dump(report_data, f, indent=2)

    print(f"\n💾 Validation report saved to: {filename}")


def main():
    print("="*60)
    print("MAMBA_BGNN - Comprehensive Data Validation")
    print("="*60)
    print("Validating all 3 datasets: IXIC, DJI, NYSE")
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # Configuration
    datasets = ['IXIC', 'DJI', 'NYSE']
    window = 5
    batch_size = 32

    print(f"\nConfiguration:")
    print(f"   Window size: {window}")
    print(f"   Batch size: {batch_size}")

    # Validate each dataset
    all_reports = []

    for i, dataset in enumerate(datasets, 1):
        print(f"\n{'#'*60}")
        print(f"Dataset {i}/{len(datasets)}: {dataset}")
        print(f"{'#'*60}")

        report = validate_single_dataset(
            dataset_name=dataset,
            window=window,
            batch_size=batch_size
        )
        all_reports.append(report)

    # Print summary
    success = print_summary(all_reports)

    # Save report
    save_validation_report(all_reports)

    # Return status
    return success


if __name__ == "__main__":
    print("\n")
    success = main()

    print("\n" + "="*60)
    if success:
        print("VALIDATION COMPLETE: All datasets ready for training")
        print("="*60)
        sys.exit(0)
    else:
        print("VALIDATION COMPLETE: Please review issues above")
        print("="*60)
        sys.exit(1)
