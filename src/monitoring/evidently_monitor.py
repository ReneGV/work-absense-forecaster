import pandas as pd
import numpy as np
import json
from datetime import datetime
from pathlib import Path

from evidently import Report
from evidently.presets import DataDriftPreset, ClassificationPreset

# Default thresholds for monitoring
DEFAULT_THRESHOLDS = {
    'accuracy_target': 0.85,
    'f1_target': 0.81,
    'psi_warning': 0.10,
    'psi_critical': 0.25,
    'min_samples': 100
}


class AbsenteeismMonitor:
    """
    Monitor for tracking model performance and data drift.
    
    Metrics:
    - Accuracy: Classification performance
    - F1 Score: Balance of precision and recall
    - PSI: Population Stability Index for drift detection
    """
    
    def __init__(self, baseline_data_path, output_dir='reports/monitoring'):
        """
        Initialize the monitor with baseline data.
        
        Args:
            baseline_data_path: Path to training data CSV
            output_dir: Where to save reports
        """
        self.reference_data = pd.read_csv(baseline_data_path)
        self.thresholds = DEFAULT_THRESHOLDS
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"Thresholds: Accuracy={self.thresholds['accuracy_target']:.0%}, "
              f"F1={self.thresholds['f1_target']:.0%}, "
              f"PSI Warning={self.thresholds['psi_warning']}, "
              f"PSI Critical={self.thresholds['psi_critical']}")
    
    def calculate_psi(self, expected, actual, bins=10):
        """
        Calculate Population Stability Index (PSI) to detect distribution shifts.
        
        PSI Interpretation:
        - < 0.10: No significant change
        - 0.10-0.25: Moderate change
        - > 0.25: Significant change
        """
        if len(expected) == 0 or len(actual) == 0:
            return 0.0
        
        # Create bins based on expected distribution
        breakpoints = np.percentile(expected, np.linspace(0, 100, bins + 1))
        breakpoints = np.unique(breakpoints)
        
        if len(breakpoints) <= 2:
            return 0.0
        
        # Calculate frequencies for both distributions
        expected_freq = np.histogram(expected, bins=breakpoints)[0] / len(expected)
        actual_freq = np.histogram(actual, bins=breakpoints)[0] / len(actual)
        
        # Avoid division by zero
        epsilon = 1e-10
        expected_freq = expected_freq + epsilon
        actual_freq = actual_freq + epsilon
        
        # PSI formula: sum((actual% - expected%) * ln(actual% / expected%))
        psi = np.sum((actual_freq - expected_freq) * np.log(actual_freq / expected_freq))
        
        return psi
    
    def check_drift(self, current_data_path):
        """
        Check for data drift by comparing current data with baseline.
        
        Returns dictionary with drift metrics and status.
        """
        current_data = pd.read_csv(current_data_path)
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        
        print(f"\n=== Drift Check: {timestamp} ===")
        
        # Validate sample size
        min_samples = self.thresholds['min_samples']
        if len(current_data) < min_samples:
            print(f"Error: Need at least {min_samples} samples, got {len(current_data)}")
            return {'status': 'error', 'message': 'Insufficient samples'}
        
        print(f"Analyzing {len(current_data)} samples...")
        
        # Calculate PSI for each numeric feature
        numeric_features = self.reference_data.select_dtypes(include=[np.number]).columns
        common_features = [col for col in numeric_features if col in current_data.columns]
        
        psi_scores = {}
        drifted_features = []
        
        print(f"\nPSI Scores:")
        for feature in common_features:
            try:
                psi = self.calculate_psi(
                    self.reference_data[feature],
                    current_data[feature]
                )
                psi_scores[feature] = psi
                
                # Check thresholds
                if psi > self.thresholds['psi_critical']:
                    status = "CRITICAL"
                    drifted_features.append(feature)
                elif psi > self.thresholds['psi_warning']:
                    status = "WARNING"
                    drifted_features.append(feature)
                else:
                    status = "OK"
                
                print(f"  {feature}: {psi:.4f} ({status})")
            except Exception as e:
                print(f"  {feature}: Error - {str(e)}")
        
        # Summary statistics
        avg_psi = np.mean(list(psi_scores.values())) if psi_scores else 0
        max_psi = max(psi_scores.values()) if psi_scores else 0
        drift_share = len(drifted_features) / len(common_features) if common_features else 0
        
        print(f"\nSummary:")
        print(f"  Average PSI: {avg_psi:.4f}")
        print(f"  Max PSI: {max_psi:.4f}")
        print(f"  Drifted: {len(drifted_features)}/{len(common_features)} features")
        
        # Determine status based on PSI
        if max_psi > self.thresholds['psi_critical']:
            status = 'critical'
        elif max_psi > self.thresholds['psi_warning']:
            status = 'warning'
        else:
            status = 'passed'
        
        print(f"\nStatus: {status.upper()}")
        
        # Generate Evidently HTML report
        report = Report(metrics=[DataDriftPreset()])
        snapshot = report.run(reference_data=self.reference_data, current_data=current_data)
        
        report_file = self.output_dir / f'drift_report_{datetime.now().strftime("%Y%m%d_%H%M%S")}.html'
        snapshot.save_html(str(report_file))
        print(f"Report saved: {report_file}")
        
        # Create result dictionary
        result = {
            'timestamp': timestamp,
            'status': status,
            'max_psi': max_psi,
            'avg_psi': avg_psi,
            'drift_share': drift_share,
            'drifted_features': drifted_features,
            'report_file': str(report_file)
        }
        
        # Save to log
        self._save_log(result)
        
        return result
        
    def _save_log(self, result):
        """Save monitoring result to JSON log file."""
        log_file = self.output_dir / 'monitoring_log.jsonl'
        with open(log_file, 'a') as f:
            f.write(json.dumps(result) + '\n')


if __name__ == "__main__":    
    print("\n=== Absenteeism Monitoring System ===\n")
    
    # Initialize monitor with baseline data
    monitor = AbsenteeismMonitor('data/raw/work_absenteeism_original.csv')
    monitor.check_drift('data/raw/work_absenteeism_modified.csv')
