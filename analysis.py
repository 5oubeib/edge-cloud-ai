"""
Analysis Script - Generate plots and statistics from experiment results
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
from datetime import datetime
from config import PATHS, PLOT_SETTINGS

# Set plotting style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")


class ResultsAnalyzer:
    """Analyze and visualize experiment results"""
    
    def __init__(self, results_file=None):
        """
        Initialize analyzer
        
        Args:
            results_file: Path to CSV results file (or None to use latest)
        """
        print("="*70)
        print("RESULTS ANALYZER")
        print("="*70)
        
        # Load results
        if results_file is None:
            results_file = self._find_latest_results()
        
        print(f"\nLoading results from: {results_file}")
        self.df = pd.read_csv(results_file)
        print(f"  Loaded {len(self.df)} records")
        
        # Create output directory for plots
        self.plot_dir = os.path.join(PATHS['results'], 'plots')
        os.makedirs(self.plot_dir, exist_ok=True)
        
        print(f"  Plots will be saved to: {self.plot_dir}")
    
    def _find_latest_results(self):
        """Find the most recent results file"""
        results_dir = PATHS['results']
        files = [f for f in os.listdir(results_dir) if f.startswith('experiment_results_') and f.endswith('.csv')]
        
        if not files:
            raise FileNotFoundError("No results files found. Run experiment_runner.py first!")
        
        latest = max(files)
        return os.path.join(results_dir, latest)
    
    def print_summary_statistics(self):
        """Print comprehensive summary statistics"""
        print("\n" + "="*70)
        print("SUMMARY STATISTICS")
        print("="*70)
        
        # Overall statistics by strategy
        print("\n### Overall Performance by Strategy ###")
        summary = self.df.groupby('strategy').agg({
            'prediction_correct': ['mean', 'std'],
            'total_latency_ms': ['mean', 'std', 'median'],
            'energy_consumed_mah': ['sum', 'mean']
        }).round(3)
        print(summary)
        
        # Network scenario analysis
        print("\n### Performance by Network Condition ###")
        network_summary = self.df.groupby(['network_condition', 'strategy']).agg({
            'prediction_correct': 'mean',
            'total_latency_ms': 'mean',
            'energy_consumed_mah': 'mean'
        }).round(3)
        print(network_summary)
        
        # Placement decision distribution
        print("\n### Placement Decision Distribution ###")
        placement = pd.crosstab(self.df['strategy'], self.df['placement_decision'], normalize='index') * 100
        print(placement.round(1))
    
    def plot_accuracy_comparison(self):
        """Plot accuracy comparison across strategies"""
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # Plot 1: Overall accuracy by strategy
        strategy_acc = self.df.groupby('strategy')['prediction_correct'].mean() * 100
        ax = axes[0]
        strategy_acc.plot(kind='bar', ax=ax, color='skyblue')
        ax.set_title('Overall Accuracy by Strategy', fontsize=14, fontweight='bold')
        ax.set_ylabel('Accuracy (%)', fontsize=12)
        ax.set_xlabel('Strategy', fontsize=12)
        ax.axhline(y=90, color='r', linestyle='--', label='Target: 90%')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Plot 2: Accuracy by network condition
        ax = axes[1]
        network_data = self.df.groupby(['network_condition', 'strategy'])['prediction_correct'].mean().unstack() * 100
        network_data.plot(kind='bar', ax=ax)
        ax.set_title('Accuracy by Network Condition', fontsize=14, fontweight='bold')
        ax.set_ylabel('Accuracy (%)', fontsize=12)
        ax.set_xlabel('Network Condition', fontsize=12)
        ax.legend(title='Strategy')
        ax.grid(True, alpha=0.3)
        
        # Plot 3: Accuracy by accuracy requirement
        ax = axes[2]
        acc_req_data = self.df.groupby(['accuracy_requirement', 'strategy'])['prediction_correct'].mean().unstack() * 100
        acc_req_data.plot(kind='bar', ax=ax)
        ax.set_title('Accuracy by Requirement Level', fontsize=14, fontweight='bold')
        ax.set_ylabel('Accuracy (%)', fontsize=12)
        ax.set_xlabel('Accuracy Requirement', fontsize=12)
        ax.legend(title='Strategy')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        filename = os.path.join(self.plot_dir, 'accuracy_comparison.png')
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"  ✓ Saved: {filename}")
        plt.close()
    
    def plot_latency_comparison(self):
        """Plot latency comparison"""
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Plot 1: Box plot of latency by strategy
        ax = axes[0]
        self.df.boxplot(column='total_latency_ms', by='strategy', ax=ax)
        ax.set_title('Latency Distribution by Strategy', fontsize=14, fontweight='bold')
        ax.set_ylabel('Latency (ms)', fontsize=12)
        ax.set_xlabel('Strategy', fontsize=12)
        ax.axhline(y=500, color='r', linestyle='--', label='Target: 500ms')
        plt.suptitle('')  # Remove default title
        
        # Plot 2: Latency by network condition
        ax = axes[1]
        network_latency = self.df.groupby(['network_condition', 'strategy'])['total_latency_ms'].mean().unstack()
        network_latency.plot(kind='bar', ax=ax)
        ax.set_title('Avg Latency by Network Condition', fontsize=14, fontweight='bold')
        ax.set_ylabel('Latency (ms)', fontsize=12)
        ax.set_xlabel('Network Condition', fontsize=12)
        ax.legend(title='Strategy')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        filename = os.path.join(self.plot_dir, 'latency_comparison.png')
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"  ✓ Saved: {filename}")
        plt.close()
    
    def plot_energy_comparison(self):
        """Plot energy consumption comparison"""
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Plot 1: Total energy by strategy
        ax = axes[0]
        energy_total = self.df.groupby('strategy')['energy_consumed_mah'].sum()
        energy_total.plot(kind='bar', ax=ax, color='coral')
        ax.set_title('Total Energy Consumption by Strategy', fontsize=14, fontweight='bold')
        ax.set_ylabel('Energy (mAh)', fontsize=12)
        ax.set_xlabel('Strategy', fontsize=12)
        ax.grid(True, alpha=0.3)
        
        # Plot 2: Energy by battery level scenario
        ax = axes[1]
        battery_scenario_df = self.df[self.df['scenario'].str.contains('battery')]
        if not battery_scenario_df.empty:
            battery_energy = battery_scenario_df.groupby(['scenario', 'strategy'])['energy_consumed_mah'].sum().unstack()
            battery_energy.plot(kind='bar', ax=ax)
            ax.set_title('Energy by Battery Level Scenario', fontsize=14, fontweight='bold')
            ax.set_ylabel('Energy (mAh)', fontsize=12)
            ax.set_xlabel('Battery Scenario', fontsize=12)
            ax.legend(title='Strategy')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        filename = os.path.join(self.plot_dir, 'energy_comparison.png')
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"  ✓ Saved: {filename}")
        plt.close()
    
    def plot_tradeoff_analysis(self):
        """Plot accuracy-energy-latency trade-offs"""
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        
        # Calculate metrics by strategy
        metrics = self.df.groupby('strategy').agg({
            'prediction_correct': 'mean',
            'total_latency_ms': 'mean',
            'energy_consumed_mah': 'sum'
        })
        metrics['accuracy_percent'] = metrics['prediction_correct'] * 100
        
        # Plot 1: Accuracy vs Energy
        ax = axes[0]
        for strategy in metrics.index:
            ax.scatter(metrics.loc[strategy, 'energy_consumed_mah'],
                      metrics.loc[strategy, 'accuracy_percent'],
                      s=200, label=strategy, alpha=0.7)
        ax.set_xlabel('Total Energy (mAh)', fontsize=12)
        ax.set_ylabel('Accuracy (%)', fontsize=12)
        ax.set_title('Accuracy vs Energy Trade-off', fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Plot 2: Accuracy vs Latency
        ax = axes[1]
        for strategy in metrics.index:
            ax.scatter(metrics.loc[strategy, 'total_latency_ms'],
                      metrics.loc[strategy, 'accuracy_percent'],
                      s=200, label=strategy, alpha=0.7)
        ax.set_xlabel('Avg Latency (ms)', fontsize=12)
        ax.set_ylabel('Accuracy (%)', fontsize=12)
        ax.set_title('Accuracy vs Latency Trade-off', fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        filename = os.path.join(self.plot_dir, 'tradeoff_analysis.png')
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"  ✓ Saved: {filename}")
        plt.close()
    
    def plot_placement_decisions(self):
        """Plot placement decision patterns"""
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Plot 1: Placement distribution by strategy
        ax = axes[0]
        placement = pd.crosstab(self.df['strategy'], self.df['placement_decision'], normalize='index') * 100
        placement.plot(kind='bar', stacked=True, ax=ax)
        ax.set_title('Placement Decision Distribution', fontsize=14, fontweight='bold')
        ax.set_ylabel('Percentage (%)', fontsize=12)
        ax.set_xlabel('Strategy', fontsize=12)
        ax.legend(title='Placement')
        ax.grid(True, alpha=0.3)
        
        # Plot 2: A²PF placement by network condition
        ax = axes[1]
        a2pf_df = self.df[self.df['strategy'] == 'a2pf']
        if not a2pf_df.empty:
            a2pf_placement = pd.crosstab(a2pf_df['network_condition'], 
                                         a2pf_df['placement_decision'], 
                                         normalize='index') * 100
            a2pf_placement.plot(kind='bar', stacked=True, ax=ax)
            ax.set_title('A²PF Placement by Network Condition', fontsize=14, fontweight='bold')
            ax.set_ylabel('Percentage (%)', fontsize=12)
            ax.set_xlabel('Network Condition', fontsize=12)
            ax.legend(title='Placement')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        filename = os.path.join(self.plot_dir, 'placement_decisions.png')
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"  ✓ Saved: {filename}")
        plt.close()
    
    def generate_all_plots(self):
        """Generate all analysis plots"""
        print("\n" + "="*70)
        print("GENERATING PLOTS")
        print("="*70)
        
        self.plot_accuracy_comparison()
        self.plot_latency_comparison()
        self.plot_energy_comparison()
        self.plot_tradeoff_analysis()
        self.plot_placement_decisions()
        
        print("\n✓ All plots generated successfully!")
        print(f"✓ Plots saved to: {self.plot_dir}")


# ============================================================================
# MAIN EXECUTION
# ============================================================================
if __name__ == "__main__":
    analyzer = ResultsAnalyzer()
    analyzer.print_summary_statistics()
    analyzer.generate_all_plots()
    
    print("\n" + "="*70)
    print("ANALYSIS COMPLETE!")
    print("="*70)
    print("\nYou can now:")
    print("  1. Check the plots in the 'results/plots' folder")
    print("  2. Review the CSV file for detailed data")
    print("  3. Use the results in your research paper")
    print("="*70)
