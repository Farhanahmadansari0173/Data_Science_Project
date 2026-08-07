"""
Script 5: Advanced A/B Testing via Bootstrapping & Power Analysis

A simple t-test just provides a p-value. To provide deep business value, 
I use Statistical Bootstrapping. By resampling the A/B test data 10,000 times, 
I can simulate thousands of alternative realities to calculate the 95% Confidence Interval 
of the actual revenue lift caused by the new UI.

I also run a Statistical Power Analysis to prove that our sample size is sufficient 
to detect this revenue lift, avoiding false negatives (Type II Errors).
"""
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.stats.power import TTestIndPower

DATA_DIR = "data/raw"
FIGURES_DIR = "reports/figures"

def run_bootstrapping_and_power():
    print("Running Statistical Bootstrapping for A/B Test...")
    os.makedirs(FIGURES_DIR, exist_ok=True)
    
    df = pd.read_csv(os.path.join(DATA_DIR, "ab_test_results.csv"))
    
    control_rev = df[df['Group'] == 'Control']['CheckoutRevenue'].values
    variant_rev = df[df['Group'] == 'Variant']['CheckoutRevenue'].values
    
    # 1. Statistical Power Analysis Curve
    print("Running Power Analysis...")
    effect_size = (np.mean(variant_rev) - np.mean(control_rev)) / np.std(control_rev)
    power_analysis = TTestIndPower()
    
    sample_sizes = np.array(range(500, 10000, 500))
    powers = [power_analysis.power(effect_size=effect_size, nobs1=n, alpha=0.05, ratio=1.0, alternative='two-sided') for n in sample_sizes]
    
    plt.figure(figsize=(8, 5))
    plt.plot(sample_sizes, powers, marker='o', color='darkorange', linewidth=2)
    plt.title('Statistical Power Curve vs Sample Size')
    plt.xlabel('Sample Size per Group')
    plt.ylabel('Statistical Power (1 - Type II Error)')
    plt.axhline(0.80, color='red', linestyle='--', label='80% Power Threshold')
    plt.axvline(len(control_rev), color='green', linestyle='-', label=f'Actual Sample Size ({len(control_rev)})')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(FIGURES_DIR, "statistical_power_curve.png"), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Bootstrapping (10,000 Simulations)
    n_iterations = 10000
    np.random.seed(42)
    
    boot_control = np.empty(n_iterations)
    boot_variant = np.empty(n_iterations)
    
    for i in range(n_iterations):
        boot_control[i] = np.mean(np.random.choice(control_rev, size=len(control_rev), replace=True))
        boot_variant[i] = np.mean(np.random.choice(variant_rev, size=len(variant_rev), replace=True))
        
    lift = boot_variant - boot_control
    
    ci_lower = np.percentile(lift, 2.5)
    ci_upper = np.percentile(lift, 97.5)
    
    plt.figure(figsize=(10, 6))
    sns.kdeplot(lift, fill=True, color='purple', alpha=0.5)
    
    plt.axvline(ci_lower, color='red', linestyle='--', label=f'95% CI Lower (${ci_lower:.2f})')
    plt.axvline(ci_upper, color='green', linestyle='--', label=f'95% CI Upper (${ci_upper:.2f})')
    plt.axvline(np.mean(lift), color='black', linestyle='-', label=f'Mean Lift (${np.mean(lift):.2f})')
    
    plt.title('Bootstrap Distribution of Revenue Lift (10,000 Simulations)')
    plt.xlabel('Increase in Average Revenue Per User (USD)')
    plt.ylabel('Density')
    plt.legend()
    
    plt.savefig(os.path.join(FIGURES_DIR, "ab_test_bootstrap_kde.png"), dpi=300)
    plt.close()
    
    print(f"Bootstrapping complete.")
    print(f"Mean Revenue Lift: ${np.mean(lift):.2f}")
    print(f"95% Confidence Interval: [${ci_lower:.2f}, ${ci_upper:.2f}]")

if __name__ == "__main__":
    run_bootstrapping_and_power()
