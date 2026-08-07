"""
Script 5: Advanced A/B Testing via Bootstrapping

A simple t-test just provides a p-value. To provide deep business value, 
I use Statistical Bootstrapping. By resampling the A/B test data 10,000 times, 
I can simulate thousands of alternative realities to calculate the 95% Confidence Interval 
of the actual revenue lift caused by the new UI.
"""
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

DATA_DIR = "data/raw"
FIGURES_DIR = "reports/figures"

def run_bootstrapping():
    print("Running Statistical Bootstrapping for A/B Test...")
    os.makedirs(FIGURES_DIR, exist_ok=True)
    
    df = pd.read_csv(os.path.join(DATA_DIR, "ab_test_results.csv"))
    
    control_rev = df[df['Group'] == 'Control']['CheckoutRevenue'].values
    variant_rev = df[df['Group'] == 'Variant']['CheckoutRevenue'].values
    
    n_iterations = 10000
    np.random.seed(42)
    
    # Initialize arrays to store bootstrapped means
    boot_control = np.empty(n_iterations)
    boot_variant = np.empty(n_iterations)
    
    # Perform Bootstrapping
    for i in range(n_iterations):
        boot_control[i] = np.mean(np.random.choice(control_rev, size=len(control_rev), replace=True))
        boot_variant[i] = np.mean(np.random.choice(variant_rev, size=len(variant_rev), replace=True))
        
    # Calculate the difference (Lift)
    lift = boot_variant - boot_control
    
    # Calculate 95% Confidence Interval
    ci_lower = np.percentile(lift, 2.5)
    ci_upper = np.percentile(lift, 97.5)
    
    # Plot the KDE Distribution
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
    run_bootstrapping()
