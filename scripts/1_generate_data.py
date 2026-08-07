"""
Script 1: Generate E-commerce Dataset

I generate a synthetic dataset of customer transactions and A/B test results.
In the real world, this data would come from a SQL database, but generating it
locally ensures this project can be run anywhere.
"""
import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

DATA_DIR = "data/raw"

def generate_transactions(num_customers=5000, num_transactions=20000):
    print("Generating synthetic customer transactions...")
    np.random.seed(42)
    
    # Generate customer IDs
    customer_ids = np.random.randint(1000, 1000 + num_customers, size=num_transactions)
    
    # Generate dates over the last 12 months
    end_date = datetime.now()
    start_date = end_date - timedelta(days=365)
    
    # Random timestamps between start and end
    random_days = np.random.randint(0, 365, size=num_transactions)
    dates = [start_date + timedelta(days=int(d)) for d in random_days]
    
    # Generate monetary values (Log-normal distribution to simulate real purchases)
    values = np.random.lognormal(mean=3.5, sigma=1.0, size=num_transactions)
    values = np.round(np.clip(values, 5, 500), 2)
    
    df = pd.DataFrame({
        'TransactionID': np.arange(1, num_transactions + 1),
        'CustomerID': customer_ids,
        'Date': dates,
        'Revenue': values
    })
    
    df_path = os.path.join(DATA_DIR, "transactions.csv")
    df.to_csv(df_path, index=False)
    print(f"Saved {num_transactions} transactions to {df_path}")

def generate_ab_test_data(num_users=10000):
    print("Generating A/B test results (Control vs Variant)...")
    np.random.seed(42)
    
    # Assign 50/50 to Control and Variant (New UI)
    groups = np.random.choice(['Control', 'Variant'], size=num_users, p=[0.5, 0.5])
    
    # Simulate conversion rates:
    # Control has ~5% conversion rate
    # Variant has ~6.5% conversion rate
    conversions = []
    revenues = []
    for g in groups:
        if g == 'Control':
            conv = np.random.choice([0, 1], p=[0.95, 0.05])
            rev = np.round(np.random.uniform(20, 100), 2) if conv else 0.0
        else:
            conv = np.random.choice([0, 1], p=[0.935, 0.065])
            rev = np.round(np.random.uniform(20, 120), 2) if conv else 0.0
        
        conversions.append(conv)
        revenues.append(rev)
        
    df = pd.DataFrame({
        'UserID': np.arange(1, num_users + 1),
        'Group': groups,
        'Converted': conversions,
        'CheckoutRevenue': revenues
    })
    
    df_path = os.path.join(DATA_DIR, "ab_test_results.csv")
    df.to_csv(df_path, index=False)
    print(f"Saved {num_users} A/B test logs to {df_path}")

if __name__ == "__main__":
    os.makedirs(DATA_DIR, exist_ok=True)
    generate_transactions()
    generate_ab_test_data()
