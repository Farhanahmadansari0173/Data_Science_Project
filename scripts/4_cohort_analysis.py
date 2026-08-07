"""
Script 4: Advanced Cohort Analysis

I perform a cohort analysis to track customer retention over time.
This generates a classic Data Science retention heatmap, showing the percentage 
of customers who return in subsequent months after their first purchase.
"""
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

DATA_DIR = "data/raw"
FIGURES_DIR = "reports/figures"

def run_cohort_analysis():
    print("Running Cohort Analysis...")
    os.makedirs(FIGURES_DIR, exist_ok=True)
    
    df = pd.read_csv(os.path.join(DATA_DIR, "transactions.csv"))
    df['Date'] = pd.to_datetime(df['Date'])
    
    # Create InvoiceMonth and CohortMonth
    df['InvoiceMonth'] = df['Date'].dt.to_period('M')
    df['CohortMonth'] = df.groupby('CustomerID')['InvoiceMonth'].transform('min')
    
    # Calculate Cohort Index (number of months since first purchase)
    def get_date_int(df, column):
        return df[column].dt.year, df[column].dt.month
        
    invoice_year, invoice_month = get_date_int(df, 'InvoiceMonth')
    cohort_year, cohort_month = get_date_int(df, 'CohortMonth')
    
    years_diff = invoice_year - cohort_year
    months_diff = invoice_month - cohort_month
    df['CohortIndex'] = years_diff * 12 + months_diff + 1
    
    # Count unique customers per cohort per month
    cohort_data = df.groupby(['CohortMonth', 'CohortIndex'])['CustomerID'].nunique().reset_index()
    cohort_counts = cohort_data.pivot(index='CohortMonth', columns='CohortIndex', values='CustomerID')
    
    # Calculate Retention Rate percentage
    cohort_sizes = cohort_counts.iloc[:, 0]
    retention = cohort_counts.divide(cohort_sizes, axis=0) * 100
    
    # Plot the Retention Heatmap
    plt.figure(figsize=(12, 8))
    plt.title('Customer Retention Rates by Monthly Cohort (%)')
    sns.heatmap(retention, annot=True, fmt='.1f', cmap='YlGnBu', vmin=0.0, vmax=100.0)
    plt.ylabel('Cohort Month')
    plt.xlabel('Months Since First Purchase')
    
    # Fix the y-axis labels to look cleaner
    plt.yticks(rotation=0)
    
    plt.savefig(os.path.join(FIGURES_DIR, "cohort_retention_heatmap.png"), dpi=300, bbox_inches='tight')
    plt.close()
    
    print("Cohort Analysis complete. Saved heatmap.")

if __name__ == "__main__":
    run_cohort_analysis()
