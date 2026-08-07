"""
Script 2: ETL and Feature Engineering (RFM Analysis)

I perform Data Engineering to clean the raw transactions and calculate 
Recency, Frequency, and Monetary (RFM) values for every customer.
"""
import os
import pandas as pd
from datetime import datetime

DATA_DIR = "data/raw"
PROCESSED_DIR = "data/processed"

def calculate_rfm():
    print("Running ETL Pipeline for RFM Analysis...")
    df = pd.read_csv(os.path.join(DATA_DIR, "transactions.csv"))
    
    # Clean data (ensure datetime)
    df['Date'] = pd.to_datetime(df['Date'])
    
    # Assume "today" is the day after the last transaction
    snapshot_date = df['Date'].max() + pd.Timedelta(days=1)
    
    # Calculate RFM metrics per customer
    rfm = df.groupby('CustomerID').agg({
        'Date': lambda x: (snapshot_date - x.max()).days,
        'TransactionID': 'count',
        'Revenue': 'sum'
    }).rename(columns={
        'Date': 'Recency',
        'TransactionID': 'Frequency',
        'Revenue': 'Monetary'
    })
    
    # Segment customers based on quantiles
    quantiles = rfm.quantile(q=[0.33, 0.66])
    
    def r_score(x):
        if x <= quantiles['Recency'][0.33]: return 3 # Bought recently (best)
        elif x <= quantiles['Recency'][0.66]: return 2
        else: return 1
        
    def fm_score(x, metric):
        if x <= quantiles[metric][0.33]: return 1
        elif x <= quantiles[metric][0.66]: return 2
        else: return 3 # Bought a lot / spent a lot (best)
        
    rfm['R'] = rfm['Recency'].apply(r_score)
    rfm['F'] = rfm['Frequency'].apply(lambda x: fm_score(x, 'Frequency'))
    rfm['M'] = rfm['Monetary'].apply(lambda x: fm_score(x, 'Monetary'))
    
    rfm['RFM_Score'] = rfm['R'].map(str) + rfm['F'].map(str) + rfm['M'].map(str)
    
    # Label segments
    def segment(score):
        if score == '333': return 'Champions'
        elif score.startswith('1'): return 'At Risk / Churned'
        elif score.startswith('3'): return 'Recent Customers'
        else: return 'Average / Needs Attention'
        
    rfm['Segment'] = rfm['RFM_Score'].apply(segment)
    
    # Save the processed, aggregated data
    out_path = os.path.join(PROCESSED_DIR, "rfm_data.csv")
    rfm.to_csv(out_path)
    print(f"ETL completed! Saved {len(rfm)} customer RFM profiles to {out_path}")

if __name__ == "__main__":
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    calculate_rfm()
