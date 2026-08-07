"""
Script 2: ETL and Feature Engineering (RFM Calculation)

I perform Data Engineering to clean the raw transactions and calculate 
raw Recency, Frequency, and Monetary (RFM) values for every customer.
This prepares the data for Unsupervised Machine Learning.
"""
import os
import pandas as pd

DATA_DIR = "data/raw"
PROCESSED_DIR = "data/processed"

def calculate_raw_rfm():
    print("Running ETL Pipeline for RFM Calculation...")
    df = pd.read_csv(os.path.join(DATA_DIR, "transactions.csv"))
    
    # Clean data (ensure datetime)
    df['Date'] = pd.to_datetime(df['Date'])
    
    # Assume "today" is the day after the last transaction
    snapshot_date = df['Date'].max() + pd.Timedelta(days=1)
    
    # Calculate raw RFM metrics per customer
    rfm = df.groupby('CustomerID').agg({
        'Date': lambda x: (snapshot_date - x.max()).days,
        'TransactionID': 'count',
        'Revenue': 'sum'
    }).rename(columns={
        'Date': 'Recency',
        'TransactionID': 'Frequency',
        'Revenue': 'Monetary'
    }).reset_index()
    
    # Save the processed, aggregated data without hardcoded segments
    out_path = os.path.join(PROCESSED_DIR, "rfm_data.csv")
    rfm.to_csv(out_path, index=False)
    print(f"ETL completed! Saved raw RFM profiles for {len(rfm)} customers to {out_path}")

if __name__ == "__main__":
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    calculate_raw_rfm()
