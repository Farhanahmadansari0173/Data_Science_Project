#!/bin/bash
set -e

echo "=== Advanced Data Science Pipeline ==="
echo "1. Creating Python Virtual Environment..."
python3 -m venv .venv
source .venv/bin/activate

echo "2. Installing requirements..."
pip install -r requirements.txt

echo "3. Generating Synthetic E-commerce Data..."
python scripts/1_generate_data.py

echo "4. Running ETL Pipeline..."
python scripts/2_run_etl.py

echo "5. Performing Unsupervised ML (K-Means & PCA)..."
python scripts/3_customer_clustering.py

echo "6. Performing Cohort Analysis..."
python scripts/4_cohort_analysis.py

echo "7. Running A/B Test Bootstrapping..."
python scripts/5_ab_test_bootstrapping.py

echo "=== Pipeline Completed Successfully! ==="
