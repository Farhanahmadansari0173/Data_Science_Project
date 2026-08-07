# 📊 Retail Analytics & A/B Testing Engine (Data Science)

Welcome to my Data Science project! My name is **Farhan Ahmad Ansari**, and I built this repository to showcase end-to-end Data Engineering, Statistical Hypothesis Testing, and Interactive Dashboarding.

While my other projects focus on predictive AI (Machine Learning and Deep Learning), this project proves my ability to extract actionable business insights from raw, messy data.

## 💡 The Architecture

This project simulates a real-world Retail/E-commerce environment:

1. **Data Engineering (ETL)**:
   - Built a robust Pandas pipeline to ingest and clean 20,000+ raw customer transaction logs.
   - Performed **RFM (Recency, Frequency, Monetary)** segmentation to categorize users into behavioral groups (e.g., "Champions", "At Risk / Churned").
   
2. **Statistical A/B Testing**:
   - Analyzed the results of a hypothetical UI redesign (Control vs. Variant) across 10,000 users.
   - Utilized `scipy.stats` to run a **Chi-Square Test** on Conversion Rates and an **Independent T-Test** on Average Revenue Per User (ARPU).
   - Automatically calculates p-values to determine if the marketing campaign's success was statistically significant or just random chance.

3. **Interactive Dashboard**:
   - Wrapped the entire pipeline into a beautiful, interactive web application using **Streamlit** and **Plotly**.
   - Executives can use this dashboard to visualize customer lifetime value and immediately see the results of live A/B tests.

## 📂 Project Structure

- `scripts/1_generate_data.py`: Simulates the raw E-commerce transactions and A/B test logs.
- `scripts/2_run_etl.py`: The ETL pipeline that cleans data and computes RFM customer segments.
- `app.py`: The main Streamlit web application.
- `requirements.txt`: Project dependencies.

## 🚀 How to Run Locally

1. Install the required libraries:
   ```bash
   pip install -r requirements.txt
   ```
2. Run the Data Pipeline (generates data and runs ETL):
   ```bash
   python scripts/1_generate_data.py
   python scripts/2_run_etl.py
   ```
3. Start the Streamlit Dashboard:
   ```bash
   streamlit run app.py
   ```

Navigate to `http://localhost:8501` in your browser to interact with the dashboard!

---
*Created by Farhan Ahmad Ansari.*
