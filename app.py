"""
Script 3: Streamlit Interactive Dashboard

I build an interactive web dashboard to present my Data Science insights.
This dashboard includes:
1. Executive Summary (KPIs)
2. RFM Customer Segmentation
3. A/B Testing Statistical Results
"""
import os
import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from scipy import stats

st.set_page_config(page_title="Retail Analytics Dashboard", layout="wide", page_icon="📊")

DATA_DIR = "data/raw"
PROCESSED_DIR = "data/processed"

@st.cache_data
def load_data():
    rfm_path = os.path.join(PROCESSED_DIR, "rfm_data.csv")
    ab_path = os.path.join(DATA_DIR, "ab_test_results.csv")
    
    if not os.path.exists(rfm_path) or not os.path.exists(ab_path):
        st.error("Data not found! Please run the ETL pipeline first.")
        st.stop()
        
    return pd.read_csv(rfm_path), pd.read_csv(ab_path)

def main():
    st.title("📊 Retail Analytics & A/B Testing Engine")
    st.markdown("*A Data Science project demonstrating end-to-end data engineering, statistical hypothesis testing, and interactive visualization.*")
    
    rfm_df, ab_df = load_data()
    
    tab1, tab2 = st.tabs(["🛍️ Customer Segmentation (RFM)", "🧪 A/B Testing Engine"])
    
    with tab1:
        st.header("RFM Customer Segmentation")
        st.markdown("We calculated **Recency**, **Frequency**, and **Monetary** value for every customer to identify high-value segments.")
        
        col1, col2, col3 = st.columns(3)
        col1.metric("Total Customers", f"{len(rfm_df):,}")
        col2.metric("Total Revenue", f"${rfm_df['Monetary'].sum():,.2f}")
        col3.metric("Avg Order Value", f"${rfm_df['Monetary'].sum() / rfm_df['Frequency'].sum():,.2f}")
        
        segment_counts = rfm_df['Segment'].value_counts().reset_index()
        segment_counts.columns = ['Segment', 'Count']
        
        fig1 = px.pie(segment_counts, values='Count', names='Segment', 
                     title="Customer Segments Breakdown", hole=0.4,
                     color_discrete_sequence=px.colors.qualitative.Pastel)
        st.plotly_chart(fig1, use_container_width=True)
        
        fig2 = px.scatter(rfm_df, x='Recency', y='Monetary', color='Segment',
                         size='Frequency', hover_name='CustomerID',
                         title="Recency vs Monetary Value (Bubble size = Frequency)",
                         opacity=0.6)
        st.plotly_chart(fig2, use_container_width=True)
        
    with tab2:
        st.header("A/B Testing Statistical Results")
        st.markdown("We launched a new UI (Variant) and compared it to the old UI (Control). We use **SciPy** to determine if the results are statistically significant.")
        
        control = ab_df[ab_df['Group'] == 'Control']
        variant = ab_df[ab_df['Group'] == 'Variant']
        
        # Metric 1: Conversion Rate
        c_conv = control['Converted'].mean()
        v_conv = variant['Converted'].mean()
        
        st.subheader("1. Conversion Rate Analysis (Chi-Square Test)")
        col_c, col_v = st.columns(2)
        col_c.metric("Control Conversion Rate", f"{c_conv*100:.2f}%")
        col_v.metric("Variant Conversion Rate", f"{v_conv*100:.2f}%", f"{(v_conv - c_conv)*100:.2f}%")
        
        # Run Chi-Square
        contingency = [
            [control['Converted'].sum(), len(control) - control['Converted'].sum()],
            [variant['Converted'].sum(), len(variant) - variant['Converted'].sum()]
        ]
        chi2, p_val, _, _ = stats.chi2_contingency(contingency)
        
        if p_val < 0.05:
            st.success(f"**Statistically Significant!** (p-value: {p_val:.4f}). We are 95%+ confident that the new UI increases conversion rate.")
        else:
            st.warning(f"**Not Significant.** (p-value: {p_val:.4f}). The difference could just be random chance.")
            
        # Metric 2: Revenue per User
        st.divider()
        st.subheader("2. Revenue Per User (Independent T-Test)")
        c_rev = control['CheckoutRevenue'].mean()
        v_rev = variant['CheckoutRevenue'].mean()
        
        col_c2, col_v2 = st.columns(2)
        col_c2.metric("Control ARPU", f"${c_rev:.2f}")
        col_v2.metric("Variant ARPU", f"${v_rev:.2f}", f"${(v_rev - c_rev):.2f}")
        
        t_stat, t_pval = stats.ttest_ind(control['CheckoutRevenue'], variant['CheckoutRevenue'], equal_var=False)
        
        if t_pval < 0.05:
            st.success(f"**Statistically Significant!** (p-value: {t_pval:.4f}). The new UI brings in more revenue per user.")
        else:
            st.warning(f"**Not Significant.** (p-value: {t_pval:.4f}). Revenue difference is not conclusive.")

if __name__ == "__main__":
    main()
