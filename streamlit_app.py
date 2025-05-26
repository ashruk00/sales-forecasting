import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Load forecast
forecast = pd.read_csv('reports/forecast.csv')
forecast['ds'] = pd.to_datetime(forecast['ds'])

# Load actuals
raw = pd.read_csv('data/train.csv')
raw['date'] = pd.to_datetime(raw['date'])
actuals = raw.groupby('date')['sales'].sum().reset_index()
actuals.columns = ['ds', 'actual_sales']

# Merge forecast + actuals
df_merged = pd.merge(forecast, actuals, on='ds', how='left')

# Streamlit page config
st.set_page_config(page_title="Sales Forecast Dashboard", layout="wide")

# Header
st.title("📈 Smart Sales Forecast Dashboard")
st.markdown("Prophet-based forecast with actuals comparison, error metrics, and export options.")

# Sidebar filters
min_date = df_merged['ds'].min()
max_date = df_merged['ds'].max()
date_range = st.sidebar.date_input("Select date range", [min_date, max_date], min_value=min_date, max_value=max_date)
show_ci = st.sidebar.checkbox("Show Confidence Intervals", value=True)

# Filter data
filtered_df = df_merged[(df_merged['ds'] >= pd.to_datetime(date_range[0])) & (df_merged['ds'] <= pd.to_datetime(date_range[1]))]

# KPIs
col1, col2, col3 = st.columns(3)
col1.metric("📊 Total Forecasted Sales", f"{filtered_df['yhat'].sum():,.2f}")
col2.metric("📈 Avg Daily Sales", f"{filtered_df['yhat'].mean():,.2f}")
col3.metric("🔺 Peak Day", f"{filtered_df.loc[filtered_df['yhat'].idxmax()]['ds'].date()}")

# Accuracy
valid = filtered_df.dropna(subset=['actual_sales'])
if not valid.empty:
    mae = mean_absolute_error(valid['actual_sales'], valid['yhat'])
    rmse = np.sqrt(mean_squared_error(valid['actual_sales'], valid['yhat']))
    st.info(f"📉 MAE: {mae:.2f} | RMSE: {rmse:.2f}")
else:
    st.warning("No actuals available in selected range.")

# Plot: Forecast vs Actual
fig = go.Figure()

fig.add_trace(go.Scatter(
    x=filtered_df['ds'], y=filtered_df['yhat'],
    name='Forecast', mode='lines', line=dict(color='blue')
))

fig.add_trace(go.Scatter(
    x=filtered_df['ds'], y=filtered_df['actual_sales'],
    name='Actual Sales', mode='lines+markers', line=dict(color='green')
))

if show_ci:
    fig.add_trace(go.Scatter(
        x=filtered_df['ds'], y=filtered_df['yhat_upper'],
        name='Upper Bound', mode='lines', line=dict(width=0.5, dash='dot')
    ))
    fig.add_trace(go.Scatter(
        x=filtered_df['ds'], y=filtered_df['yhat_lower'],
        name='Lower Bound', mode='lines', line=dict(width=0.5, dash='dot'),
        fill='tonexty', fillcolor='rgba(0, 0, 255, 0.1)'
    ))

fig.update_layout(title="📊 Actual vs Forecasted Sales", xaxis_title="Date", yaxis_title="Sales", template="plotly_white")
st.plotly_chart(fig, use_container_width=True)

# Table
with st.expander("📋 View Forecast + Actual Data"):
    st.dataframe(filtered_df.reset_index(drop=True))

# Export data
st.markdown("### 📤 Export Forecast Data")

csv = filtered_df.to_csv(index=False).encode('utf-8')

st.download_button(
    label="Download CSV",
    data=csv,
    file_name='forecast_vs_actual.csv',
    mime='text/csv'
)
