import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from lifetimes import BetaGeoFitter
from lifetimes.plotting import plot_frequency_recency_matrix, plot_probability_alive_matrix


# --------------------
# Load datasets
# --------------------
customers = pd.read_csv("Dataset/olist_customers_dataset.csv")
orders = pd.read_csv("Dataset/olist_orders_dataset.csv", parse_dates=[
    "order_purchase_timestamp",
    "order_approved_at",
    "order_delivered_carrier_date",
    "order_delivered_customer_date",
    "order_estimated_delivery_date"
])
payments = pd.read_csv("Dataset/olist_order_payments_dataset.csv")
# Merge orders with customers
orders_customers = pd.merge(orders, customers, on="customer_id", how="left")

# Merge with payments (only keep relevant payment rows — optional)
full_df = pd.merge(orders_customers, payments, on="order_id", how="left")

# Keep only delivered orders
full_df = full_df[full_df["order_status"] == "delivered"]


# ==== Build transaction log in the shape lifetimes expects ====
# Use the same elog you already prepared earlier:
# elog has: ['CUSTOMER_ID', 'ORDER_DATE'] with ORDER_DATE as datetime64

from lifetimes.utils import summary_data_from_transaction_data
from lifetimes import BetaGeoFitter, ModifiedBetaGeoFitter
from lifetimes.plotting import (
    plot_frequency_recency_matrix,
    plot_probability_alive_matrix,
    plot_calibration_purchases_vs_holdout_purchases,
    plot_cumulative_transactions,
    plot_incremental_transactions,
)
import numpy as np
elog = full_df[['customer_id', 'order_purchase_timestamp']].rename(
    columns={'customer_id': 'CUSTOMER_ID', 'order_purchase_timestamp': 'ORDER_DATE'})
# 1) Create the summary directly from transactions (this avoids many pitfalls)
#    Keep units in DAYS by setting freq='D'.
observation_end = elog['ORDER_DATE'].max()
summary = summary_data_from_transaction_data(
    transactions=elog,
    customer_id_col='CUSTOMER_ID',
    datetime_col='ORDER_DATE',
    observation_period_end=observation_end,
    freq='D'  # days
)

# Optional sanity checks
# Drop impossible rows and NaNs (rare, but good hygiene)
summary = summary.replace([np.inf, -np.inf], np.nan).dropna()
summary = summary[(summary['T'] > 0) & (summary['recency'] >= 0) & (summary['recency'] <= summary['T'])]

# Winsorize extremes to stabilize optimization (helps with overflow)
for col in ['recency', 'T']:
    cap = summary[col].quantile(0.99)
    summary[col] = np.minimum(summary[col], cap)

# You can keep zero-frequency customers for BG/NBD; the fitter handles them.
print('Rows for fitting:', len(summary), 
      ' | freq>0:', (summary['frequency']>0).sum(), 
      ' | zeros:', (summary['frequency']==0).sum())

# 2) Fit BG/NBD with a stabilizing penalizer; fall back to MBG/NBD if needed
def fit_bgnbd_stable(df):
    for penal in [0.1, 0.01, 0.001, 0.0001]:
        try:
            bgf = BetaGeoFitter(penalizer_coef=penal)
            
            bgf.fit(df['frequency'], df['recency'], df['T'], verbose=True)
            print(f'BG/NBD converged with penalizer={penal}')
            return bgf, 'BG/NBD'
        except Exception as e:
            print(f'BG/NBD failed (penalizer={penal}): {e}')
    # Fallback: Modified BG/NBD is often more stable
    mbg = ModifiedBetaGeoFitter(penalizer_coef=0.1)
    mbg.fit(df['frequency'], df['recency'], df['T'], verbose=True)
    print('Fell back to Modified BG/NBD (MBG/NBD).')
    return mbg, 'MBG/NBD'

model, model_name = fit_bgnbd_stable(summary)
print(model)

# 3) Predictions (next 90 days)
t_future = 90.0  # days
summary['predicted_purchases_90d'] = model.conditional_expected_number_of_purchases_up_to_time(
    t_future, summary['frequency'], summary['recency'], summary['T']
)
summary['p_alive'] = model.conditional_probability_alive(
    summary['frequency'], summary['recency'], summary['T']
)

# 4) Plots (will work for either BG/NBD or MBG/NBD)
plt.figure(figsize=(8, 6))
plot_frequency_recency_matrix(model)
plt.title(f'{model_name} – Expected Transactions by Frequency/Recency')
plt.tight_layout()
plt.show()
sns.despine()

plt.figure(figsize=(8, 6))
plot_probability_alive_matrix(model)
plt.title(f'{model_name} – Probability Alive Matrix')
plt.tight_layout()
plt.show()
sns.despine()

# 5) Calibration vs Holdout, Cumulative & Incremental (using your earlier split)
# Recreate your calibration/holdout split using the same elog and dates
from lifetimes.utils import calibration_and_holdout_data

calibration_period_ends = '2018-06-30'
summary_cal_holdout = calibration_and_holdout_data(
    elog,
    customer_id_col='CUSTOMER_ID',
    datetime_col='ORDER_DATE',
    freq='D',
    calibration_period_end=calibration_period_ends,
    observation_period_end='2018-09-28',
)

plt.figure(figsize=(7, 5))
plot_calibration_purchases_vs_holdout_purchases(model, summary_cal_holdout)
plt.title(f'{model_name} – Calibration vs Holdout Purchases')
plt.tight_layout()
plt.show()
sns.despine()

# Cumulative & Incremental transactions plots prefer the raw transactions
from datetime import datetime
elog_renamed = elog.rename(columns={'ORDER_DATE': 'date'})  # expected col name
t_total = (elog_renamed['date'].max() - elog_renamed['date'].min()).days
t_cal = (datetime.strptime('2018-06-30', '%Y-%m-%d') - elog_renamed['date'].min()).days

plt.figure(figsize=(8, 5))
plot_cumulative_transactions(model, elog_renamed, 'date', 'CUSTOMER_ID', t_total, t_cal, freq='D')
plt.title(f'{model_name} – Cumulative Transactions')
plt.tight_layout()
plt.show()
sns.despine()

plt.figure(figsize=(8, 5))
plot_incremental_transactions(model, elog_renamed, 'date', 'CUSTOMER_ID', t_total, t_cal, freq='D')
plt.title(f'{model_name} – Incremental Transactions')
plt.tight_layout()
plt.show()
sns.despine()

# 6) Save results
summary_out = summary.copy()
summary_out['p_alive'] = summary_out['p_alive'].round(4)
summary_out['predicted_purchases_90d'] = summary_out['predicted_purchases_90d'].round(3)
summary_out.to_csv('customer_lifetime_predictions.csv', index=False)
print('Saved: customer_lifetime_predictions.csv')


# Plot calibration vs holdout purchases
plt.figure(figsize=(7, 5))  
plot_calibration_purchases_vs_holdout_purchases(model, summary_cal_holdout)
plt.title(f'{model_name} – Calibration vs Holdout Purchases')
plt.tight_layout()
plt.show()
sns.despine()

