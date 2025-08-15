# Importing necessary libraries

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import lifetimes

#Let's make this notebook reproducible 
np.random.seed(42)

import random
random.seed(42)

import warnings
warnings.filterwarnings('ignore')

# Plotting parameters
# Make the default figures a bit bigger
plt.rcParams['figure.figsize'] = (7,4.5) 
plt.rcParams["figure.dpi"] = 140 

sns.set(style="ticks")
sns.set_context("poster", font_scale = .5, rc={"grid.linewidth": 5})

df1 = pd.read_csv('Dataset/olist_orders_dataset.csv')
df2 = pd.read_csv('Dataset/olist_customers_dataset.csv')
df3 = pd.read_csv('Dataset/olist_order_payments_dataset.csv')

cols = ['customer_id', 'order_id', 'order_purchase_timestamp']
orders = df1[cols]
orders = orders.set_index('customer_id')
orders.drop_duplicates(inplace=True)

# too few 
cols = ['order_id', 'payment_value']
payment = df3[cols]
payment = payment.set_index('order_id')
payment.drop_duplicates(inplace=True)

cols = ['customer_id', 'customer_unique_id']
customers = df2[cols]
customers = customers.set_index('customer_id')

elog = pd.concat([orders,customers], axis=1, join='inner')
elog.reset_index(inplace=True)

cols = ['customer_unique_id', 'order_purchase_timestamp']
elog = elog[cols]

elog['order_purchase_timestamp'] = pd.to_datetime(elog['order_purchase_timestamp'])
elog['order_date'] = elog.order_purchase_timestamp.dt.date
elog['order_date'] = pd.to_datetime(elog['order_date'])

cols = ['customer_unique_id', 'order_date']
elog = elog[cols]

elog.columns = ['CUSTOMER_ID', 'ORDER_DATE']


elog.info()
print(elog.sample(5))

# Date range of orders¶
print(elog.ORDER_DATE.describe())


# Creating RFM Matrix based on transaction log¶
# Spliting calibration and holdout period
# We will use the last 3 months as holdout period
# and the rest as calibration period
calibration_period_ends = '2018-06-30'

from lifetimes.utils import calibration_and_holdout_data

summary_cal_holdout = calibration_and_holdout_data(elog, 
                                                   customer_id_col = 'CUSTOMER_ID', 
                                                   datetime_col = 'ORDER_DATE', 
                                                   freq = 'D', #days
                                        calibration_period_end=calibration_period_ends,
                                        observation_period_end='2018-09-28' )

# Feature set¶
print(summary_cal_holdout.head())

from lifetimes import ModifiedBetaGeoFitter

mbgnbd = ModifiedBetaGeoFitter(penalizer_coef=0.01)
mbgnbd.fit(summary_cal_holdout['frequency_cal'], 
        summary_cal_holdout['recency_cal'], 
        summary_cal_holdout['T_cal'],
       verbose=True)

print(mbgnbd)
# Predictions for each customer in the holdout period
t = 90 # days to predict in the future 
summary_cal_holdout['predicted_purchases'] = mbgnbd.conditional_expected_number_of_purchases_up_to_time(t, 
                                                                                      summary_cal_holdout['frequency_cal'], 
                                                                                      summary_cal_holdout['recency_cal'], 
                                                                                      summary_cal_holdout['T_cal'])

summary_cal_holdout['p_alive'] = mbgnbd.conditional_probability_alive(summary_cal_holdout['frequency_cal'], 
                                                                         summary_cal_holdout['recency_cal'], 
                                                                         summary_cal_holdout['T_cal'])
summary_cal_holdout['p_alive'] = np.round(summary_cal_holdout['p_alive'] / summary_cal_holdout['p_alive'].max(), 2)

print(summary_cal_holdout.sample(2).T)

# Accessing the model fit 
from lifetimes.plotting import plot_period_transactions
ax = plot_period_transactions(mbgnbd, max_frequency=7)
ax.set_yscale('log')
plt.show()
sns.despine()

from lifetimes.plotting import plot_calibration_purchases_vs_holdout_purchases

plot_calibration_purchases_vs_holdout_purchases(mbgnbd, summary_cal_holdout)
plt.show()
sns.despine()

# Customer Probability History
from lifetimes.plotting import plot_history_alive
from datetime import date
from pylab import figure, text, scatter, show

individual = summary_cal_holdout.iloc[4942]

id = individual.name
t = 365*50

today = date.today()
two_year_ago = today.replace(year=today.year - 2)
one_year_from_now = today.replace(year=today.year + 1)

id = 1  # or whichever customer

sp_trans = (
    elog.loc[elog['CUSTOMER_ID'] == id, ['ORDER_DATE']]
    .copy()
)

sp_trans['ORDER_DATE'] = pd.to_datetime(sp_trans['ORDER_DATE'])
sp_trans['orders'] = 1  # numeric

# Make ORDER_DATE the index because plot_history_alive expects that
sp_trans = sp_trans.set_index('ORDER_DATE')

from lifetimes.utils import calculate_alive_path 
from lifetimes.plotting import plot_history_alive

ax = plot_history_alive(
    mbgnbd,
    t,
    sp_trans,
    datetime_col='ORDER_DATE',
    transaction_col='orders',
    start_date=two_year_ago,
    freq='D'
)
ax.set_title(f'Customer {id} - Probability of being alive')
ax.set_xlabel('Date')
ax.set_ylabel('Probability of being alive')
ax.set_xlim(two_year_ago, one_year_from_now)
ax.set_ylim(0, 1)
ax.axvline(x=today, color='red', linestyle='--', label='Today')
ax.legend()
plt.show()