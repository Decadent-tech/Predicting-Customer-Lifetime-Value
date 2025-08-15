# Customer Lifetime Value Prediction using BG/NBD Model

This project predicts customer lifetime value and purchasing patterns using the **BG/NBD** (Beta-Geometric / Negative Binomial Distribution) model from the [`lifetimes`]library, applied to the **Olist Brazilian E-Commerce Public Dataset**.

It estimates:
- **Probability a customer is still active (`p_alive`)**
- **Expected number of future purchases over a given time period**
- **Calibration vs holdout performance**
- **Cumulative and incremental transaction trends**

---

##  Dataset

The model uses three CSV datasets from Olist:

1. **Customers** – `olist_customers_dataset.csv`  
    customer_id, customer_unique_id, customer_zip_code_prefix, customer_city, customer_state


2. **Orders** – `olist_orders_dataset.csv`  


    order_id, customer_id, order_status, order_purchase_timestamp,
    order_approved_at, order_delivered_carrier_date,
    order_delivered_customer_date, order_estimated_delivery_date


3. **Order Payments** – `olist_order_payments_dataset.csv`  


    order_id, payment_sequential, payment_type, payment_installments, payment_value

##  Installation

Clone the repository and install dependencies:

git clone https://github.com/Decadent-tech/Predicting-Customer-Lifetime-Value
cd Predicting-Customer-Lifetime-Value
pip install -r requirements.txt

        pandas
        matplotlib
        seaborn
        lifetimes
        numpy

##  Usage
Place your datasets in a folder named Dataset/:

        Dataset/
        ├─ olist_customers_dataset.csv
        ├─ olist_orders_dataset.csv
        ├─ olist_order_payments_dataset.csv


Run the script:

python main.py

##  Methodology
        1. Data Preparation

                Merge orders, customers, and payments.
                Filter for delivered orders.
                Create transaction logs (elog) with:
                CUSTOMER_ID | ORDER_DATE

        2. Summary Table Creation

                Use summary_data_from_transaction_data to calculate:
                frequency – repeat purchase count
                recency – time between first and last purchase
                T – customer's age in the dataset

        3. Model Fitting

                Try fitting BG/NBD with different penalizer coefficients for stability.
                Fall back to Modified BG/NBD if fitting fails.

        4. Predictions

                Predict purchases in the next 90 days.
                Calculate probability_alive for each customer.

        5. Visualization

                Frequency–Recency Matrix
                Probability Alive Matrix
                Calibration vs Holdout Purchases
                Cumulative Transactions
                Incremental Transactions

        6. Results

                Save predictions to customer_lifetime_predictions.csv with:
                CUSTOMER_ID, frequency, recency, T, predicted_purchases_90d, p_alive


## Notes

The script caps extreme values of recency and T at the 99th percentile to stabilize fitting.
Zero-frequency customers are included in the model, as BG/NBD can handle them.
The calibration period end date is set to 2018-06-30, with a holdout period until 2018-09-28.        

## License

This project is released under the MIT License.