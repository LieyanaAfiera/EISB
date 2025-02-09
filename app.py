from flask import Flask, render_template, request, redirect, url_for, flash, session
import pandas as pd
from datetime import datetime
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import numpy as np
import plotly.graph_objects as go
import os
import threading
import plotly.express as px

# Initialize Flask app
app = Flask(__name__)
app.secret_key = "supersecretkey"

# File path for CSV
CSV_FILE = 'order.csv'
DATA_LOCK = threading.Lock()  # Lock for thread-safe access to CSV

# Helper function to format the date as mm-yy
def format_date_mm_yy(date_str):
    try:
        return datetime.strptime(date_str, '%b-%y').strftime('%m-%y')
    except ValueError:
        return date_str  # Return original if formatting fails

# Helper function to load dataset
def load_dataset():
    if not os.path.exists(CSV_FILE):
        return pd.DataFrame(columns=['Month', 'Product Code', 'Customer', 'Qty (pcs)'])
    try:
        df = pd.read_csv(CSV_FILE)
        df['Qty (pcs)'] = df['Qty (pcs)'].astype(str).str.replace(',', '').astype(int)
        df['Month'] = df['Month'].apply(format_date_mm_yy)  # Ensure mm-yy format
        return df
    except Exception as e:
        flash(f"Error loading dataset: {e}", "danger")
        return pd.DataFrame(columns=['Month', 'Product Code', 'Customer', 'Qty (pcs)'])

# Helper function to save dataset
def save_dataset(df):
    with DATA_LOCK:
        try:
            df['Month'] = df['Month'].apply(format_date_mm_yy)  # Format before saving
            df.to_csv(CSV_FILE, index=False)
        except Exception as e:
            flash(f"Error saving dataset: {e}", "danger")

# Initialize dataset
df = load_dataset()

# Routes
@app.route('/')
def homepage():
    """Homepage"""
    return render_template('index.html')

@app.route('/addOrder', methods=['GET', 'POST'])
def add_order():
    """Add a new order"""
    global df

    product_codes = df['Product Code'].unique()
    customers = df['Customer'].unique()

    if request.method == 'POST':
        try:
            month = request.form['month']
            try:
                formatted_month = datetime.strptime(month, '%m-%y').strftime('%m-%y')
            except ValueError:
                flash("Invalid date format. Use mm-yy.", "danger")
                return redirect(url_for('add_order'))

            product_code = request.form['product_code']
            customer = request.form['customer']
            qty = int(request.form['qty'].replace(',', ''))

            new_order = pd.DataFrame({
                'Month': [formatted_month],
                'Product Code': [product_code],
                'Customer': [customer],
                'Qty (pcs)': [qty]
            })

            df = pd.concat([df, new_order], ignore_index=True)
            save_dataset(df)

            flash("Order added successfully!", "success")
        except Exception as e:
            flash(f"Error adding order: {e}", "danger")
        return redirect(url_for('view_orders'))

    return render_template('addOrder.html', product_codes=product_codes, customers=customers)

@app.route('/view_orders', methods=['GET'])
def view_orders():
    """View all orders with filtering functionality"""
    global df
    df = load_dataset()  # Reload the dataset each time to ensure consistency

    if df.empty:
        flash("No orders found!", "warning")
        return render_template('viewOrders.html', orders=[], most_ordered_product="No data", most_regular_customer="No data")

    filter_date = request.args.get('filter_date')
    filter_product_code = request.args.get('filter_product_code')
    filter_customer = request.args.get('filter_customer')

    filtered_df = df.copy()

    if filter_date:
        filter_date = filter_date[5:7] + '-' + filter_date[2:4]
        filtered_df = filtered_df[filtered_df['Month'] == filter_date]

    if filter_product_code:
        filtered_df = filtered_df[filtered_df['Product Code'].str.contains(filter_product_code, case=False, na=False)]

    if filter_customer:
        filtered_df = filtered_df[filtered_df['Customer'].str.contains(filter_customer, case=False, na=False)]

    most_ordered_product = filtered_df['Product Code'].mode()[0] if not filtered_df.empty else "No data"
    most_regular_customer = filtered_df['Customer'].mode()[0] if not filtered_df.empty else "No data"

    orders = filtered_df.to_dict(orient='records')

    product_codes = df['Product Code'].dropna().unique()
    customers = df['Customer'].dropna().unique()

    return render_template('viewOrders.html', 
                           orders=orders, 
                           filter_date=filter_date, 
                           filter_product_code=filter_product_code, 
                           filter_customer=filter_customer,
                           most_ordered_product=most_ordered_product, 
                           most_regular_customer=most_regular_customer,
                           product_codes=product_codes,
                           customers=customers)

@app.route('/editOrder', methods=['GET', 'POST'])
def edit_order():
    """Edit an existing order"""
    global df

    if request.method == 'POST':
        try:
            index = int(request.form['index'])
            month = request.form['month']
            try:
                formatted_month = datetime.strptime(month, '%m-%y').strftime('%m-%y')
            except ValueError:
                flash("Invalid date format. Use mm-yy.", "danger")
                return redirect(url_for('view_orders'))

            df.loc[index, 'Month'] = formatted_month
            df.loc[index, 'Product Code'] = request.form['product_code']
            df.loc[index, 'Customer'] = request.form['customer']
            df.loc[index, 'Qty (pcs)'] = int(request.form['qty'].replace(',', ''))
            save_dataset(df)

            flash("Order updated successfully!", "success")
        except Exception as e:
            flash(f"Error updating order: {e}", "danger")
        return redirect(url_for('view_orders'))

    index = request.args.get('index', None)
    if index is not None:
        order = df.iloc[int(index)].to_dict()
        order['index'] = int(index)
        return render_template('editOrder.html', order=order)
    flash("No order selected for editing.", "danger")
    return redirect(url_for('view_orders'))

@app.route('/delete_order', methods=['POST'])
def delete_order():
    """Delete an order"""
    global df
    try:
        index = int(request.form['index'])
        df = df.drop(index).reset_index(drop=True)
        save_dataset(df)

        flash("Order deleted successfully!", "success")
    except Exception as e:
        flash(f"Error deleting order: {e}", "danger")
    return redirect(url_for('view_orders'))

@app.route('/generateReport')
def generate_report():
    """Generate monthly and yearly reports"""
    global df
    df = load_dataset()  # Ensure data is up-to-date

    df['Year'] = df['Month'].str[-2:]  # Extract the year from the 'Month' column

    # Calculate monthly totals
    monthly_totals = df.groupby('Month')['Qty (pcs)'].sum().to_dict()

    # Calculate yearly totals
    yearly_totals = df.groupby('Year')['Qty (pcs)'].sum().to_dict()

    return render_template('generateReport.html', monthly_totals=monthly_totals, yearly_totals=yearly_totals)


@app.route('/dashboard')
def dashboard():
    """Dashboard for visualizing data with advanced charts."""
    global df
    if df.empty:
        flash("No data available for the dashboard!", "danger")
        return redirect(url_for('homepage'))

    # Data preparation
    df['Month'] = pd.to_datetime(df['Month'], format='%m-%y', errors='coerce')
    df = df.dropna(subset=['Month'])
    monthly_data = df.groupby(df['Month'].dt.strftime('%Y-%m'))['Qty (pcs)'].sum().reset_index()
    monthly_data.columns = ['Month', 'Qty (pcs)']

    # Line chart for trends
    line_fig = go.Figure()
    line_fig.add_trace(go.Scatter(
        x=monthly_data['Month'],
        y=monthly_data['Qty (pcs)'],
        mode='lines+markers',
        line=dict(color='darkblue'),
        marker=dict(size=8)
    ))
    line_fig.update_layout(
        title="Monthly Order Trends",
        xaxis_title="Month",
        yaxis_title="Quantity",
        template="plotly_white"
    )

    # Stacked bar chart for product categories
    product_monthly = df.groupby(['Month', 'Product Code'])['Qty (pcs)'].sum().reset_index()
    stacked_fig = px.bar(
        product_monthly,
        x='Month',
        y='Qty (pcs)',
        color='Product Code',
        title="Product Distribution by Month",
        labels={'Qty (pcs)': 'Quantity', 'Month': 'Month'},
        template="plotly_white"
    )

    # Updated charts
    line_chart_html = line_fig.to_html(full_html=False)
    stacked_chart_html = stacked_fig.to_html(full_html=False)

    return render_template(
        'dashboard.html',
        line_chart_html=line_chart_html,
        stacked_chart_html=stacked_chart_html
    )


# Run app
if __name__ == '__main__':
    app.run(debug=True)
