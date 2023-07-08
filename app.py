    #```````````````````````````````````````````````````````````````````````````````````````````````````#`
# Add the click button task#
from flask import Flask, render_template, redirect, url_for,request
import sqlite3
import pickle
import numpy as np
import os

app = Flask(__name__)

# Specify the path to the model file
model_path = os.path.join(os.path.dirname(__file__), 'static', 'model.pkl')

# Load the trained model
with open(model_path, 'rb') as f:
    model = pickle.load(f)

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/dashboard')
def dashboard():
    # Add code to handle the dashboard route
    return "Dashboard Page"

@app.route('/prediction')
def prediction():
    # Redirect to the result route
    return redirect(url_for('show_data', nameOrig='customer_name_here'))

# Configure the database connection
db_path = 'C:/Users/23059/OneDrive/Desktop/Amiira/DB_test/Pastdata.db'

@app.route('/result', methods=['GET', 'POST'])
def show_data():
    if request.method == 'POST':
        # Handle the form submission
        step = int(request.form['step'])
        type = request.form['type']
        amount = float(request.form['amount'])
        nameOrig = request.form['nameOrig']
        oldbalanceOrg = float(request.form['oldbalanceOrg'])
        newbalanceOrig = float(request.form['newbalanceOrig'])
        newDest = request.form['newDest']
        oldbalanceDest = float(request.form['oldbalanceDest'])
        newbalanceDest = float(request.form['newbalanceDest'])
        isFlaggedFraud = int(request.form['isFlaggedFraud'])

        # Prepare the input data for prediction
        input_data = np.array([[step, type, amount, nameOrig, oldbalanceOrg, newbalanceOrig, newDest,
                                oldbalanceDest, newbalanceDest, isFlaggedFraud]])
        input_data = input_data.astype(np.float32)
        input_data = np.reshape(input_data, (input_data.shape[0], 1, input_data.shape[1]))

        # Make predictions using the model
        prediction = model.predict(input_data)[0]

        # Connect to the SQLite database
        connection = sqlite3.connect(db_path)
        cursor = connection.cursor()

        # Execute a query to retrieve data for the given customer name
        query = "SELECT * FROM train_data WHERE nameOrig = ?"
        cursor.execute(query, (nameOrig,))

        # Fetch the data from the query result
        data = cursor.fetchall()

        # Close the database connection
        cursor.close()
        connection.close()

        # Check if any row has isFraud = 1
        is_fraudulent = any(row[10] == 1 for row in data)

        # Pass the data, prediction, and is_fraudulent flag to the template
        return render_template('result.html', data=data, prediction=prediction, is_fraudulent=is_fraudulent)

    # Handle the GET request (initial page load and customer search)
    search_type = request.args.get('searchType')
    search_value = request.args.get('searchValue')

    if search_type and search_value:
        # Connect to the SQLite database
        connection = sqlite3.connect(db_path)
        cursor = connection.cursor()

        # Execute a query to retrieve data based on the search type and value
        if search_type == 'nameOrig':
            query = "SELECT * FROM train_data WHERE nameOrig = ?"
        elif search_type == 'nameDest':
            query = "SELECT * FROM train_data WHERE nameDest = ?"

        cursor.execute(query, (search_value,))

        # Fetch the data from the query result
        data = cursor.fetchall()

        # Close the database connection
        cursor.close()
        connection.close()

        # Check if any row has isFraud = 1
        is_fraudulent = any(row[10] == 1 for row in data)

        # Pass the data, is_fraudulent flag, and search_type/search_value to the template
        return render_template('result.html', data=data, is_fraudulent=is_fraudulent,
                               searchType=search_type, searchValue=search_value)

    return render_template('result.html')

if __name__ == '__main__':
    app.run(debug=True)


# import pandas as pd
# import sqlite3

# df=pd.read_csv('concatenated_data.csv')

# connection=sqlite3.connect('PastData.db')

# df.to_sql('train_data',connection,if_exists='replace')

# connection.close()