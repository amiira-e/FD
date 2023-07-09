from flask import Flask, render_template, redirect, url_for, request
import sqlite3
import pickle
import numpy as np
import os
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
import io
import base64

app = Flask(__name__)

# Configure the database connection
db_path = 'C:/Users/23059/OneDrive/Desktop/Amiira/DB_test/Pastdata.db'

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

@app.route('/prediction', methods=['GET', 'POST'])
def prediction():
    if request.method == 'POST':
        # Handle the form submission and perform the prediction
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

        # Connect to the SQLite database and fetch the data
        connection = sqlite3.connect(db_path)
        cursor = connection.cursor()
        query = "SELECT * FROM train_data WHERE nameOrig = ?"
        cursor.execute(query, (nameOrig,))
        data = cursor.fetchall()
        cursor.close()
        connection.close()

        # Check if any row has isFraud = 1
        is_fraudulent = any(row[11] == 1 for row in data)

        # Perform the prediction and get the prediction probabilities
        prediction_probabilities = model.predict_proba(input_data)[0]

        # Generate the ROC curve image
        roc_curve_image = generate_roc_curve(prediction_probabilities)

        # Pass the data, prediction, is_fraudulent flag, and ROC curve image to the template
        return render_template('prediction.html', data=data, prediction=prediction, is_fraudulent=is_fraudulent, roc_curve_image=roc_curve_image)

    return redirect(url_for('show_data', nameOrig='customer_name_here'))

def generate_roc_curve(prediction_probabilities):
    if len(prediction_probabilities) == 1:
        # Create a random probability for the negative class label (non-fraud)
        random_prob = np.random.rand()
        prediction_probabilities = [prediction_probabilities[0], random_prob]

    fpr, tpr, _ = roc_curve([0, 1], prediction_probabilities)
    roc_auc = auc(fpr, tpr)
    # Plot the ROC curve
    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")

    # Convert the plot to an image
    image_stream = io.BytesIO()
    plt.savefig(image_stream, format='png')
    image_stream.seek(0)
    encoded_image = base64.b64encode(image_stream.getvalue()).decode('utf-8')
    plt.close()

    return encoded_image

@app.route('/fraud', methods=['GET', 'POST'])
def fraud():
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
        is_fraudulent = any(row[11] == 1 for row in data)

        # Perform the prediction and get the prediction probabilities
        prediction_probabilities = model.predict(input_data)[0]

        # Generate the ROC curve image
        roc_curve_image = generate_roc_curve(prediction_probabilities)

        # Pass the data, prediction, is_fraudulent flag, and ROC curve image to the template
        return render_template('fraud.html', data=data, prediction=prediction, is_fraudulent=is_fraudulent, roc_curve_image=roc_curve_image)

    return render_template('fraud.html')

def generate_bar_chart(labels, values):
    fig, ax = plt.subplots()
    ax.bar(labels, values)

    # Set chart title and labels
    ax.set_title('Transaction Fraud Distribution')
    ax.set_xlabel('Transaction Type')
    ax.set_ylabel('Count')

    # Convert the plot to an image
    image_stream = io.BytesIO()
    plt.savefig(image_stream, format='png')
    image_stream.seek(0)
    encoded_image = base64.b64encode(image_stream.getvalue()).decode('utf-8')
    plt.close()

    return encoded_image

# @app.route('/monitor', methods=['GET', 'POST'])
# def monitor():
#     if request.method == 'POST':
#         # Handle the form submission
#         step = int(request.form['step'])
#         type = request.form['type']
#         amount = float(request.form['amount'])
#         nameOrig = request.form['nameOrig']
#         oldbalanceOrg = float(request.form['oldbalanceOrg'])
#         newbalanceOrig = float(request.form['newbalanceOrig'])
#         newDest = request.form['newDest']
#         oldbalanceDest = float(request.form['oldbalanceDest'])
#         newbalanceDest = float(request.form['newbalanceDest'])
#         isFlaggedFraud = int(request.form['isFlaggedFraud'])

#         # Prepare the input data for prediction
#         input_data = np.array([[step, type, amount, nameOrig, oldbalanceOrg, newbalanceOrig, newDest,
#                                 oldbalanceDest, newbalanceDest, isFlaggedFraud]])
#         input_data = input_data.astype(np.float32)
#         input_data = np.reshape(input_data, (input_data.shape[0], 1, input_data.shape[1]))

#         # Make predictions using the model
#         prediction = model.predict(input_data)[0]

#         # Connect to the SQLite database
#         connection = sqlite3.connect(db_path)
#         cursor = connection.cursor()

#         # Execute a query to retrieve data for the given customer name
#         query = "SELECT * FROM train_data WHERE nameOrig = ?"
#         cursor.execute(query, (nameOrig,))

#         # Fetch the data from the query result
#         data = cursor.fetchall()

#         # Close the database connection
#         cursor.close()
#         connection.close()

#         # Check if any row has isFraud = 1
#         is_fraudulent = any(row[11] == 1 for row in data)

#         # Perform the prediction and get the prediction probabilities
#         prediction_probabilities = model.predict_proba(input_data)[0]

#         # Generate the ROC curve image
#         roc_curve_image = generate_roc_curve(prediction_probabilities)

#     else:
#         # Handle the GET request (initial page load and customer search)
#         search_type = request.args.get('searchType')
#         search_value = request.args.get('searchValue')

#         if search_type and search_value:
#             # Connect to the SQLite database
#             connection = sqlite3.connect(db_path)
#             cursor = connection.cursor()

#             # Execute a query to retrieve data based on the search type and value
#             if search_type == 'nameOrig':
#                 query = "SELECT * FROM train_data WHERE nameOrig = ?"
#             elif search_type == 'nameDest':
#                 query = "SELECT * FROM train_data WHERE nameDest = ?"

#             cursor.execute(query, (search_value,))

#             # Fetch the data from the query result
#             data = cursor.fetchall()

#             # Close the database connection
#             cursor.close()
#             connection.close()
            
#             # Check if any row has isFraud = 1
#             is_fraudulent = any(row[11] == 1 for row in data)

#             # Pass the data, is_fraudulent flag, and search_type/search_value to the template
#             return render_template('monitor.html', data=data, is_fraudulent=is_fraudulent,
#                                    searchType=search_type, searchValue=search_value)

#         # Render the empty form when it's a GET request without searchType and searchValue
#         return render_template('monitor.html')

@app.route('/monitor', methods=['GET', 'POST'])
def monitor():
    if request.method == 'POST':
        # Handle the form submission
        nameOrig = request.form['nameOrig']

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
        is_fraudulent = any(row[11] == 1 for row in data)

        # Generate the bar chart
        labels = ['Fraudulent', 'Non-Fraudulent']
        values = [0, 0]  # Initialize with 0 occurrences
        for row in data:
            if row[11] == 1:
                values[0] += 1
            elif row[11] == 0:
                values[1] += 1

        bar_chart = generate_bar_chart(labels, values)

        # Pass the data, is_fraudulent flag, and bar chart to the template
        return render_template('monitor.html', data=data, is_fraudulent=is_fraudulent, bar_chart=bar_chart)

    else:
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
            is_fraudulent = any(row[11] == 1 for row in data)

            # Generate the bar chart
            labels = ['Fraudulent', 'Non-Fraudulent']
            values = [0, 0]  # Initialize with 0 occurrences
            for row in data:
                if row[11] == 1:
                    values[0] += 1
                elif row[11] == 0:
                    values[1] += 1

            bar_chart = generate_bar_chart(labels, values)

            # Pass the data, is_fraudulent flag, search_type, search_value, and bar chart to the template
            return render_template('monitor.html', data=data, is_fraudulent=is_fraudulent,
                                   searchType=search_type, searchValue=search_value, bar_chart=bar_chart)

        # Render the empty form when it's a GET request without searchType and searchValue
        return render_template('monitor.html')

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
        print(prediction)

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
        is_fraudulent = any(row[11] == 1 for row in data)

        # Pass the data, prediction, and is_fraudulent flag to the template
        return render_template('result.html', data=data, prediction=prediction, is_fraudulent=is_fraudulent)
    else:
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
            is_fraudulent = any(row[11] == 1 for row in data)

            # Pass the data, is_fraudulent flag, and search_type/search_value to the template
            return render_template('result.html', data=data, is_fraudulent=is_fraudulent,
                                searchType=search_type, searchValue=search_value)

    return render_template('result.html')

if __name__ == '__main__':
    app.run(debug=True)
















