from flask import Flask, render_template, redirect, url_for, request
import sqlite3
import pickle
import numpy as np
import tensorflow as tf
import os
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
import io
import base64
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import precision_score, recall_score, f1_score, roc_curve, precision_recall_curve

app = Flask(__name__)

# Create a custom enumerate function for Jinja2
def jinja2_enumerate(iterable, start=0):
    return enumerate(iterable, start=start)

# Add the custom enumerate function to the template environment
app.jinja_env.globals['enumerate'] = jinja2_enumerate

# Configure the database connection
db_path = 'C:/Users/23059/OneDrive/Desktop/Amiira/DB_test/Pastdata.db'

# Specify the path to the model file
model_path = os.path.join(os.path.dirname(__file__), 'static', 'modelnew.pkl')

# Load the trained model
with open(model_path, 'rb') as f:
    model = pickle.load(f)

# model = tf.keras.models.load_model(model_path)

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

# @app.route('/fraud', methods=['GET', 'POST'])
# def fraud():
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
#         prediction_probabilities = model.predict(input_data)[0]

#         # Generate the ROC curve image
#         roc_curve_image = generate_roc_curve(prediction_probabilities)

#         # Pass the data, prediction, is_fraudulent flag, and ROC curve image to the template
#         return render_template('fraud.html', data=data, prediction=prediction, is_fraudulent=is_fraudulent, roc_curve_image=roc_curve_image)

#     return render_template('fraud.html')

from sklearn import metrics

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

        # Convert the true labels to binary values (0 and 1)
        true_labels = [row[11] for row in data]

        # Convert prediction probabilities to binary predictions using a threshold
        threshold = 0.7  # Adjust the threshold as per your requirement
        binary_predictions = [1 if prob >= threshold else 0 for prob in prediction_probabilities]

        # Calculate precision, recall, and F1 score
        precision = metrics.precision_score(true_labels, binary_predictions)
        recall = metrics.recall_score(true_labels, binary_predictions)
        f1_score = metrics.f1_score(true_labels, binary_predictions)

        # Pass the data, prediction, is_fraudulent flag, ROC curve image, precision, recall, and F1 score to the template
        return render_template('fraud.html', data=data, prediction=binary_predictions, is_fraudulent=is_fraudulent, roc_curve_image=roc_curve_image,
                               precision=precision, recall=recall, f1_score=f1_score)

    return render_template('fraud.html')


import matplotlib.pyplot as plt

def generate_bar_chart(labels, values):
    fig, ax = plt.subplots()
    ax.bar(labels, values)

    # Set chart title and labels
    ax.set_title('Transaction Fraud Distribution', color='white')  # Set the title color to white
    ax.set_xlabel('Transaction Type', color='white')  # Set the x-label color to white
    ax.set_ylabel('Count', color='white')  # Set the y-label color to white
    ax.set_facecolor('none')  # Remove the white background

    # Set the color of tick labels to white
    ax.tick_params(colors='white')

    # Convert the plot to an image
    image_stream = io.BytesIO()
    plt.savefig(image_stream, format='png', transparent=True)  # Save the plot with transparent background
    image_stream.seek(0)
    encoded_image = base64.b64encode(image_stream.getvalue()).decode('utf-8')
    plt.close()

    return encoded_image

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

        # Generate the pie chart
        transaction_types = [row[8] for row in data]
        type_counts = {}
        for transaction_type in transaction_types:
            if transaction_type in type_counts:
                type_counts[transaction_type] += 1
            else:
                type_counts[transaction_type] = 1

        pie_chart = generate_pie_chart(type_counts.keys(), type_counts.values())

        # Pass the data, is_fraudulent flag, bar chart, and pie chart to the template
        return render_template('monitor.html', data=data, is_fraudulent=is_fraudulent,
                               bar_chart=bar_chart, pie_chart=pie_chart)
    
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

            # Generate the pie chart
            transaction_types = [row[8] for row in data]
            type_counts = {}
            for transaction_type in transaction_types:
                if transaction_type in type_counts:
                    type_counts[transaction_type] += 1
                else:
                    type_counts[transaction_type] = 1

            pie_chart = generate_pie_chart(type_counts.keys(), type_counts.values())

            # Pass the data, is_fraudulent flag, search_type, search_value, bar chart, and pie chart to the template
            return render_template('monitor.html', data=data, is_fraudulent=is_fraudulent,
                                   searchType=search_type, searchValue=search_value, bar_chart=bar_chart, pie_chart=pie_chart)

        # Render the empty form when it's a GET request without searchType and searchValue
        return render_template('monitor.html')

import matplotlib.pyplot as plt

def generate_pie_chart(labels, values):
    # Create a pie chart with custom colors and explode one or more slices
    colors = ['#A63922', '#106675', '#4F0B73', '#ffcc99']  # Custom colors for slices

    fig, ax = plt.subplots()
    ax.pie(values, labels=labels, colors=colors,autopct='%1.1f%%', startangle=90)

    # Customize additional properties of the pie chart
    ax.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle
    ax.set_title('Transaction Type Distribution', color='white')  # Set the title of the pie chart and make it white
    ax.set_facecolor('none')  # Remove the white background

    # Set the text color of labels and percentages to white
    for text in ax.texts:
        text.set_color('white')

    # Convert the plot to an image
    image_stream = io.BytesIO()
    plt.savefig(image_stream, format='png', transparent=True)  # Save the plot with transparent background
    image_stream.seek(0)
    encoded_image = base64.b64encode(image_stream.getvalue()).decode('utf-8')
    plt.close()

    return encoded_image


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

import csv
from werkzeug.utils import secure_filename

# @app.route('/upload', methods=['GET', 'POST'])
# def upload():
#     if request.method == 'POST':
#         # Get the uploaded file
#         file = request.files['file']
        
#         # Save the file to a secure location
#         filename = secure_filename(file.filename)
#         file_path = os.path.join(app.root_path, filename)
#         file.save(file_path)
        
#         # Perform predictions and calculate metrics using the model
#         predictions, target, metrics = predict_and_calculate_metrics(file_path)
        
#         # Return the predictions and metrics as a response
#         return render_template('upload.html', results=list(zip(predictions, target)), metrics=metrics)

#     return render_template('upload.html')

# def predict_and_calculate_metrics(file_path):
#     # Read the CSV file
#     data = []
#     target = []
#     with open(file_path, 'r') as csvfile:
#         reader = csv.reader(csvfile)
#         next(reader)  # Skip the header row
#         for row in reader:
#             data.append(row[:-1])  # Exclude the last column (target variable)
#             target.append(float(row[-1]))  # Convert target variable to numeric type

#     # Prepare the input data for prediction
#     input_data = np.array(data, dtype=np.float32)
#     input_data = np.reshape(input_data, (input_data.shape[0], 1, input_data.shape[1]))

#     # Make predictions using the model
#     predictions = model.predict(input_data)

#     # Apply threshold and convert predictions to binary values
#     binary_predictions = (predictions >=  0.000082354).astype(int) #0.0000082354

#     # Calculate evaluation metrics
#     precision = precision_score(target, binary_predictions)
#     recall = recall_score(target, binary_predictions)
#     f1 = f1_score(target, binary_predictions)

#     metrics = {
#         'precision': precision,
#         'recall': recall,
#         'f1_score': f1
#     }

#     return binary_predictions, target, metrics

from sklearn.metrics import confusion_matrix

@app.route('/upload', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        # Get the uploaded file
        file = request.files['file']
        
        # Save the file to a secure location
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.root_path, filename)
        file.save(file_path)
        
        # Perform predictions and calculate metrics using the model
        predictions, target, metrics, confusion = predict_and_calculate_metrics(file_path)
        
        # Return the predictions, metrics, and confusion matrix as a response
        return render_template('upload.html', results=list(zip(predictions, target)), metrics=metrics, confusion=confusion)

    return render_template('upload.html')

# Good: fraud_data and predictions >= 0.5

def predict_and_calculate_metrics(file_path):
    # Read the CSV file
    data = []
    target = []
    with open(file_path, 'r') as csvfile:
        reader = csv.reader(csvfile)
        next(reader)  # Skip the header row
        for row in reader:
            data.append(row[:-1])  # Exclude the last column (target variable)
            target.append(float(row[-1]))  # Convert target variable to numeric type

    # Prepare the input data for prediction
    input_data = np.array(data, dtype=np.float32)
    input_data = np.reshape(input_data, (input_data.shape[0], 1, input_data.shape[1]))

    # Make predictions using the model
    predictions = model.predict(input_data)

    # Apply threshold and convert predictions to binary values
    binary_predictions = (predictions >= 0.4).astype(int) #0.0000082354  #predictions >= 0.5 predictions >= 0.6

    # Calculate evaluation metrics
    precision = precision_score(target, binary_predictions)
    recall = recall_score(target, binary_predictions)
    f1 = f1_score(target, binary_predictions)

    # Calculate confusion matrix
    confusion = confusion_matrix(target, binary_predictions).tolist()

    metrics = {
        'precision': precision,
        'recall': recall,
        'f1_score': f1
    }

    return binary_predictions, target, metrics, confusion

if __name__ == '__main__':
    app.run(debug=True)
















