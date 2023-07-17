from flask import Flask, render_template, redirect, url_for, request
import sqlite3
import pickle
import numpy as np
import tensorflow as tf
import os
from sklearn.metrics import roc_curve, auc
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot
import io
import base64
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import precision_score, recall_score, f1_score, roc_curve, precision_recall_curve
from tabulate import tabulate
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest
from sklearn.metrics import classification_report

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
    modelnew = pickle.load(f)

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
        prediction_probabilities = modelnew.predict_proba(input_data)[0]

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
        prediction = modelnew.predict(input_data)[0]

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
        # is_fraudulent = any(row[11] == 1 for row in data)

        # Perform the prediction and get the prediction probabilities
        prediction_probabilities = modelnew.predict(input_data)[0]

        # Generate the ROC curve image
        roc_curve_image = generate_roc_curve(prediction_probabilities)

        # Convert the true labels to binary values (0 and 1)
        true_labels = [row[11] for row in data]

        # Convert prediction probabilities to binary predictions using a threshold
        threshold = 0.5  # Adjust the threshold as per your requirement  #0.000000000001 
        binary_predictions = [1 if prob >= threshold else 0 for prob in prediction_probabilities]

        # Set is_fraudulent to True if binary_predictions is equal to 1
        if 1 in binary_predictions:
            is_fraudulent = True
        else:
            is_fraudulent = False
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

# @app.route('/monitor', methods=['GET', 'POST'])
# def monitor():
#     if request.method == 'POST':
#         # Handle the form submission
#         nameOrig = request.form['nameOrig']

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

#         # Generate the bar chart
#         labels = ['Fraudulent', 'Non-Fraudulent']
#         values = [0, 0]  # Initialize with 0 occurrences
#         for row in data:
#             if row[11] == 1:
#                 values[0] += 1
#             elif row[11] == 0:
#                 values[1] += 1

#         bar_chart = generate_bar_chart(labels, values)

#         # Generate the pie chart
#         transaction_types = [row[8] for row in data]
#         type_counts = {}
#         for transaction_type in transaction_types:
#             if transaction_type in type_counts:
#                 type_counts[transaction_type] += 1
#             else:
#                 type_counts[transaction_type] = 1

#         pie_chart = generate_pie_chart(type_counts.keys(), type_counts.values())

#         # Pass the data, is_fraudulent flag, bar chart, and pie chart to the template
#         return render_template('monitor.html', data=data, is_fraudulent=is_fraudulent,
#                                bar_chart=bar_chart, pie_chart=pie_chart)
    
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

#             # Generate the bar chart
#             labels = ['Fraudulent', 'Non-Fraudulent']
#             values = [0, 0]  # Initialize with 0 occurrences
#             for row in data:
#                 if row[11] == 1:
#                     values[0] += 1
#                 elif row[11] == 0:
#                     values[1] += 1

#             bar_chart = generate_bar_chart(labels, values)

#             # Generate the pie chart
#             transaction_types = [row[8] for row in data]
#             type_counts = {}
#             for transaction_type in transaction_types:
#                 if transaction_type in type_counts:
#                     type_counts[transaction_type] += 1
#                 else:
#                     type_counts[transaction_type] = 1

#             pie_chart = generate_pie_chart(type_counts.keys(), type_counts.values())

#             # Pass the data, is_fraudulent flag, search_type, search_value, bar chart, and pie chart to the template
#             return render_template('monitor.html', data=data, is_fraudulent=is_fraudulent,
#                                    searchType=search_type, searchValue=search_value, bar_chart=bar_chart, pie_chart=pie_chart)

#         # Render the empty form when it's a GET request without searchType and searchValue
#         return render_template('monitor.html')

import itertools
# Custom filter to enable zip_longest in Jinja2 templates
@app.template_filter('zip_longest')
def zip_longest_filter(*args, fillvalue=None):
    return itertools.zip_longest(*args, fillvalue=fillvalue)

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


        # Pass the data, is_fraudulent flag, bar chart, pie chart, recipients, and counts to the template
        return render_template('monitor.html', data=data, is_fraudulent=is_fraudulent,
                               bar_chart=bar_chart, pie_chart=pie_chart,)
    
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
import io
import base64

# def generate_recipient_chart_image(recipients_counts):
#     recipients = [entry[0] for entry in recipients_counts]  # Access recipient information from the tuple
#     counts = [entry[1] for entry in recipients_counts]  # Access count information from the tuple

#     # Generate the bar chart
#     plt.figure(figsize=(8, 6))
#     plt.bar(recipients, counts)
#     plt.xlabel('Recipient')
#     plt.ylabel('Count')
#     plt.title('Recipient Transaction Counts')

#     # Convert the chart to a base64 encoded string
#     buffer = io.BytesIO()
#     plt.savefig(buffer, format='png')
#     buffer.seek(0)
#     chart_image = base64.b64encode(buffer.read()).decode()
#     plt.close()

#     return chart_image

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
        prediction = modelnew.predict(input_data)[0]
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
from sklearn.metrics import confusion_matrix

# Load CNN-LSTM model
model_path_new = os.path.join(os.path.dirname(__file__), 'static', 'cnnlstm.pkl')

with open(model_path_new, 'rb') as f:
    cnnlstm = pickle.load(f)

# Specify the paths to the model files
model_path_new = os.path.join(os.getcwd(), 'static', 'cnnlstm.pkl')
model_path = os.path.join(os.getcwd(), 'static', 'modelnew.pkl')

# Load the trained model
with open(model_path_new, 'rb') as f:
    cnnlstm= pickle.load(f)

# Load the trained model
with open(model_path, 'rb') as f:
    modelnew = pickle.load(f)

# GOOOODD
# Define your route for the page containing the prediction results
@app.route('/upload', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        # Get the uploaded file
        file = request.files['file']

        # Determine the selected model
        selected_model = request.form.get('selected_model')

        # Perform predictions and calculate metrics using the selected model
        if selected_model == 'modelnew':
            model = modelnew
        elif selected_model == 'cnnlstm':
            model = cnnlstm
        else:
            return 'Invalid model selection'

        # Save the file to a secure location
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.root_path, 'static', filename)
        file.save(file_path)

        predictions, target, metrics, confusion = predict_and_calculate_metrics(file_path, model)

        # Return the predictions, metrics, and confusion matrix as a response
        return render_template('upload.html', results=list(zip(predictions, target)), metrics=metrics, confusion=confusion)

    return render_template('upload.html')

def predict_and_calculate_metrics(file_path, model):
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
    if model == cnnlstm:
        input_data = np.array(data, dtype=np.float32)
        input_data = np.reshape(input_data, (input_data.shape[0], 10, 1))  # Reshape the input data for "Model 2" (cnnlstm)
    else:
        input_data = np.array(data, dtype=np.float32)
        input_data = np.reshape(input_data, (input_data.shape[0], 1, input_data.shape[1]))
    # Prepare the input data for prediction
    # input_data = np.array(data, dtype=np.float32)
   

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

from flask import jsonify, render_template
import pandas as pd
import joblib
import io

# @app.route('/anomaly', methods=['GET', 'POST'])
# def anomaly():
#     if request.method == 'POST':
#         # Load your dataset and perform any necessary preprocessing
#         df = pd.read_csv("C:\\Users\\23059\\OneDrive\\Desktop\\Amiira\\Y3S1\\fyp\\cleandata.csv")

#         # Load the pre-trained isolation forest model
#         isolationforest = joblib.load('static/isolationforest.pkl')

#         # Extract the features for anomaly detection (e.g., 'amount' column)
#         X = df[['amount']]

#         # Predict outliers for the 'amount' variable
#         outlier_scores = isolationforest.decision_function(X)
#         outlier_predictions = isolationforest.predict(X)

#         # Create a DataFrame with the original data and outlier scores
#         df_outliers = pd.DataFrame({'nameOrig': df['nameOrig'], 'amount': X['amount'], 'Outlier Score': outlier_scores})

#         # Sort the DataFrame by outlier scores in descending order
#         df_outliers_sorted = df_outliers.sort_values(by='Outlier Score', ascending=False)

#         # Retrieve the customer(s) with the highest outlier score
#         num_rows = int(request.form.get('num_rows'))
#         highest_outlier_customers = df_outliers_sorted.head(num_rows)['nameOrig'].tolist()

#         # Perform anomaly detection and generate the plot
#         plt.figure()  # Create a new figure
#         plt.scatter(df_outliers['amount'], df_outliers['Outlier Score'], c=outlier_predictions, cmap='coolwarm')
#         plt.xlabel('Transaction Amount', color='white')
#         plt.ylabel('Outlier Score', color='white')
#         plt.title('Anomaly Detection: Transaction Amount vs Outlier Score', color='white')
#         colorbar = plt.colorbar(orientation='vertical')
#         colorbar.set_label('Outlier Prediction', color='white')
#         colorbar.ax.yaxis.set_tick_params(color='white')  # Set tick labels color to white
#         plt.tick_params(colors='white')

#         # Save the plot to a BytesIO buffer
#         buffer = io.BytesIO()
#         plt.savefig(buffer, format='png', transparent=True)
#         buffer.seek(0)

#         # Convert the buffer to base64 encoded string
#         plot_data = base64.b64encode(buffer.getvalue()).decode('utf-8')

#         amounts = df_outliers['amount'].tolist()
#         # Convert the outlier predictions to anomaly values (-1 for outliers, 1 for inliers)
#         anomaly_values = [-1 if prediction == -1 else 1 for prediction in outlier_predictions]

#         # # Return the HTML template with the classification report and plot data
#         # return render_template('anomaly.html', outlier_scores=outlier_scores, highest_outlier_customers=highest_outlier_customers, show_plot=True, plot_data=plot_data, amounts=amounts, anomaly_values=anomaly_values)


#     # Render the HTML template for GET requests
#     return render_template('anomaly.html', show_plot=False)

@app.route('/anomaly', methods=['GET', 'POST'])
def anomaly():
    if request.method == 'POST':
        # Load your dataset and perform any necessary preprocessing
        df = pd.read_csv("C:\\Users\\23059\\OneDrive\\Desktop\\Amiira\\Y3S1\\fyp\\cleandata.csv")

        # Load the pre-trained isolation forest model
        isolationforest = joblib.load('static/isolationforest.pkl')

        # Extract the features for anomaly detection (e.g., 'amount' column)
        X = df[['amount']]

        # Predict outliers for the 'amount' variable
        outlier_scores = isolationforest.decision_function(X)
        outlier_predictions = isolationforest.predict(X)

        # Create a DataFrame with the original data and outlier scores
        df_outliers = pd.DataFrame({'nameOrig': df['nameOrig'], 'amount': X['amount'], 'Outlier Score': outlier_scores})

        # Sort the DataFrame by outlier scores in descending order
        df_outliers_sorted = df_outliers.sort_values(by='Outlier Score', ascending=False)

        # Retrieve the customer(s) with the highest outlier score
        num_rows = request.form.get('num_rows')
        if num_rows:
            num_rows = int(num_rows)
        else:
            num_rows = 0
        highest_outlier_customers = df_outliers_sorted.head(num_rows)['nameOrig'].tolist()

        # Get the customer name from the form input
        customer_name = request.form.get('customer_name')

        amounts = df_outliers['amount'].tolist()

        # Convert the outlier predictions to anomaly values (-1 for outliers, 1 for inliers)
        anomaly_values = [-1 if prediction == -1 else 1 for prediction in outlier_predictions]


        # Perform anomaly detection and generate the plot
        plt.figure()  # Create a new figure
        plt.scatter(df_outliers['amount'], df_outliers['Outlier Score'], c=outlier_predictions, cmap='coolwarm')
        plt.xlabel('Transaction Amount', color='white')
        plt.ylabel('Outlier Score', color='white')
        plt.title('Anomaly Detection: Transaction Amount vs Outlier Score', color='white')
        colorbar = plt.colorbar(orientation='vertical')
        colorbar.set_label('Outlier Prediction', color='white')
        colorbar.ax.yaxis.set_tick_params(color='white')  # Set tick labels color to white
        plt.tick_params(colors='white')

        # Save the plot to a BytesIO buffer
        buffer = io.BytesIO()
        plt.savefig(buffer, format='png', transparent=True)
        buffer.seek(0)

        # Convert the buffer to base64 encoded string
        plot_data = base64.b64encode(buffer.getvalue()).decode('utf-8')
        # Return the HTML template with the specific customer's information
        return render_template('anomaly.html', outlier_scores=outlier_scores,
                               highest_outlier_customers=highest_outlier_customers,
                               show_plot=True, plot_data=plot_data, amounts=amounts,
                               anomaly_values=anomaly_values)

    # Render the HTML template for GET requests
    return render_template('anomaly.html', show_plot=False)

# @app.route('/anomaly', methods=['GET', 'POST'])
# def anomaly():
#     if request.method == 'POST':
#         # Load your dataset and perform any necessary preprocessing
#         df = pd.read_csv("C:\\Users\\23059\\OneDrive\\Desktop\\Amiira\\Y3S1\\fyp\\cleandata.csv")

#         # Load the pre-trained isolation forest model
#         isolationforest = joblib.load('static/isolationforest.pkl')

#         # Extract the features for anomaly detection (e.g., 'amount' column)
#         X = df[['amount']]

#         # Predict outliers for the 'amount' variable
#         outlier_scores = isolationforest.decision_function(X)
#         outlier_predictions = isolationforest.predict(X)

#         # Create a DataFrame with the original data and outlier scores
#         df_outliers = pd.DataFrame({'nameOrig': df['nameOrig'], 'amount': X['amount'], 'Outlier Score': outlier_scores})

#         # Sort the DataFrame by outlier scores in descending order
#         df_outliers_sorted = df_outliers.sort_values(by='Outlier Score', ascending=False)

#         # Retrieve the customer(s) with the highest outlier score
#         num_rows = request.form.get('num_rows')
#         if num_rows:
#             num_rows = int(num_rows)
#         else:
#             num_rows = 0

#         highest_outlier_customers = df_outliers_sorted.head(num_rows)['nameOrig'].tolist()

#         # Get the customer name from the form input
#         customer_name = request.form.get('customer_name')

#         # Filter the data for the specified customer name
#         customer_data = df_outliers[df_outliers['nameOrig'] == customer_name]

#         amounts = df_outliers['amount'].tolist()

#         # Convert the outlier predictions to anomaly values (-1 for outliers, 1 for inliers)
#         anomaly_values = [-1 if prediction == -1 else 1 for prediction in outlier_predictions]

#         if num_rows == 0 and customer_name:
#             # Filter the data for the specified customer name
#             customer_data = df_outliers[df_outliers['nameOrig'] == customer_name]

#             if not customer_data.empty:
#                 # Retrieve the specific customer's information
#                 customer_outlier_score = customer_data['Outlier Score'].values[0]
#                 customer_amount = customer_data['amount'].values[0]
#                 customer_anomaly_value = -1 if outlier_predictions[customer_data.index[0]] == -1 else 1

#         if not customer_data.empty:
#             # Retrieve the specific customer's information
#             customer_outlier_score = customer_data['Outlier Score'].values[0]
#             customer_amount = customer_data['amount'].values[0]
#             customer_anomaly_value = -1 if outlier_predictions[customer_data.index[0]] == -1 else 1
#         else:
#             # Set default values if the customer data is not found
#             customer_outlier_score = 0
#             customer_amount = 0
#             customer_anomaly_value = 0

#         # Perform anomaly detection and generate the plot
#         plt.figure()  # Create a new figure
#         plt.scatter(df_outliers['amount'], df_outliers['Outlier Score'], c=outlier_predictions, cmap='coolwarm')
#         plt.xlabel('Transaction Amount', color='white')
#         plt.ylabel('Outlier Score', color='white')
#         plt.title('Anomaly Detection: Transaction Amount vs Outlier Score', color='white')
#         colorbar = plt.colorbar(orientation='vertical')
#         colorbar.set_label('Outlier Prediction', color='white')
#         colorbar.ax.yaxis.set_tick_params(color='white')  # Set tick labels color to white
#         plt.tick_params(colors='white')

#         # Save the plot to a BytesIO buffer
#         buffer = io.BytesIO()
#         plt.savefig(buffer, format='png', transparent=True)
#         buffer.seek(0)

#         # Convert the buffer to base64 encoded string
#         plot_data = base64.b64encode(buffer.getvalue()).decode('utf-8')

#         # Return the HTML template with the specific customer's information
#         return render_template('anomaly.html', outlier_scores=outlier_scores,
#                                highest_outlier_customers=highest_outlier_customers,
#                                show_plot=True, plot_data=plot_data, amounts=amounts,
#                                anomaly_values=anomaly_values, customer_name=customer_name,
#                                customer_outlier_score=customer_outlier_score,
#                                customer_amount=customer_amount, customer_anomaly_value=customer_anomaly_value)

#     # Render the HTML template for GET requests
#     return render_template('anomaly.html', show_plot=False)


if __name__ == '__main__':
    app.run(debug=True)













