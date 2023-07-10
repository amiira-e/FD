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
        return render_template('fraud.html', data=data, prediction=binary_predictions[0], is_fraudulent=is_fraudulent, roc_curve_image=roc_curve_image,
                               precision=precision, recall=recall, f1_score=f1_score)

    return render_template('fraud.html')