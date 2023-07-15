# Load CNN-LSTM model
model_path_new = os.path.join(os.path.dirname(__file__), 'static', 'cnnlstm.pkl')

with open(model_path_new, 'rb') as f:
    cnnlstm = pickle.load(f)

# Specify the path to the model file
model_path = os.path.join(os.path.dirname(__file__), 'static', 'modelnew.pkl')

# Load the trained model
with open(model_path, 'rb') as f:
    modelnew = pickle.load(f)

@app.route('/upload', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        # Get the uploaded file
        file = request.files['file']
        
         # Save the file to a secure location
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.root_path, 'static', filename)
        
        file.save(file_path)
        
        # Determine the selected model
        selected_model = request.form.get('selected_model')

        # Perform predictions and calculate metrics using the selected model
        if selected_model == 'modelnew':
            model = modelnew
        elif selected_model == 'cnnlstm':
            model = cnnlstm
        else:
            return 'Invalid model selection'
        
        predictions, target, metrics, confusion = predict_and_calculate_metrics(file_path, model)
        
        # Return the predictions, metrics, and confusion matrix as a response
        return render_template('upload.html', results=list(zip(predictions, target)), metrics=metrics, confusion=confusion)

    return render_template('upload.html')

# Good: fraud_data and predictions >= 0.5

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



############
