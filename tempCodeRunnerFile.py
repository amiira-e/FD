# Create the LSTM model
modelnew = Sequential()
modelnew.add(LSTM(64, input_shape=(1, 10), return_sequences=False))
modelnew.add(Dense(1, activation='sigmoid'))

opt = Adam(learning_rate=0.0012541)

# Compile the model
modelnew.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])

# Define early stopping callback
early_stopping = EarlyStopping(monitor='val_loss', patience=5, verbose=1)

# Train the model
history = modelnew.fit(X_train_reshaped, y_train_resampled_final, epochs=10, batch_size=64, validation_split=0.1, callbacks=[early_stopping], verbose=1)

# Save the trained model using pickle
with open('modelnew.pkl', 'wb') as file:
    pickle.dump(modelnew, file)

# Reshape test data for predictions
X_test_reshaped = np.reshape(X_test_scaled, (X_test_scaled.shape[0], 1, X_test_scaled.shape[1]))

# Make predictions on test data
y_pred_prob = modelnew.predict(X_test_reshaped)
y_pred = (y_pred_prob > 0.5).astype(int)

# Evaluate the model
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score

# Threshold tuning
thresholds = np.arange(0.1, 1.0, 0.1)
best_f1_score = 0
best_threshold = 0

for threshold in thresholds:
    y_pred = (y_pred_prob > threshold).astype(int)
    f1 = f1_score(y_test, y_pred)

    if f1 > best_f1_score:
        best_f1_score = f1
        best_threshold = threshold

# Apply best threshold to obtain final predictions
y_pred = (y_pred_prob > best_threshold).astype(int)

# Compute evaluation metrics and confusion matrix
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)

print("Best Threshold:", best_threshold)
print("Precision:", precision)
print("Recall:", recall)
print("F1 Score:", f1)
print("Confusion Matrix:")
print(cm)

from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score

# Make predictions on test data
y_pred_prob = model.predict(X_test_reshaped)

# Threshold tuning
thresholds = np.arange(0.1, 1.0, 0.1)
best_f1_score = 0
best_threshold = 0

for threshold in thresholds:
    y_pred = (y_pred_prob > threshold).astype(int)
    f1 = f1_score(y_test, y_pred)

    if f1 > best_f1_score:
        best_f1_score = f1
        best_threshold = threshold

# Apply best threshold to obtain final predictions
y_pred = (y_pred_prob > best_threshold).astype(int)

# Compute evaluation metrics and confusion matrix
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)

print("Best Threshold:", best_threshold)
print("Precision:", precision)
print("Recall:", recall)
print("F1 Score:", f1)
print("Confusion Matrix:")
print(cm)

import pickle
# Save the trained model using pickle
with open('modelnew.pkl', 'wb') as file:
    pickle.dump(modelnew, file)

import pandas as pd