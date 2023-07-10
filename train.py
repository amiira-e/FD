import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import TomekLinks, EditedNearestNeighbours, OneSidedSelection
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import TimeDistributed, Conv1D, Flatten, Dense, LSTM
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam
from keras import regularizers
from keras.regularizers import l1
import pickle

# Load the dataset
df_sample_big = pd.read_csv(r"C:\\Users\\23059\\OneDrive\\Desktop\\Amiira\\new\\sample.csv")

# Label encoding
le = LabelEncoder()
df_sample_big['type'] = le.fit_transform(df_sample_big['type'])
df_sample_big['nameDest'] = le.fit_transform(df_sample_big['nameDest'])
df_sample_big['nameOrig'] = le.fit_transform(df_sample_big['nameOrig'])

X = df_sample_big.drop('isFraud', axis=1)
y = df_sample_big['isFraud']

# Upsampling via SMOTE
smote = SMOTE(sampling_strategy=0.55, random_state=0)
X_resampled, y_resampled = smote.fit_resample(X, y)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.1, stratify=y_resampled, random_state=2)

# Resample using TomekLinks
tomek_links = TomekLinks(sampling_strategy='majority')
X_train_resampled, y_train_resampled = tomek_links.fit_resample(X_train, y_train)

# Resample using EditedNearestNeighbours
enn = EditedNearestNeighbours(sampling_strategy='majority')
X_train_resampled_new, y_train_resampled_new = enn.fit_resample(X_train_resampled, y_train_resampled)

# Resample using One-Sided Selection
oss = OneSidedSelection(sampling_strategy='majority')
X_train_resampled_final, y_train_resampled_final = oss.fit_resample(X_train_resampled_new, y_train_resampled_new)

# Outlier replacement
def replace_outliers_with_mad(column):
    median = np.median(column)
    mad = np.median(np.abs(column - median))
    threshold = 2.5 * mad
    column[np.abs(column - median) > threshold] = median
    return column

for i in range(X_train_resampled_final.shape[1]):
    X_train_resampled_final.iloc[:, i] = replace_outliers_with_mad(X_train_resampled_final.iloc[:, i])

for i in range(X_test.shape[1]):
    X_test.iloc[:, i] = replace_outliers_with_mad(X_test.iloc[:, i])

# Scaling the data
scaler = StandardScaler()
model = scaler.fit(X_train_resampled_final)
X_train_resampled_final_scaled = model.transform(X_train_resampled_final)
X_test_scaled = model.transform(X_test)

# Reshape input data
X_train_reshaped = np.reshape(X_train_resampled_final_scaled, (X_train_resampled_final_scaled.shape[0], 1, X_train_resampled_final_scaled.shape[1]))

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

# Combine X_train_resampled_final and y_train_resampled_final into a DataFrame
df_combined = pd.DataFrame(data=X_test_scaled, columns=['step', 'amount','oldbalanceOrg','newbalanceOrig','oldbalanceDest','newbalanceDest','isFlaggedFraud','type','nameDest','nameOrig'])  # Replace 'feature1', 'feature2', ... with actual column names of X_train_resampled_final
df_combined['isFraud'] = y_test

# Print the combined DataFrame
print(df_combined)

df_combined.to_csv("C:\\Users\\23059\OneDrive\\Desktop\\Amiira\\Y3S1\\fyp\\testing.csv", index=False)

