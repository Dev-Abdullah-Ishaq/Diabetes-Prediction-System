# train_model.py

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
import joblib

# Load the dataset
data = pd.read_csv("D:/dppp/data/diabetes.csv")

# Data Preprocessing
X = data.drop("Outcome", axis=1)
Y = data['Outcome']
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Model Training
model = LogisticRegression(max_iter=1000, solver='lbfgs')
model.fit(X_train_scaled, Y_train)

# Save the model and scaler to files
joblib.dump(model, 'model/model.pkl')
joblib.dump(scaler, 'model/scaler.pkl')
