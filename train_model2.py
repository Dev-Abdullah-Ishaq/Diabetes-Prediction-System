import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
import joblib

# Load the dataset
data = pd.read_csv("Diabetes Classification.csv")

# Data Preprocessing
X = data.drop("Diagnosis", axis=1)  # Exclude the target variable 'Diagnosis'
Y = data['Diagnosis']

# Perform one-hot encoding on the 'Gender' column
encoder_gender = OneHotEncoder(sparse=False)
X_encoded_gender = encoder_gender.fit_transform(X[['Gender']])
X_encoded_gender_df = pd.DataFrame(X_encoded_gender, columns=encoder_gender.get_feature_names_out(['Gender']))

# Perform one-hot encoding on the 'Blood Pressure' column
encoder_bp = OneHotEncoder(sparse=False)
X_encoded_bp = encoder_bp.fit_transform(X[['Blood Pressure']])
X_encoded_bp_df = pd.DataFrame(X_encoded_bp, columns=encoder_bp.get_feature_names_out(['Blood Pressure']))

# Perform one-hot encoding on other categorical columns
categorical_cols = ['Family History of Diabetes', 'Smoking', 'Diet', 'Exercise']
encoder_other = OneHotEncoder(sparse=False)
X_encoded_other = encoder_other.fit_transform(X[categorical_cols])
X_encoded_other_df = pd.DataFrame(X_encoded_other, columns=encoder_other.get_feature_names_out(categorical_cols))

# Combine the one-hot encoded columns and drop the original categorical columns
X = pd.concat([X.drop(columns=['Gender', 'Blood Pressure'] + categorical_cols), X_encoded_gender_df, X_encoded_bp_df, X_encoded_other_df], axis=1)

# Split the data
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.1)

scaler1 = StandardScaler()
X_train_scaled = scaler1.fit_transform(X_train)
X_test_scaled = scaler1.transform(X_test)

# Model Training
model1 = LogisticRegression(max_iter=1000, solver='lbfgs')
model1.fit(X_train_scaled, Y_train)

# Save the model and scaler to files
joblib.dump(model1, 'modell.pkl')
joblib.dump(scaler1, 'scalerr.pkl')