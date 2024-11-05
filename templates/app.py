from flask import Flask, render_template, request
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import joblib

app = Flask(__name__)

# Define the route for the main page
@app.route('/')
def index():
    return render_template('home.html')

# Define the route for handling form submission and displaying results
@app.route('/result', methods=['POST'])
def result():
    # Retrieve form input values
    pregnancies = float(request.form['n1'])
    glucose = float(request.form['n2'])
    blood_pressure = float(request.form['n3'])
    skin_thickness = float(request.form['n4'])
    insulin = float(request.form['n5'])
    bmi = float(request.form['n6'])
    diabetes_pedigree_function = float(request.form['n7'])
    age = float(request.form['n8'])

    # Create a DataFrame for the input data
    input_data = pd.DataFrame({
        'Pregnancies': [pregnancies],
        'Glucose': [glucose],
        'BloodPressure': [blood_pressure],
        'SkinThickness': [skin_thickness],
        'Insulin': [insulin],
        'BMI': [bmi],
        'DiabetesPedigreeFunction': [diabetes_pedigree_function],
        'Age': [age]
    })

    # Load the trained model and scaler
    model = LogisticRegression(max_iter=1000, solver='lbfgs')
    model=joblib.load('model.pkl')  # Load your trained model
    scaler = StandardScaler()
    scaler=joblib.load('scaler.pkl')  # Load your scaler

    # Preprocess the input data
    input_data_scaled = scaler.transform(input_data)

    # Make predictions using the loaded model
    predictions = model.predict(input_data_scaled)
    result_message = ""

    # Determine the result message
    if predictions[0] == 1:
        result_message = "You have diabetes."
    else:
        result_message = "You don't have diabetes."

    return render_template('result.html', result=result_message)

if __name__ == '__main__':
    app.run(debug=True)

