from flask import Flask, render_template, request
import pandas as pd
from sklearn.preprocessing import StandardScaler
import joblib

app = Flask(__name__, template_folder='templates')

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict')
def predict():
    return render_template('index.html')

@app.route('/predictmale')
def predictmale():
    return render_template('indexmale.html')

@app.route('/RiskCal')
def Risk():
    return render_template('risk.html')

@app.route('/contact')
def contact():
    return render_template('contact.html')

@app.route('/aboutus')
def aboutus():
    return render_template('Aboutus.html')

@app.route('/indexmale')
def male():
    return render_template('indexmale.html')

@app.route('/result', methods=['POST'])
def result():
    pregnancies = float(request.form['n1'])
    glucose = float(request.form['n2'])
    blood_pressure = float(request.form['n3'])
    skin_thickness = float(request.form['n4'])
    insulin = float(request.form['n5'])
    bmi = float(request.form['n6'])
    diabetes_pedigree_function = float(request.form['n7'])
    age = float(request.form['n8'])

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
    model = joblib.load('model.pkl')
    scaler = joblib.load('scaler.pkl')

    # Preprocess the input data
    input_data_scaled = scaler.transform(input_data)

    # Make predictions using the loaded model
    predictions = model.predict(input_data_scaled)

    result_message = "You have diabetes." if predictions[0] == 1 else "You don't have diabetes."

    return render_template('result.html', result=result_message)

@app.route('/resultmale', methods=['POST'])
def resultmale():
    glucose = float(request.form['nn1'])
    blood_pressure = float(request.form['nn2'])
    skin_thickness = float(request.form['nn3'])
    insulin = float(request.form['nn4'])
    bmi = float(request.form['nn5'])
    diabetes_pedigree_function = float(request.form['nn6'])
    age = float(request.form['nn7'])

    input_data = pd.DataFrame({
        'Pregnancies': [0],  # For males
        'Glucose': [glucose],
        'BloodPressure': [blood_pressure],
        'SkinThickness': [skin_thickness],
        'Insulin': [insulin],
        'BMI': [bmi],
        'DiabetesPedigreeFunction': [diabetes_pedigree_function],
        'Age': [age]
    })

    # Load the trained model and scaler for male prediction
    model_male = joblib.load('model_male.pkl')
    scaler_male = joblib.load('scaler_male.pkl')

    # Preprocess the input data for male prediction
    input_data_scaled = scaler_male.transform(input_data)

    # Make predictions using the loaded model for males
    predictions = model_male.predict(input_data_scaled)

    result_message = "You have diabetes." if predictions[0] == 1 else "You don't have diabetes."

    return render_template('result.html', result=result_message)

if __name__ == '__main__':
    app.run(debug=True)
