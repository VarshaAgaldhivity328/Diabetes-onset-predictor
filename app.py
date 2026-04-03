from flask import Flask, render_template, request
import pandas as pd
import joblib
import os

app = Flask(__name__)

# Load models safely
def load_models():
    try:
        model = joblib.load('diabetes_model.pkl')
        encoders = joblib.load('label_encoders.pkl')
        return model, encoders
    except FileNotFoundError:
        return None, None

model, encoders = load_models()

@app.route('/')
def home():
    if model is None or encoders is None:
        return "Model files not found! Please run 'python training.py' first.", 500
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if model is None or encoders is None:
        return "Model not loaded.", 500
        
    try:
        # Get data from form
        gender = request.form['gender']
        age = float(request.form['age'])
        smoking_history = request.form['smoking_history']
        
        hypertension_str = request.form['hypertension']
        bmi = float(request.form['bmi'])
        
        heart_disease_str = request.form['heart_disease']
        hba1c_level = float(request.form['hba1c_level'])
        blood_glucose_level = float(request.form['blood_glucose_level'])

        # Convert simple categoricals
        hypertension_val = 1 if hypertension_str == "Yes" else 0
        heart_disease_val = 1 if heart_disease_str == "Yes" else 0

        # Create dataframe for prediction
        input_data = pd.DataFrame([{
            'gender': gender,
            'age': age,
            'hypertension': hypertension_val,
            'heart_disease': heart_disease_val,
            'smoking_history': smoking_history,
            'bmi': bmi,
            'HbA1c_level': hba1c_level,
            'blood_glucose_level': blood_glucose_level
        }])
        
        # Apply label encoding
        input_data['gender'] = encoders['gender'].transform(input_data['gender'])
        input_data['smoking_history'] = encoders['smoking_history'].transform(input_data['smoking_history'])
        
        # Predict
        prediction = model.predict(input_data)[0]
        prob = model.predict_proba(input_data)[0]
        probability = prob[1] * 100 if prediction == 1 else prob[0] * 100
        
        return render_template('index.html', 
                               prediction=prediction, 
                               probability=round(probability, 1))

    except Exception as e:
        return render_template('index.html', error=f"An error occurred: {str(e)}")

if __name__ == '__main__':
    # Use the PORT environment variable if provided by a cloud host, otherwise default to 5000
    port = int(os.environ.get('PORT', 5000))
    # '0.0.0.0' allows external connections, and debug=False is required for production
    app.run(host='0.0.0.0', port=port, debug=False)
