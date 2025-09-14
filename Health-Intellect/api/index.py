from flask import Flask, request, render_template
import joblib
import pandas as pd
import numpy as np

# Initialize the Flask application
app = Flask(__name__)

model = joblib.load('final_diabetes_model.pkl')
scaler = joblib.load('scaler.pkl')

# --- FEATURE NAME LISTS ---
# 1. Features collected directly from the HTML form
form_feature_names = [
    'HighBP', 'HighChol', 'CholCheck', 'BMI', 'Smoker', 'Stroke', 
    'HeartDiseaseorAttack', 'PhysActivity', 'Fruits', 'Veggies', 
    'HvyAlcoholConsump', 'AnyHealthcare', 'NoDocbcCost', 'GenHlth', 
    'MentHlth', 'PhysHlth', 'DiffWalk', 'Sex', 'Age', 'Education', 'Income'
]
# 2. Final features the trained model expects (including engineered ones)
final_model_features = [
    'HighBP', 'HighChol', 'CholCheck', 'BMI', 'Smoker', 'Stroke',
    'HeartDiseaseorAttack', 'PhysActivity', 'Fruits', 'Veggies',
    'HvyAlcoholConsump', 'AnyHealthcare', 'NoDocbcCost', 'GenHlth',
    'MentHlth', 'PhysHlth', 'DiffWalk', 'Sex', 'Age', 'Education',
    'Income', 'UnhealthyDays', 'HealthScore'
]

# --- FLASK ROUTES ---
@app.route('/')
def home():
    """Renders the main page."""
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """Handles the prediction logic."""
    try:
        # 1. Validate user input
        form_values = [request.form.get(name) for name in form_feature_names]
        if any(value is None or value == '' for value in form_values):
            return render_template('index.html', 
                                   prediction_text="Error: Please fill in all the fields.", 
                                   risk_level='error')

        input_features = [float(value) for value in form_values]
        
        # 2. Create DataFrame for the model
        features_df = pd.DataFrame([input_features], columns=form_feature_names)
        
        # 3. Perform Feature Engineering (must match the notebook)
        features_df['UnhealthyDays'] = features_df['MentHlth'] + features_df['PhysHlth']
        health_score_features = ['HighBP', 'HighChol', 'Smoker', 'Stroke', 'HeartDiseaseorAttack', 'DiffWalk']
        features_df['HealthScore'] = features_df[health_score_features].sum(axis=1)
        features_df = features_df[final_model_features] # Ensure correct column order
        
        # 4. Scale features and make a prediction
        features_scaled_np = scaler.transform(features_df)
        features_scaled_df = pd.DataFrame(features_scaled_np, columns=final_model_features)
        
        prediction_proba = model.predict_proba(features_scaled_df)
        diabetes_probability = prediction_proba[0][1]
        
        # 5. Prepare results for rendering
        result_text = f"The model predicts a {diabetes_probability*100:.2f}% probability of diabetes based on these indicators."
        risk_level = 'low-risk'
        if diabetes_probability > 0.5: risk_level = 'high-risk'
        elif diabetes_probability > 0.2: risk_level = 'medium-risk'
            
        return render_template('index.html', 
                               prediction_text=result_text, 
                               risk_level=risk_level,
                               probability=diabetes_probability)

    except Exception as e:
        print(f"An error occurred in the predict function: {e}")
        return render_template('index.html', prediction_text=f"An application error occurred: {e}", risk_level='error')

if __name__ == "__main__":
    app.run(debug=True)
