import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import streamlit.components.v1 as components
import xgboost as xgb

# --- Page Configuration ---
st.set_page_config(
    page_title="Diabetic Patient Readmission Predictor",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Helper function for SHAP plots ---
def st_shap(plot, height=None):
    """
    A wrapper to display SHAP plots in Streamlit.
    This function injects the SHAP javascript library into the HTML component.
    """
    shap_html = f"<head>{shap.getjs()}</head><body>{plot.html()}</body>"
    components.html(shap_html, height=height, scrolling=True)

# --- Caching and Loading Model Artifacts ---
@st.cache_resource
def load_artifacts():
    """
    Loads all necessary model artifacts from disk using caching for efficiency.
    """
    # Load the XGBoost model from the .joblib file
    model = joblib.load('xgb_model.joblib')
    
    # Load the other artifacts saved with joblib
    encoder = joblib.load('one_hot_encoder.joblib')
    numerical_features = joblib.load('numerical_features.joblib')
    categorical_features = joblib.load('categorical_features.joblib')
    explainer = joblib.load('shap_explainer.joblib')
    
    # Get the feature names directly from the trained model itself for a perfect match.
    model_feature_names = model.get_booster().feature_names
    return model, encoder, model_feature_names, numerical_features, categorical_features, explainer

try:
    model, encoder, feature_names, numerical_features, categorical_features, explainer = load_artifacts()
except (FileNotFoundError, Exception) as e:
    st.error(f"Error loading model artifacts: {e}. Please ensure all .joblib files are present and not corrupted.")
    st.stop()

# --- Application UI ---
st.title("🏥 Intelligent Readmission Predictor for Diabetic Patients")
st.markdown("""
This application uses a machine learning model to predict 30-day readmission risk and explains its reasoning. 
Input patient details in the sidebar to receive an instant, interpretable risk assessment.
""")

# --- Sidebar for User Input ---
st.sidebar.header("Patient Input Features")

def get_user_input():
    """
    Creates the input fields in the sidebar and returns a DataFrame with the user's input.
    """
    inputs = {}
    
    st.sidebar.subheader("Demographics")
    inputs['race'] = st.sidebar.selectbox('Race', ['Caucasian', 'AfricanAmerican', 'Hispanic', 'Asian', 'Other'])
    inputs['gender'] = st.sidebar.selectbox('Gender', ['Male', 'Female'])
    inputs['age'] = st.sidebar.selectbox('Age Group', ['[0-10)', '[10-20)', '[20-30)', '[30-40)', '[40-50)', '[50-60)', '[60-70)', '[70-80)', '[80-90)', '[90-100)'])

    st.sidebar.subheader("Hospitalization Details")
    inputs['admission_type'] = st.sidebar.selectbox('Admission Type', ['Emergency', 'Urgent', 'Elective', 'Newborn', 'Not Available', 'Trauma Center'])
    inputs['admission_src'] = st.sidebar.selectbox('Admission Source', ['Physician Referral', 'Emergency Room', 'Transfer from a hospital', 'Clinic Referral', 'Other'])
    inputs['time_in_hospital'] = st.sidebar.slider('Time in Hospital (Days)', 1, 14, 4)
    
    st.sidebar.subheader("Clinical Information")
    inputs['diag_1_category'] = st.sidebar.selectbox('Primary Diagnosis Category', ['Circulatory', 'Respiratory', 'Digestive', 'Diabetes', 'Injury', 'Musculoskeletal', 'Neoplasms', 'Other'])
    inputs['num_lab_procedures'] = st.sidebar.slider('Number of Lab Procedures', 1, 132, 45)
    inputs['num_procedures'] = st.sidebar.slider('Number of Other Procedures', 0, 6, 1)
    inputs['num_medications'] = st.sidebar.slider('Number of Medications', 1, 81, 16)
    inputs['number_diagnoses'] = st.sidebar.slider('Number of Diagnoses', 1, 16, 9)
    inputs['number_outpatient'] = st.sidebar.slider('Outpatient Visits (Past Year)', 0, 42, 0)
    inputs['number_inpatient'] = st.sidebar.slider('Inpatient Visits (Past Year)', 0, 21, 1)
    inputs['number_emergency'] = st.sidebar.slider('Emergency Visits (Past Year)', 0, 76, 1)

    st.sidebar.subheader("Medication Status")
    inputs['insulin'] = st.sidebar.select_slider('Insulin Status', options=['No', 'Up', 'Down', 'Steady'], value='Steady')
    inputs['change'] = st.sidebar.selectbox('Change of Meds?', ['No', 'Ch'])
    inputs['diabetesMed'] = st.sidebar.selectbox('Diabetes Medication Prescribed?', ['No', 'Yes'])
    
    # Add placeholders for all medication features the model was trained on
    placeholder_meds = [
        'metformin', 'repaglinide', 'nateglinide', 'chlorpropamide', 'glimepiride', 
        'glipizide', 'glyburide', 'pioglitazone', 'rosiglitazone', 'acetohexamide', 
        'tolbutamide', 'acarbose', 'miglitol', 'troglitazone', 'tolazamide', 
        'examide', 'citoglipton', 'glyburide-metformin', 'glipizide-metformin', 
        'glimepiride-pioglitazone', 'metformin-rosiglitazone', 'metformin-pioglitazone'
    ]
    for med in placeholder_meds:
        if med != 'insulin': 
            inputs[med] = 'No'
    
    inputs['med_changes'] = 1 if inputs['insulin'] in ['Up', 'Down'] else 0
    
    return pd.DataFrame([inputs])

def process_input(input_df, model_feature_names):
    """
    Processes the raw user input DataFrame to match the model's expected format.
    """
    input_num = input_df[numerical_features]
    input_cat = input_df[categorical_features]
    
    input_cat_encoded = pd.DataFrame(encoder.transform(input_cat), 
                                     columns=encoder.get_feature_names_out(categorical_features))
    
    input_processed = pd.concat([input_num.reset_index(drop=True), 
                               input_cat_encoded.reset_index(drop=True)], axis=1)

    # Reorder the columns to exactly match the order the model was trained on.
    final_input = input_processed.reindex(columns=model_feature_names, fill_value=0)
    
    return final_input

# Get user input from the sidebar BEFORE the button is checked.
input_df = get_user_input()

# --- Display Prediction and Interpretation ---
st.subheader('Prediction Analysis')
if st.sidebar.button('**Analyze Patient Risk**', use_container_width=True):
    # Process the input that was already captured from the user's selections.
    input_final = process_input(input_df, feature_names)
    
    prediction_proba = model.predict_proba(input_final)[:, 1][0]
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.metric(label="**Readmission Risk Score**", value=f"{prediction_proba:.1%}")
        if prediction_proba >= 0.5:
            st.error("🔴 **Status: High Risk**")
        elif prediction_proba >= 0.25:
            st.warning("🟠 **Status: Moderate Risk**")
        else:
            st.success("🟢 **Status: Low Risk**")

    with col2:
        st.markdown("**How this prediction was made:**")
        shap_values = explainer.shap_values(input_final)
        
        # Use the processed data for the 'features' argument to ensure dimension consistency
        force_plot = shap.force_plot(
            base_value=explainer.expected_value,
            shap_values=shap_values[0,:],
            features=input_final.iloc[0,:], 
            link="logit"
        )
        
        st_shap(force_plot, height=160)
        
        st.markdown(
            "The plot above shows the features contributing to the prediction. "
            "Features in **<span style='color:red'>red</span>** push the prediction higher (increased risk), "
            "while features in **<span style='color:blue'>blue</span>** push it lower.",
            unsafe_allow_html=True
        )

    with st.expander("Show Raw Patient Data"):
        # Convert all values to string to resolve the Arrow serialization warning.
        display_df = input_df.T.rename(columns={0: 'Value'}).astype(str)
        st.dataframe(display_df)
else:
    st.info("Please fill in the patient details and click 'Analyze Patient Risk' to generate a prediction.")

