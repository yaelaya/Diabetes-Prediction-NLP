import streamlit as st
import pandas as pd
import numpy as np
import joblib
import pickle
import os

# Set page config
st.set_page_config(
    page_title="Diabetes Prediction App",
    page_icon="🏥",
    layout="wide"
)

@st.cache_resource
def load_models():
    """Load the trained models and feature names"""
    try:
        base_path = r"C:\Users\Lenovo\Desktop\2ite3eme annee\NLP\NLP Project- Diabetes\notebook"
        
        # Load XGBoost model
        model_path = os.path.join(base_path, "best_diabetes_model_xgboost.pkl")
        model = joblib.load(model_path)
        
        # Load scaler
        scaler_path = os.path.join(base_path, "scaler.pkl")
        scaler = joblib.load(scaler_path)
        
        # Load feature names
        feature_names_path = os.path.join(base_path, "feature_names.pkl")
        with open(feature_names_path, 'rb') as f:
            feature_names = pickle.load(f)
        
        return model, scaler, feature_names, base_path
        
    except FileNotFoundError as e:
        st.error(f"Model file not found: {e}")
        st.stop()
    except Exception as e:
        st.error(f"Error loading models: {e}")
        st.stop()

# Load models
model, scaler, feature_names, base_path = load_models()

# Function to compute engineered features
def compute_engineered_features(age, bmi, hba1c_level, blood_glucose_level):
    """Compute engineered features used during training"""
    # BMI Category
    if bmi < 18.5:
        bmi_category = 'Underweight'
    elif 18.5 <= bmi < 25:
        bmi_category = 'Normal'
    elif 25 <= bmi < 30:
        bmi_category = 'Overweight'
    elif 30 <= bmi < 35:
        bmi_category = 'Obese I'
    elif 35 <= bmi < 40:
        bmi_category = 'Obese II'
    else:
        bmi_category = 'Obese III'
    
    # Age Group
    if age <= 30:
        age_group = 'Young'
    elif 30 < age <= 45:
        age_group = 'Adult'
    elif 45 < age <= 60:
        age_group = 'Middle-aged'
    else:
        age_group = 'Senior'
    
    # Risk Score
    risk_score = (age / 80 * 0.25 + 
                 (bmi - 15) / 30 * 0.15 + 
                 hba1c_level / 15 * 0.30 + 
                 blood_glucose_level / 300 * 0.20)
    
    return bmi_category, age_group, risk_score

# Function to prepare input data
def prepare_input(gender, age, hypertension, heart_disease, smoking_history, bmi, hba1c_level, blood_glucose_level):
    """Prepare input data in the same format as training"""
    # Compute engineered features
    bmi_category, age_group, risk_score = compute_engineered_features(age, bmi, hba1c_level, blood_glucose_level)
    
    # Create a dictionary for all possible features
    input_dict = {
        'age': age,
        'hypertension': hypertension,
        'heart_disease': heart_disease,
        'bmi': bmi,
        'HbA1c_level': hba1c_level,
        'blood_glucose_level': blood_glucose_level,
        'risk_score': risk_score,
        
        # Gender encoding
        'gender_Male': 1 if gender == 'Male' else 0,
        'gender_Other': 1 if gender == 'Other' else 0,
        
        # Smoking history encoding
        'smoking_history_Never': 1 if smoking_history == 'Never' else 0,
        'smoking_history_No Information': 1 if smoking_history == 'No Information' else 0,
        'smoking_history_Not Current': 1 if smoking_history == 'Not Current' else 0,
        'smoking_history_Current': 1 if smoking_history == 'Current' else 0,
        'smoking_history_Former': 1 if smoking_history == 'Former' else 0,
        'smoking_history_Ever': 1 if smoking_history == 'Ever' else 0,
        
        # BMI category encoding
        'bmi_category_Normal': 1 if bmi_category == 'Normal' else 0,
        'bmi_category_Overweight': 1 if bmi_category == 'Overweight' else 0,
        'bmi_category_Obese I': 1 if bmi_category == 'Obese I' else 0,
        'bmi_category_Obese II': 1 if bmi_category == 'Obese II' else 0,
        'bmi_category_Obese III': 1 if bmi_category == 'Obese III' else 0,
        
        # Age group encoding
        'age_group_Adult': 1 if age_group == 'Adult' else 0,
        'age_group_Middle-aged': 1 if age_group == 'Middle-aged' else 0,
        'age_group_Senior': 1 if age_group == 'Senior' else 0
    }
    
    # Create DataFrame ensuring all training features are present
    input_df = pd.DataFrame([input_dict])
    
    # Ensure all feature names from training are present (fill missing with 0)
    for feature in feature_names:
        if feature not in input_df.columns:
            input_df[feature] = 0
    
    # Reorder columns to match training data
    input_df = input_df[feature_names]
    
    return input_df

# Streamlit app UI
st.title("🏥 Diabetes Prediction App")
st.markdown("""
This app predicts the likelihood of diabetes based on medical parameters using our trained **XGBoost** model.
""")

# Sidebar for inputs
st.sidebar.header("Patient Information")

# Input fields
col1, col2 = st.columns(2)

with col1:
    st.subheader("Demographic Information")
    gender = st.selectbox("Gender", ["Female", "Male", "Other"])
    age = st.slider("Age", min_value=0, max_value=100, value=40)
    smoking_history = st.selectbox("Smoking History", 
                                 ["Never", "No Information", "Not Current", "Current", "Former", "Ever"])

with col2:
    st.subheader("Medical Parameters")
    hypertension = st.radio("Hypertension", [0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
    heart_disease = st.radio("Heart Disease", [0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
    bmi = st.slider("BMI", min_value=10.0, max_value=50.0, value=25.0, step=0.1)
    hba1c_level = st.slider("HbA1c Level", min_value=3.0, max_value=15.0, value=5.0, step=0.1)
    blood_glucose_level = st.slider("Blood Glucose Level", min_value=50, max_value=300, value=100)

# Risk indicators
st.sidebar.subheader("Risk Indicators")
st.sidebar.markdown(f"""
- **Age**: {'🔴 High risk' if age > 60 else '🟡 Medium risk' if age > 45 else '🟢 Low risk'}
- **BMI**: {'🔴 High risk' if bmi >= 30 else '🟡 Medium risk' if bmi >= 25 else '🟢 Low risk'}
- **HbA1c**: {'🔴 High risk' if hba1c_level >= 6.5 else '🟡 Medium risk' if hba1c_level >= 5.7 else '🟢 Low risk'}
- **Glucose**: {'🔴 High risk' if blood_glucose_level >= 140 else '🟡 Medium risk' if blood_glucose_level >= 100 else '🟢 Low risk'}
""")

# Prediction button
if st.button("🔍 Predict Diabetes Risk", type="primary"):
    with st.spinner("Analyzing patient data..."):
        try:
            # Prepare input data
            input_data = prepare_input(gender, age, hypertension, heart_disease, 
                                     smoking_history, bmi, hba1c_level, blood_glucose_level)
            
            # Make prediction
            prediction = model.predict(input_data)[0]
            probability = model.predict_proba(input_data)[0][1]
            
            # Convert probability to Python float (fix for float32 issue)
            probability_float = float(probability)
            
            # Display results
            st.markdown("---")
            st.subheader("Prediction Results")
            
            # Create columns for results
            result_col1, result_col2 = st.columns(2)
            
            with result_col1:
                if prediction == 1:
                    st.error("**🔴 DIABETES DETECTED**")
                    st.write(f"Probability: **{probability_float*100:.2f}%**")
                else:
                    st.success("**🟢 NO DIABETES DETECTED**")
                    st.write(f"Probability of no diabetes: **{(1-probability_float)*100:.2f}%**")
            
            with result_col2:
                # Probability gauge with fixed float conversion
                st.metric(
                    label="Diabetes Probability", 
                    value=f"{probability_float*100:.2f}%",
                    delta="High risk" if probability_float > 0.5 else "Low risk"
                )
            
            # Progress bar with fixed float conversion
            st.write("Confidence Level:")
            st.progress(probability_float)
            st.caption(f"Confidence: {probability_float*100:.2f}%")
            
            # Risk factors analysis
            st.subheader("📊 Risk Factors Analysis")
            risk_factors = []
            
            if hba1c_level >= 6.5:
                risk_factors.append(f"🔴 HbA1c level ({hba1c_level}) indicates diabetes range (≥6.5%)")
            elif hba1c_level >= 5.7:
                risk_factors.append(f"🟡 HbA1c level ({hba1c_level}) indicates pre-diabetes range (5.7-6.4%)")
            else:
                risk_factors.append(f"🟢 HbA1c level ({hba1c_level}) is in normal range (<5.7%)")
                
            if blood_glucose_level >= 140:
                risk_factors.append(f"🔴 Blood glucose ({blood_glucose_level}) indicates diabetes range (≥140 mg/dL)")
            elif blood_glucose_level >= 100:
                risk_factors.append(f"🟡 Blood glucose ({blood_glucose_level}) indicates pre-diabetes range (100-139 mg/dL)")
            else:
                risk_factors.append(f"🟢 Blood glucose ({blood_glucose_level}) is in normal range (<100 mg/dL)")
                
            if bmi >= 30:
                risk_factors.append(f"🔴 BMI ({bmi}) indicates obesity (≥30)")
            elif bmi >= 25:
                risk_factors.append(f"🟡 BMI ({bmi}) indicates overweight (25-29.9)")
            else:
                risk_factors.append(f"🟢 BMI ({bmi}) is in normal range (18.5-24.9)")
                
            if age >= 45:
                risk_factors.append(f"🔴 Age ({age}) increases diabetes risk (≥45 years)")
            else:
                risk_factors.append(f"🟢 Age ({age}) is lower risk for diabetes")
            
            if hypertension == 1:
                risk_factors.append(f"🔴 Hypertension present (increases diabetes risk)")
            else:
                risk_factors.append(f"🟢 No hypertension (lower risk)")
                
            if heart_disease == 1:
                risk_factors.append(f"🔴 Heart disease present (increases diabetes risk)")
            else:
                risk_factors.append(f"🟢 No heart disease (lower risk)")
                
            for factor in risk_factors:
                st.write(factor)
            
            # Recommendations based on prediction
            st.subheader("💡 Recommendations")
            if prediction == 1:
                st.warning("""
                - **Consult a healthcare professional** for proper diagnosis and treatment
                - **Monitor blood sugar levels** regularly
                - **Follow a balanced diet** and maintain healthy weight
                - **Engage in regular physical activity**
                - **Attend regular medical check-ups**
                """)
            else:
                st.info("""
                - **Maintain healthy lifestyle** to prevent diabetes
                - **Regular exercise** and balanced diet
                - **Annual health check-ups** recommended
                - **Monitor risk factors** like weight and blood sugar
                """)
                
            # Disclaimer
            st.markdown("---")
            st.warning("""
            **⚠️ Disclaimer:** This prediction is based on machine learning models and should not replace 
            professional medical diagnosis. Always consult with healthcare professionals for medical advice.
            """)
            
        except Exception as e:
            st.error(f"Error making prediction: {str(e)}")

# Model information
with st.expander("ℹ️ About the Model"):
    st.markdown("""
    **Model Details:**
    - **Algorithm**: XGBoost (Extreme Gradient Boosting)
    - **Training Data**: 96,146 medical records
    - **Performance**: High accuracy in diabetes prediction
    
    **Top Predictive Features:**
    1. Blood Glucose Level
    2. HbA1c Level  
    3. Age
    4. BMI
    5. Risk Score
    
    **Clinical Ranges:**
    - **HbA1c**: Normal <5.7%, Pre-diabetes 5.7-6.4%, Diabetes ≥6.5%
    - **Blood Glucose**: Normal <100 mg/dL, Pre-diabetes 100-125 mg/dL, Diabetes ≥126 mg/dL
    - **BMI**: Underweight <18.5, Normal 18.5-24.9, Overweight 25-29.9, Obese ≥30
    """)
