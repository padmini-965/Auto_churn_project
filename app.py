import streamlit as st
import pickle
import pandas as pd
import numpy as np
import os

# --- 1. ASSET LOADING & DIAGNOSIS ---

model = None
scaler = None
feature_names = None
scaler_feature_names = None

try:
    with open('final_churn_model.pkl', 'rb') as f:
        model = pickle.load(f)
    with open('scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    with open('feature_names.pkl', 'rb') as f:
        feature_names = pickle.load(f)
    
    # CRITICAL: Extract the EXACT feature names the scaler was fitted on
    # This is the authoritative source for scaling
    try:
        scaler_feature_names = scaler.get_feature_names_out().tolist()
    except AttributeError:
        # Fallback if scaler doesn't have get_feature_names_out
        scaler_feature_names = feature_names
    
    st.sidebar.success("✅ Model assets loaded successfully.")
    
    # DEBUG INFO (Remove after confirming it works)
    with st.sidebar.expander("🔍 Debug Info"):
        st.write("**Scaler Features:**", scaler_feature_names)
        st.write("**Model Features:**", feature_names)

except FileNotFoundError as e:
    st.error(f"❌ DEPLOYMENT ERROR: One or more critical files are missing.")
    st.error(f"Please ensure {e.filename} is in the same folder as app.py.")
    st.markdown("---")
    st.markdown("**Current Working Directory (where Python is looking):**")
    st.code(os.getcwd(), language='text')
    st.stop()
except Exception as e:
    st.error(f"❌ A generic error occurred during asset loading: {e}")
    st.stop()


# --- 2. STREAMLIT UI SETUP ---

st.set_page_config(page_title="Auto Churn Prediction", layout="centered")
st.title("🚗 Predictive Auto Churn Risk Score")
st.markdown("Adjust the customer profile below to estimate their churn probability.")

# --- INPUT FIELDS ---

st.header("1. Customer Stability (Top Drivers)")

col1, col2, col3 = st.columns(3)

with col1:
    days_tenure = st.slider("Tenure (Days)", 
                            min_value=1, max_value=8000, value=730, step=30,
                            help="Newer customers (< 365 days) are highest risk.")
with col2:
    length_of_residence = st.slider("Years at Residence", 
                                    min_value=0, max_value=30, value=3, step=1,
                                    help="Recent movers are higher risk.")
with col3:
    age_in_years = st.slider("Age (Years)", 
                             min_value=18, max_value=85, value=45, step=1)

st.header("2. Financial & Policy Metrics")

col4, col5 = st.columns(2)

with col4:
    income = st.number_input("Annual Income", 
                             min_value=10000, max_value=500000, value=65000, step=5000)
    
    has_children = st.selectbox("Has Children", [1.0, 0.0], format_func=lambda x: 'Yes' if x == 1.0 else 'No')
    college_degree = st.selectbox("Has College Degree", [1.0, 0.0], format_func=lambda x: 'Yes' if x == 1.0 else 'No')
    home_owner = st.selectbox("Is Home Owner", [1.0, 0.0], format_func=lambda x: 'Yes' if x == 1.0 else 'No')

with col5:
    home_market_value = st.number_input("Home Market Value", 
                                        min_value=0, max_value=1500000, value=250000, step=10000,
                                        help="Enter 0 if the customer is a renter or value is unknown.")
    curr_ann_amt = st.number_input("Annual Premium ($)", 
                                   min_value=500, max_value=5000, value=1800, step=10)
    good_credit = st.selectbox("Has Good Credit", [1.0, 0.0], format_func=lambda x: 'Yes' if x == 1.0 else 'No')
    
    acct_suspended_flag = st.selectbox("Account Suspended?", [0.0, 1.0], index=0, format_func=lambda x: 'No' if x == 0.0 else 'Yes', help="Was the account suspended in the last 12 months?")
    prior_claim_flag = st.selectbox("Prior Claim History?", [0.0, 1.0], index=0, format_func=lambda x: 'No' if x == 0.0 else 'Yes', help="Has the customer ever filed a claim?")

col6, col7 = st.columns(2)
with col6:
    marital_status = st.radio("Marital Status", ['Single', 'Married'], horizontal=True)
with col7:
    num_of_claims = st.number_input("Number of Past Claims", min_value=0, max_value=10, value=0, step=1, help="Total number of claims filed.")


if st.button("Calculate Churn Risk", type="primary"):
    
    # --- 3. PRE-PROCESSING & FEATURE ENGINEERING ---
    
    # Step 1: Engineer derived features
    income_log = np.log1p(income)
    tenure_years = days_tenure / 365.0
    
    # Step 2: Data cleaning/capping
    income_cap_value = 200000
    income_capped = min(income, income_cap_value)
    income_outlier = 1.0 if income > income_cap_value else 0.0
    home_market_value_missing = 1.0 if home_market_value == 0 else 0.0
    has_geo = 1.0
    
    # Step 3: Create tenure buckets (ONE-HOT ENCODED - ONLY ONE should be 1.0)
    tb_1_3 = 1.0 if (days_tenure >= 365 and days_tenure < 365*3) else 0.0
    tb_3_5 = 1.0 if (days_tenure >= 365*3 and days_tenure < 365*5) else 0.0
    tb_5_plus = 1.0 if (days_tenure >= 365*5) else 0.0
    
    # Step 4: One-hot encode marital status (ONLY ONE should be 1.0)
    ms_married = 1.0 if marital_status == 'Married' else 0.0
    ms_single = 1.0 if marital_status == 'Single' else 0.0
    
    # Step 5: Create initial input dictionary with ALL features
    input_data = {
        'curr_ann_amt': curr_ann_amt, 
        'days_tenure': days_tenure, 
        'age_in_years': age_in_years,
        'length_of_residence': length_of_residence,
        'home_market_value': home_market_value,
        'income': income,
        'home_market_value_num': home_market_value,
        'income_capped': income_capped, 
        'income_log': income_log,
        'num_of_claims': num_of_claims,
        'age_from_dob': age_in_years, 
        'tenure_years': tenure_years,
        'home_owner': home_owner,
        'has_children': has_children,
        'college_degree': college_degree,
        'good_credit': good_credit,
        'acct_suspended_flag': acct_suspended_flag,
        'prior_claim_flag': prior_claim_flag,
        'income_outlier': income_outlier, 
        'home_market_value_missing': home_market_value_missing, 
        'has_geo': has_geo, 
        'age_mismatch_flag': 0.0,
        'vehicle_status_flag': 0.0,
        'marital_status_Married': ms_married,
        'marital_status_single': ms_single, 
        'tenure_bucket_1-3 Years': tb_1_3,
        'tenure_bucket_3-5 Years': tb_3_5,
        'tenure_bucket_5+ Years': tb_5_plus,
    }
    
    # Step 6: Convert to DataFrame
    input_df = pd.DataFrame([input_data])
    
    # --- 4. CRITICAL: ALIGN WITH SCALER FEATURES ---
    # The scaler was fitted on a SPECIFIC set of features
    # We MUST provide EXACTLY those features, no more, no less
    
    # Create a new DataFrame with ONLY the scaler features
    # Initialize with all zeros (for any missing columns)
    scaler_input_df = pd.DataFrame(0.0, index=[0], columns=scaler_feature_names)
    
    # Fill in the values we have from input_df
    for col in scaler_feature_names:
        if col in input_df.columns:
            scaler_input_df[col] = input_df[col].values[0]
        # else: column stays 0.0 (safe default for missing features)
    
    # Step 7: Apply scaler transformation
    try:
        scaler_input_df_scaled = scaler_input_df.copy()
        scaler_input_df_scaled = pd.DataFrame(
            scaler.transform(scaler_input_df),
            columns=scaler_feature_names
        )
        st.sidebar.info("✅ Scaler transformation successful!")
    except ValueError as e:
        st.error(f"❌ Scaler Error: {e}")
        st.error(f"Expected features: {scaler_feature_names}")
        st.error(f"Provided features: {scaler_input_df.columns.tolist()}")
        st.stop()
    
    # --- 5. CRITICAL: PREPARE FOR MODEL PREDICTION ---
    # Select only the features the model expects, in the correct order
    
    final_input_df = pd.DataFrame(0.0, index=[0], columns=feature_names)
    
    # Fill in scaled values we have
    for col in feature_names:
        if col in scaler_input_df_scaled.columns:
            final_input_df[col] = scaler_input_df_scaled[col].values[0]
        elif col in input_df.columns:
            final_input_df[col] = input_df[col].values[0]
        # else: column stays 0.0
    
    # Ensure column order matches model's expected feature_names exactly
    final_input_df = final_input_df[feature_names]
    
    # --- 6. PREDICTION ---
    try:
        probability = model.predict_proba(final_input_df)[0][1]
        churn_risk_score = probability * 100
        st.sidebar.info("✅ Prediction successful!")
    except Exception as e:
        st.error(f"❌ Prediction Error: {e}")
        st.stop()
    
    # --- 7. RESULTS DISPLAY ---
    
    st.subheader("--- Risk Assessment ---")
    st.metric(label="Predicted Churn Probability", value=f"{churn_risk_score:.1f}%")
    
    if churn_risk_score >= 30.0:
        st.error("🚨 HIGH RISK: Immediate proactive retention intervention is required for this profile.")
        st.markdown("Recommended Action: Targeted follow-up call offering value-add (not just discount).")
    elif churn_risk_score >= 15.0:
        st.warning("⚠️ MEDIUM RISK: Flag for routine monitoring and non-aggressive intervention.")
        st.markdown("Recommended Action: Send high-value content or policy review summary.")
    else:
        st.success("✅ LOW RISK: Customer appears stable and loyal.")
        st.markdown("Recommended Action: Continue standard customer service.")