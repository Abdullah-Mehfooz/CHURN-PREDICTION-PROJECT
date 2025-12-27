# --------------------------------------------------------------------------------
# 1. IMPORT REQUIRED LIBRARIES
# --------------------------------------------------------------------------------
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import streamlit as st

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# --------------------------------------------------------------------------------
# 2. PAGE CONFIGURATION
# --------------------------------------------------------------------------------
st.set_page_config(
    page_title="CHURN PREDICTION",
    page_icon="📊",
    layout="wide",                    # Wide layout for better use of screen space
    initial_sidebar_state="expanded"  # Sidebar open by default
)

# --------------------------------------------------------------------------------
# 3. CUSTOM CSS STYLING
# --------------------------------------------------------------------------------
st.markdown("""
<style>
    /* Overall app background */
    .stApp { background-color: #FFFFFF; }

    /* Main headers */
    .main-header { font-size: 2.8rem; font-weight: 600; text-align: center; color: #004D40; margin-bottom: 0.5rem; font-family: 'Segoe UI', sans-serif; }
    .sub-header { font-size: 1.4rem; text-align: center; color: #455A64; margin-bottom: 3rem; font-weight: 400; }

    /* Buttons */
    .stButton>button { background-color: #00695C; color: white; border-radius: 8px; height: 3.2em; font-weight: 500; border: none; box-shadow: 0 2px 8px rgba(0,0,0,0.1); }
    .stButton>button:hover { background-color: #004D40; box-shadow: 0 4px 12px rgba(0,0,0,0.15); }

    /* Prediction result cards */
    .prediction-box { font-size: 1.8rem; text-align: center; padding: 2rem; border-radius: 12px; margin: 2.5rem 0; font-weight: 600; box-shadow: 0 6px 20px rgba(0,0,0,0.1); border-left: 6px solid; }
    .churn-yes { background-color: #FFF3E0; color: #E65100; border-left-color: #FF8A65; }
    .churn-no { background-color: #E0F2F1; color: #004D40; border-left-color: #26A69A; }

    /* Sidebar styling */
    section[data-testid="stSidebar"] { background-color: #F5F9F9; border-right: 1px solid #B2DFDB; }
    .sidebar-logo { display: flex; flex-direction: column; align-items: center; justify-content: center; text-align: center; padding: 1rem 0; }

    h1, h2, h3, h4, h5, h6, p, div, span, label { color: #004D40 !important; }
    .stMetric > label { color: #455A64 !important; font-size: 1.1rem !important; }
    .stMetric > div { color: #004D40 !important; font-size: 1.8rem !important; font-weight: 600 !important; }
    .stTabs [data-baseweb="tab"] { color: #004D40 !important; }
    label { color: #004D40 !important; font-weight: 600; }

  
    div[data-baseweb="select"] div[role="listbox"], div[data-baseweb="select"] ul { background-color: white !important; border: 1px solid #B2DFDB !important; }
    div[data-baseweb="select"] div[role="option"], div[data-baseweb="select"] li { color: #004D40 !important; background-color: white !important; font-weight: 500 !important; }
    div[data-baseweb="select"] div[role="option"]:hover, div[data-baseweb="select"] li:hover { background-color: #E0F2F1 !important; color: #004D40 !important; }
    div[data-baseweb="select"] > div > div > div { color: #004D40 !important; font-weight: 600; }

    
    div[data-testid="stToolbar"] > div > div:nth-child(2) > div > button { display: none !important; }
    div[data-testid="stToolbar"] > div > div:nth-child(3) > div > button { display: none !important; }
    div[data-testid="stToolbarActions"] > button:nth-child(2), div[data-testid="stToolbarActions"] > button:nth-child(3) { display: none !important; }
</style>
""", unsafe_allow_html=True)



# --------------------------------------------------------------------------------
# 4. INDUSTRY SELECTION (Sidebar)
# --------------------------------------------------------------------------------
industry = st.sidebar.selectbox(
    "🔀 Select Industry",
    options=["Banking", "Telecom"],
    index=0
)

# --------------------------------------------------------------------------------
# 5. DATASET URLS
# --------------------------------------------------------------------------------
BANKING_URL = "https://raw.githubusercontent.com/sharmaroshan/Churn-Modelling-Dataset/master/Churn_Modelling.csv"
TELECOM_URL = "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv"

# --------------------------------------------------------------------------------
# 6. MODEL LOADING FUNCTIONS (Cached for performance)
# Each function loads data, preprocesses, trains model, and returns trained objects
# --------------------------------------------------------------------------------
@st.cache_resource
def load_banking_model():
    # Load and select relevant columns
    df = pd.read_csv(BANKING_URL)
    df = df[["CreditScore", "Gender", "Age", "Tenure", "Balance",
             "NumOfProducts", "HasCrCard", "IsActiveMember",
             "EstimatedSalary", "Exited"]]
    
    # Encode Gender (Male/Female → 0/1)
    le_gender = LabelEncoder()
    df["Gender"] = le_gender.fit_transform(df["Gender"])
    
    # Split features and target
    X = df.drop("Exited", axis=1)
    y = df["Exited"]
    
    # Train-test split and model training
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)
    
    # Evaluate model
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)
    cm = confusion_matrix(y_test, y_pred)
    
    return model, le_gender, df, acc, report, cm

@st.cache_resource
def load_telecom_model():
    # Load and clean data
    df = pd.read_csv(TELECOM_URL)
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df.dropna(inplace=True)
    df.drop("customerID", axis=1, inplace=True)
    df["Churn"] = (df["Churn"] == "Yes").astype(int)
    
    # Split features and target
    X = df.drop("Churn", axis=1)
    y = df["Churn"]
    
    # Preprocess categorical columns with OneHotEncoder
    cat_cols = X.select_dtypes(include="object").columns
    preprocessor = ColumnTransformer([("cat", OneHotEncoder(drop="first"), cat_cols)], remainder="passthrough")
    model = Pipeline([("prep", preprocessor), ("clf", LogisticRegression(max_iter=1000))])
    
    # Train-test split and model training
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model.fit(X_train, y_train)
    
    # Evaluate model
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)
    cm = confusion_matrix(y_test, y_pred)
    
    return model, df, acc, report, cm

# --------------------------------------------------------------------------------
# 7. LOAD MODEL & DATA BASED ON SELECTED INDUSTRY
# Also sets target column name and sidebar icon
# --------------------------------------------------------------------------------
if industry == "Banking":
    model, gender_encoder, data, accuracy, report, cm = load_banking_model()
    target_col = "Exited"
    icon_url = "https://img.icons8.com/color/120/000000/bank-building.png"
else:
    model, data, accuracy, report, cm = load_telecom_model()
    target_col = "Churn"
    icon_url = "https://img.icons8.com/color/120/000000/phone-office.png"

# --------------------------------------------------------------------------------
# 8. SIDEBAR CONTENT
# Displays logo, title, key metrics
# --------------------------------------------------------------------------------
with st.sidebar:
    
    st.markdown(f"""
    <div class="sidebar-logo">
        <img src="{icon_url}" width="120">
        <h3 style="margin-top: 1rem; color:#004D40;">{industry} Churn Analytics</h3>
        <p style="color:#455A64; margin-top: -1rem;">Customer Retention Dashboard</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    # Model performance metrics
    st.metric("**Model Accuracy**", f"{accuracy:.2%}")
    st.metric("**Total Customers**", f"{len(data):,}")
    st.markdown("---")

# --------------------------------------------------------------------------------
# 9. MAIN HEADER
# Dynamic title and subtitle based on selected industry
# --------------------------------------------------------------------------------
st.markdown(f'<div class="main-header">{industry} Customer Churn Analytics Dashboard</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Data-driven insights to reduce customer attrition and improve retention</div>', unsafe_allow_html=True)

# --------------------------------------------------------------------------------
# 10. TABS NAVIGATION
# Three main sections: Analytics, Prediction, Model Performance
# --------------------------------------------------------------------------------
tab1, tab2, tab3 = st.tabs(["📊 Analytics Overview", "🔮 Predict Churn", "📈 Model Performance"])

# --------------------------------------------------------------------------------
# 11. TAB 1: ANALYTICS OVERVIEW
# Displays key metrics and visualizations
# --------------------------------------------------------------------------------
with tab1:
    st.header("Customer Portfolio Overview")
    churn_rate = data[target_col].mean()
    
    # 4-column key metrics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Churn Rate", f"{churn_rate:.2%}")
    if industry == "Banking":
        with col2: st.metric("Average Age", f"{data['Age'].mean():.0f} years")
        with col3: st.metric("Avg. Balance", f"${data['Balance'].mean():,.0f}")
        with col4: st.metric("Active Members", f"{data['IsActiveMember'].mean():.2%}")
    else:
        with col2: st.metric("Average Tenure", f"{data['tenure'].mean():.0f} months")
        with col3: st.metric("Avg. Monthly Charges", f"${data['MonthlyCharges'].mean():.1f}")
        with col4: st.metric("Internet Users", f"{(data['InternetService'] != 'No').mean():.2%}")

    st.markdown("---")
    
    # 2-column visualizations
    col_v1, col_v2 = st.columns(2)
    colors = ['#26A69A', '#FF8A65']
    
    with col_v1:
        st.subheader("Churn Distribution")
        fig_pie, ax_pie = plt.subplots(figsize=(7, 5))
        ax_pie.pie([1-churn_rate, churn_rate], labels=['Retained', 'Churned'], autopct='%1.1f%%',
                   startangle=90, colors=colors, textprops={'fontsize': 12, 'fontweight': '600'})
        ax_pie.axis('equal')
        st.pyplot(fig_pie)
        plt.close(fig_pie)
    
    with col_v2:
        st.subheader("Churn by Key Factor")
        plot_data = data.copy()
        fig_bar, ax_bar = plt.subplots(figsize=(7, 5))
        if industry == "Banking":
            plot_data["Gender"] = plot_data["Gender"].map({0: "Female", 1: "Male"})
            sns.countplot(data=plot_data, x="Gender", hue=target_col, palette=colors, ax=ax_bar)
            ax_bar.set_title("Churn Rate by Gender")
        else:
            sns.countplot(data=data, x="Contract", hue=target_col, palette=colors, ax=ax_bar)
            ax_bar.set_title("Churn Rate by Contract Type")
        ax_bar.legend(["Retained", "Churned"])
        st.pyplot(fig_bar)
        plt.close(fig_bar)

    # Age/Tenure distribution
    st.subheader("Age/Tenure Distribution by Churn Status")
    fig_age = sns.histplot(data=data, x="Age" if industry == "Banking" else "tenure",
                           hue=target_col, multiple="stack", palette=colors, bins=20, edgecolor="white")
    plt.title("Age/Tenure Profile: Retained vs Churned Customers", fontsize=14, pad=20)
    plt.xlabel("Age" if industry == "Banking" else "Tenure (Months)")
    plt.ylabel("Number of Customers")
    st.pyplot(fig_age.figure)
    plt.close()

# --------------------------------------------------------------------------------
# 12. TAB 2: INDIVIDUAL CHURN PREDICTION
# Form for entering customer details and getting real-time prediction
# --------------------------------------------------------------------------------
with tab2:
    st.header("Individual Customer Risk Assessment")
    st.write("Enter customer details to receive a real-time churn probability prediction.")
    
    with st.form("prediction_form"):
        col_left, col_right = st.columns(2)
        
        # Banking-specific inputs
        if industry == "Banking":
            with col_left:
                credit_score = st.slider("Credit Score", 300, 850, 650)
                age = st.slider("Age", 18, 100, 38)
                tenure = st.slider("Tenure (Years with Bank)", 0, 10, 5)
                balance = st.number_input("Account Balance ($)", 0.0, 300000.0, 0.0, step=1000.0)
                num_products = st.slider("Number of Products", 1, 4, 2)
            with col_right:
                gender = st.selectbox("Gender", ["Male", "Female"])
                has_card = st.selectbox("Has Credit Card?", ["Yes", "No"])
                is_active = st.selectbox("Active Member?", ["Yes", "No"])
                salary = st.number_input("Estimated Salary ($)", 0.0, 200000.0, 60000.0, step=1000.0)
        # Telecom-specific inputs
        else:
            with col_left:
                tenure = st.slider("Tenure (Months)", 0, 72, 24)
                monthly = st.number_input("Monthly Charges ($)", 18.0, 120.0, 70.0, step=1.0)
                total = st.number_input("Total Charges ($)", 18.0, 9000.0, 1000.0, step=50.0)
            with col_right:
                contract = st.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])
                internet = st.selectbox("Internet Service", ["No", "DSL", "Fiber optic"])
                payment = st.selectbox("Payment Method", ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"])

        submitted = st.form_submit_button("🔍 Assess Churn Risk", use_container_width=True)
    
    # Process prediction when form is submitted
    if submitted:
        if industry == "Banking":
            input_df = pd.DataFrame([{
                "CreditScore": credit_score, "Gender": gender, "Age": age, "Tenure": tenure,
                "Balance": balance, "NumOfProducts": num_products,
                "HasCrCard": 1 if has_card == "Yes" else 0,
                "IsActiveMember": 1 if is_active == "Yes" else 0,
                "EstimatedSalary": salary
            }])
            input_df["Gender"] = gender_encoder.transform(input_df["Gender"])
        else:
            input_df = pd.DataFrame([{
                "tenure": tenure, "MonthlyCharges": monthly, "TotalCharges": total,
                "Contract": contract, "InternetService": internet, "PaymentMethod": payment,
                "gender": "Male", "SeniorCitizen": 0, "Partner": "No", "Dependents": "No",
                "PhoneService": "Yes", "MultipleLines": "No", "OnlineSecurity": "No",
                "OnlineBackup": "No", "DeviceProtection": "No", "TechSupport": "No",
                "StreamingTV": "No", "StreamingMovies": "No", "PaperlessBilling": "Yes"
            }])

        prediction = model.predict(input_df)[0]
        probability = model.predict_proba(input_df)[0]
        
        if prediction == 1:
            st.markdown(f'''
            <div class="prediction-box churn-yes">
                High Churn Risk Detected<br><br>
                <span style="font-size:1.4rem;">Probability of Churn: <strong>{probability[1]:.2%}</strong></span>
            </div>
            ''', unsafe_allow_html=True)
            st.warning("Recommendation: Prioritize retention outreach (e.g., personalized offers, relationship manager contact).")
        else:
            st.markdown(f'''
            <div class="prediction-box churn-no">
                Low Churn Risk<br><br>
                <span style="font-size:1.4rem;">Probability of Retention: <strong>{probability[0]:.2%}</strong></span>
            </div>
            ''', unsafe_allow_html=True)
            st.success("Customer shows strong loyalty indicators. Continue standard engagement.")

# --------------------------------------------------------------------------------
# 13. TAB 3: MODEL PERFORMANCE
# Shows accuracy metrics and confusion matrix
# --------------------------------------------------------------------------------
with tab3:
    st.header("Model Performance & Reliability")
    col1, col2, col3 = st.columns(3)
    col1.metric("Accuracy", f"{accuracy:.2%}")
    col2.metric("Precision (Churn)", f"{report['1']['precision']:.2%}")
    col3.metric("Recall (Churn)", f"{report['1']['recall']:.2%}")
    
    st.markdown("---")
    col_cm, _ = st.columns([2, 1])
    with col_cm:
        st.subheader("Confusion Matrix")
        fig_cm, ax_cm = plt.subplots(figsize=(6, 5))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", linewidths=1,
                    annot_kws={"size": 14}, cbar=False, ax=ax_cm)
        ax_cm.set_xlabel("Predicted", fontsize=12)
        ax_cm.set_ylabel("Actual", fontsize=12)
        ax_cm.set_title("Model Prediction Accuracy", fontsize=14, pad=20)
        st.pyplot(fig_cm)
        plt.close(fig_cm)
    
    with st.expander("View Detailed Classification Report"):
        st.dataframe(pd.DataFrame(report).transpose().style.format("{:.2%}"))

