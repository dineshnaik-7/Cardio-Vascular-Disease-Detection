import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# ---------------------------
# 🎯 Title and Info
# ---------------------------
st.set_page_config(page_title="Heart Disease Prediction", layout="wide")
st.title("🫀 Heart Disease Machine Learning Web App")
st.markdown("Analyze and predict heart disease using ML models – Logistic Regression, Random Forest, KNN, Decision Tree, and SVM.")

# ---------------------------
# 📂 File Upload Section
# ---------------------------
uploaded_file = st.file_uploader("📥 Upload your Heart Disease Dataset (CSV)", type=["csv"])

if uploaded_file:
    dataset = pd.read_csv(uploaded_file)
    dataset.columns = dataset.columns.str.strip()  # Clean spaces in headers

    st.subheader("📊 Dataset Preview")
    st.dataframe(dataset, width='stretch', height=400)

    # ---------------------------
    # 🧹 Data Cleaning
    # ---------------------------
    if "Heart Disease" in dataset.columns:
        dataset["Heart Disease"] = dataset["Heart Disease"].replace({"Presence": 1, "Absence": 0}).astype(int)
    else:
        st.error("❌ Column 'Heart Disease' not found. Please check your CSV header name.")
        st.stop()

    col1, col2 = st.columns(2)
    with col1:
        st.metric("Dataset Shape", f"{dataset.shape[0]} rows × {dataset.shape[1]} columns")
    with col2:
        st.metric("Missing Values", dataset.isnull().sum().sum())

    # ---------------------------
    # 🔥 Correlation Heatmap with Fixed Size
    # ---------------------------
    st.subheader("📈 Correlation Heatmap")
    
    # Create container with fixed height
    with st.container():
        fig, ax = plt.subplots(figsize=(12, 8))
        sns.heatmap(dataset.corr(), annot=True, cmap="coolwarm", ax=ax, fmt='.2f', 
                    linewidths=0.5, cbar_kws={'shrink': 0.8})
        plt.title("Feature Correlation Matrix", fontsize=14, pad=15)
        plt.tight_layout()
        
        # Display with constrained height
        st.pyplot(fig, width='stretch')
        plt.close()

    # ---------------------------
    # ⚙️ Model Training
    # ---------------------------
    X = dataset.drop("Heart Disease", axis=1)
    y = dataset["Heart Disease"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    st.subheader("🤖 Select and Train Model")
    
    # Model selection with better layout
    col1, col2 = st.columns([1, 3])
    with col1:
        model_choice = st.selectbox(
            "Choose a model:",
            ["Logistic Regression", "Random Forest", "KNN", "Decision Tree", "SVM"]
        )

    # Initialize model based on selection
    if model_choice == "Logistic Regression":
        model = LogisticRegression(max_iter=1000)
    elif model_choice == "Random Forest":
        model = RandomForestClassifier(random_state=42)
    elif model_choice == "KNN":
        model = KNeighborsClassifier(n_neighbors=5)
    elif model_choice == "Decision Tree":
        model = DecisionTreeClassifier(random_state=42)
    else:  # SVM
        model = SVC(random_state=42)

    # Train model
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    
    with col2:
        st.success(f"✅ {model_choice} trained successfully! Accuracy: **{acc:.2%}**")

    # ---------------------------
    # 🧮 Confusion Matrix & Report with Fixed Size
    # ---------------------------
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📊 Confusion Matrix")
        fig, ax = plt.subplots(figsize=(6, 5))
        cm = confusion_matrix(y_test, y_pred)
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
                    xticklabels=['No Disease', 'Disease'],
                    yticklabels=['No Disease', 'Disease'])
        plt.title(f"{model_choice} - Confusion Matrix", fontsize=12, pad=10)
        plt.ylabel('Actual', fontsize=10)
        plt.xlabel('Predicted', fontsize=10)
        plt.tight_layout()
        st.pyplot(fig, width='stretch')
        plt.close()

    with col2:
        st.subheader("📋 Classification Report")
        report = classification_report(y_test, y_pred, output_dict=True)
        report_df = pd.DataFrame(report).transpose()
        st.dataframe(report_df.style.format("{:.2f}"), width='stretch')

    # ---------------------------
    # 💬 Live Prediction Section
    # ---------------------------
    st.subheader("🩺 Try a Live Prediction")
    st.markdown("Enter patient details below to predict heart disease risk:")
    
    # Create input fields in a more organized layout
    input_data = {}
    cols = st.columns(3)
    
    for idx, col in enumerate(X.columns):
        with cols[idx % 3]:
            if dataset[col].dtype in [int, float]:
                input_data[col] = st.number_input(
                    f"{col}",
                    min_value=float(X[col].min()),
                    max_value=float(X[col].max()),
                    value=float(X[col].mean()),
                    key=f"input_{col}"
                )
            else:
                input_data[col] = st.text_input(f"{col}", key=f"input_{col}")

    if st.button("🔍 Predict", type="primary"):
        user_df = pd.DataFrame([input_data])
        pred = model.predict(user_df)[0]
        
        # Display prediction result
        if pred == 1:
            st.error("⚠️ The model predicts **Heart Disease Presence**.")
        else:
            st.success("✅ The model predicts **No Heart Disease**.")
            
            # ---------------------------
            # Additional Health Risk Factors Analysis
            # ---------------------------
            st.markdown("---")
            st.subheader("📋 Health Risk Factor Analysis")
            
            risk_factors = []
            
            # Define normal ranges for common health metrics
            health_thresholds = {
                'Age': (30, 65, 'Age is in higher risk range'),
                'BP': (90, 130, 'Blood Pressure is elevated'),
                'Cholesterol': (125, 200, 'Cholesterol level is high'),
                'Max HR': (100, 180, 'Maximum Heart Rate is abnormal'),
                'FBS over 120': (0, 0, 'Fasting Blood Sugar is over 120 mg/dL')
            }
            
            # Check each metric against thresholds
            for feature, (lower, upper, message) in health_thresholds.items():
                if feature in input_data:
                    value = input_data[feature]
                    
                    if feature == 'FBS over 120':
                        if value > 0:
                            risk_factors.append({
                                'Factor': feature,
                                'Value': value,
                                'Status': '⚠️ ' + message
                            })
                    elif feature == 'Max HR':
                        if value < lower or value > upper:
                            risk_factors.append({
                                'Factor': feature,
                                'Value': value,
                                'Status': '⚠️ ' + message,
                                'Normal Range': f'{lower}-{upper}'
                            })
                    elif value > upper:
                        risk_factors.append({
                            'Factor': feature,
                            'Value': value,
                            'Status': '⚠️ ' + message,
                            'Normal Range': f'{lower}-{upper}'
                        })
            
            # Display risk factors if any
            if risk_factors:
                st.warning("⚠️ **Other Health Risk Factors Detected:**")
                risk_df = pd.DataFrame(risk_factors)
                st.dataframe(risk_df, width='stretch', hide_index=True)
                
                st.info("💡 **Recommendation:** While the model predicts no heart disease, "
                       "the above factors indicate areas for health improvement. "
                       "Please consult with a healthcare professional for personalized advice.")
            else:
                st.success("✅ All checked health metrics are within normal ranges!")

else:
    st.info("👆 Upload your dataset to get started.")
    st.markdown("""
    ### Expected Dataset Format:
    Your CSV should contain the following columns:
    - Age, Sex, Chest pain type, BP, Cholesterol
    - FBS over 120, EKG results, Max HR
    - Exercise angina, ST depression, Slope of ST
    - Number of vessels fluro, Thallium
    - **Heart Disease** (target variable: 'Presence' or 'Absence')
    """)