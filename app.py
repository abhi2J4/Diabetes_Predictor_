import streamlit as st
import pandas as pd
import pickle
import shap
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# ------------------------ Streamlit Page Config ------------------------
st.set_page_config(
    page_title="Diabetes Prediction AI",
    page_icon="🩺",
    layout="centered"
)


# Custom CSS Styling for Dark Theme
st.markdown(""" 
    <style>
    .stApp {
        background-color: #6200ee;
        color: #e0e0e0;
        font-family: 'Segoe UI', sans-serif;
    }
    h1 {
        text-align: center;
        color: #bb86fc;
    }
    label, .stTextInput label, .stSelectbox label {
        color: #bb86fc;
        font-weight: bold;
    }
    .stAlert {
        background-color: #333333 !important;
        color: #e0e0e0 !important;
    }
    .stButton>button {
        background-color: red;
        color: white;
        border-radius: 10px;
        padding: 10px 20px;
        font-weight: bold;
    }
    .stButton>button:hover {
        background-color: #3700b3;
        color: white;
    }
    </style>
""", unsafe_allow_html=True)

st.markdown("<h1>🧠 AI Diabetes Predictor</h1>", unsafe_allow_html=True)
st.markdown("<hr style='border: 1px solid #444;'>", unsafe_allow_html=True)

# ------------------------ Load and Train Model ------------------------

@st.cache_resource
def load_data_and_model():
    # Load dataset
    df = pd.read_csv('https://raw.githubusercontent.com/plotly/datasets/master/diabetes.csv')
    
    # Features and labels
    X = df.drop('Outcome', axis=1)
    y = df['Outcome']
    
    # Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train Random Forest
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # Accuracy
    acc = accuracy_score(y_test, model.predict(X_test))
    report = classification_report(y_test, model.predict(X_test))
    matrix = confusion_matrix(y_test, model.predict(X_test))
    
    return model, X.columns, acc, report, matrix

model, feature_names, accuracy, report, matrix = load_data_and_model()

# Cache SHAP explainer to avoid re-creation
@st.cache_resource
def get_shap_explainer(_model):
    import shap
    explainer = shap.Explainer(_model)
    return explainer

explainer = get_shap_explainer(model)

# ------------------------ User Input ------------------------

st.subheader("Enter Patient Details:")

def user_input_features():
    input_data = {}
    for feature in feature_names:
        input_data[feature] = st.number_input(f"{feature}", min_value=0.0, step=0.1)
    return pd.DataFrame([input_data])

user_input = user_input_features()

# ------------------------ Prediction ------------------------

if st.button("Predict Diabetes"):
    prediction = model.predict(user_input)[0]
    prediction_proba = model.predict_proba(user_input)[0][prediction]
    
    if prediction == 1:
        st.error(f"⚠️ High risk of Diabetes with confidence {prediction_proba:.2f}")
    else:
        st.success(f"✅ No Diabetes detected with confidence {prediction_proba:.2f}")

    # ------------------------ SHAP Explainability ------------------------
    st.subheader("Feature Contribution (Explainability)")
    shap_values = explainer.shap_values(user_input)

    # Force Plot
    # st.set_option('deprecation.showPyplotGlobalUse', False)
    plt.title("Feature Impact Visualization (SHAP)")
    shap.summary_plot(shap_values, user_input, plot_type="bar", show=False)
    st.pyplot()

    # Model Evaluation Metrics
    st.subheader("Model Evaluation Metrics")
    st.write("Accuracy:", accuracy)
    st.write("Classification Report:\n", report)
    st.write("Confusion Matrix:\n", matrix)

# ------------------------ Footer ------------------------

st.markdown("<hr style='border: 1px solid #444;'>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: #e0e0e0;'>Built with ❤️ using Streamlit, Scikit-learn & SHAP</p>", unsafe_allow_html=True);  
