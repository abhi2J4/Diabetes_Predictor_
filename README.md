🧠 AI Diabetes Predictor
An AI-powered web app to predict the likelihood of diabetes based on patient input data, using a trained Random Forest Classifier and SHAP Explainability for feature contribution analysis.

Built with Streamlit, Scikit-learn, Matplotlib, Seaborn, and SHAP.

📋 Features
🏥 Diabetes Prediction based on standard medical features.

🧠 AI Explainability using SHAP values to visualize feature impact.

📊 Model Performance Metrics (Accuracy, Classification Report, Confusion Matrix).

🎨 Modern Dark-Themed UI with custom CSS.

⚡ Fast Inference with model and SHAP explainer caching.

📂 Project Structure

├── diabetes_predictor_app.py  # Main Streamlit app
├── README.md                  # Project documentation
└── requirements.txt           # Python dependencies (create it manually)

🛠️ Installation & Setup

1. Clone the Repository
   git clone https://github.com/abhi2J4/Diabetes_Predictor_
   cd Diabetes_Predictor_

2. Create a Virtual Environment (optional but recommended)
   python -m venv venv
   source venv/bin/activate      # On Windows: venv\Scripts\activate

3. Install Dependencies
   Create a requirements.txt file with:
   streamlit
   pandas
   scikit-learn
   matplotlib
   seaborn
   shap

3. Then install:
   pip install -r requirements.txt

4. Run the App
   streamlit run diabetes_predictor_app.py


🧠 Technologies Used
Streamlit — UI/UX Framework

Scikit-learn — Machine Learning

Pandas — Data Handling

SHAP — Model Explainability

Matplotlib & Seaborn — Visualization



🌟 Future Improvements
Deploy the app publicly (Streamlit Sharing, HuggingFace Spaces).

Allow CSV upload for bulk predictions.

Train with more advanced models (XGBoost, LightGBM).

Add patient record history using a database.

🤝 Contributing
Pull requests are welcome! For major changes, please open an issue first to discuss what you would like to change.

📜 License
This project is licensed under the MIT License.

✨ Credits
Built with ❤️ by Abhishek Yadav.
