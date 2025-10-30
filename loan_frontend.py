import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.neighbors import KNeighborsClassifier

st.set_page_config(page_title="Loan Default Predictor", page_icon="💰", layout="centered")
st.title("💰 Loan Default Prediction System")

df = pd.read_csv('loan_default.csv')

X = df.drop('Default', axis=1)
y = df['Default']

numeric_features = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
categorical_features = X.select_dtypes(include=['object']).columns.tolist()

x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=10)

preprocessor = ColumnTransformer([
    ('num', StandardScaler(), numeric_features),
    ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
])

pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', KNeighborsClassifier())
])
pipeline.fit(x_train, y_train)

st.header("📋 Applicant Details")
user_data = {}

for col in numeric_features:
    user_data[col] = st.number_input(f"{col}", min_value=0.0)

for col in categorical_features:
    unique_values = list(df[col].dropna().unique())
    user_data[col] = st.selectbox(f"{col}", unique_values + ["Other"])

input_data = pd.DataFrame([user_data])

if st.button("Predict"):
    result = pipeline.predict(input_data)[0]
    if result == 1:
        st.error("❌ Applicant is likely to DEFAULT")
    else:
        st.success("✅ Applicant is NOT likely to default")

st.markdown("---")
st.caption("✅ Automatically detects columns and safely ignores unknown categories.")

