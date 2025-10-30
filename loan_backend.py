import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import pickle

df = pd.read_csv('Loan_default.csv')

print("✅ Dataset Loaded Successfully!")
print(df.head())

X = df.drop('Default', axis=1)
y = df['Default']

x_train, x_test, y_train, y_test = train_test_split(
    X, y, test_size=0.1, random_state=10
)

print(f"Training data shape: {x_train.shape}")
print(f"Testing data shape: {x_test.shape}")

preprocessor = ColumnTransformer([
    ('onehot', OneHotEncoder(), [4,5,6,7,8]), 
    ('scaler', StandardScaler(), [0,1,2,3])    
])

models = {
    "RandomForest": RandomForestClassifier(),
    "KNN": KNeighborsClassifier(),
    "LogisticRegression": LogisticRegression(max_iter=1000),
    "DecisionTree": DecisionTreeClassifier()
}

results = {}
for name, model in models.items():
    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', model)
    ])
    pipeline.fit(x_train, y_train)
    y_pred = pipeline.predict(x_test)
    accuracy = accuracy_score(y_test, y_pred)
    results[name] = accuracy
    print(f"{name} Accuracy: {accuracy:.4f}")

best_model_name = max(results, key=results.get)
print(f"\n🏆 Best Model: {best_model_name} with Accuracy {results[best_model_name]:.4f}")

best_pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', models[best_model_name])
])
best_pipeline.fit(x_train, y_train)

pickle.dump(best_pipeline, open('Loan_Default_Model.sav', 'wb'))
print("💾 Model saved as 'Loan_Default_Model.sav'")
