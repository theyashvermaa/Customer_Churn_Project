import pandas as pd
import numpy as np
import joblib
import json
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.pipeline import Pipeline

def train():
    # 1. Load data
    print("Loading dataset...")
    df = pd.read_csv('dataset.csv')
    
    # 2. Feature Engineering
    print("Performing feature engineering...")
    df['Total Charges'] = df['Total day charge'] + df['Total eve charge'] + df['Total night charge'] + df['Total intl charge']
    df['Total_Usage'] = df['Total day minutes'] + df['Total eve minutes'] + df['Total night minutes'] + df['Total intl minutes']
    df['Service_Stress'] = df['Customer service calls'] / (df['Account length'] + 1)
    
    # Calculate quantile bins for Revenue_Segment on the training/full data
    # Note: We return bins to save them for Streamlit application consistency
    revenue_segment, bins = pd.qcut(df['Total Charges'], q=3, labels=['Low', 'Medium', 'High'], retbins=True)
    df['Revenue_Segment'] = revenue_segment
    
    # 3. Label Encoding for binary features
    df['International plan'] = df['International plan'].map({'No': 0, 'Yes': 1})
    df['Voice mail plan'] = df['Voice mail plan'].map({'No': 0, 'Yes': 1})
    df['Churn'] = df['Churn'].map({False: 0, True: 1})
    
    # 4. One-hot encoding
    df = pd.get_dummies(df, columns=['State', 'Revenue_Segment'], drop_first=True, dtype=int)
    
    # 5. Drop columns
    high_corr = ['Number vmail messages', 'Total night charge', 'Total day charge', 'Total intl charge', 'Total eve charge']
    df.drop(columns=high_corr, inplace=True, errors='ignore')
    df.drop(columns=['Area code'], inplace=True, errors='ignore')
    
    # 6. Split features & target
    X = df.drop(['Churn'], axis=1)
    y = df['Churn']
    
    # Stratified split to preserve class distribution
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    
    # 7. Build and train pipeline
    print("Training pipeline...")
    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", DecisionTreeClassifier(criterion='entropy', max_depth=5, min_samples_leaf=4, random_state=42))
    ])
    pipeline.fit(X_train, y_train)
    
    # Evaluate model
    train_score = pipeline.score(X_train, y_train)
    test_score = pipeline.score(X_test, y_test)
    print(f"Model trained successfully!")
    print(f"Train Accuracy: {train_score*100:.2f}%")
    print(f"Test Accuracy: {test_score*100:.2f}%")
    
    # 8. Save artifacts
    print("Saving best_model.pkl...")
    joblib.dump(pipeline, "best_model.pkl")
    
    # Save metadata for Streamlit preprocessing
    # bins are saved as list of floats
    metadata = {
        "features": X_train.columns.tolist(),
        "revenue_bins": bins.tolist(),
        "revenue_labels": ['Low', 'Medium', 'High'],
        "states": sorted(df.columns[df.columns.str.startswith('State_')].str.replace('State_', '').tolist())
    }
    
    print("Saving model_metadata.json...")
    with open("model_metadata.json", "w") as f:
        json.dump(metadata, f, indent=4)
        
    print("All artifacts saved successfully!")

if __name__ == "__main__":
    train()
