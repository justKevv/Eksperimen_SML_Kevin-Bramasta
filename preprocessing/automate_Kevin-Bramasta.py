import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
import os
import subprocess

def load_data():
    zip_path = "telco-customer-churn.zip"
    extract_path = ""
    csv_path = os.path.join(extract_path, "WA_Fn-UseC_-Telco-Customer-Churn.csv")
    
    download_cmd = [
        "curl", "-L", "-o", zip_path,
        "https://www.kaggle.com/api/v1/datasets/download/blastchar/telco-customer-churn"
    ]
    subprocess.run(download_cmd, check=True)
    
    os.makedirs(extract_path, exist_ok=True)
    
    unzip_cmd = ["unzip", "-o", zip_path, "-d", extract_path]
    subprocess.run(unzip_cmd, check=True)
    
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
        print(f"Data loaded successfully. Shape: {df.shape}")
        return df
    else:
        raise FileNotFoundError(f"CSV file not found at {csv_path}. Check download/unzip steps.")

def preprocess_data(df):
    if 'customerID' in df.columns:
        df = df.drop('customerID', axis=1)
        
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    df['TotalCharges'].fillna(0, inplace=True)
    
    bins = [0, 12, 48, 72]
    labels = ['New', 'Existing', 'Loyal']
    df['TenureGroup'] = pd.cut(df['tenure'], bins=bins, labels=labels, include_lowest=True)
    
    le = LabelEncoder()
    df['Churn'] = le.fit_transform(df['Churn'])
    
    num_cols = ['tenure', 'MonthlyCharges', 'TotalCharges']
    
    cat_cols = [col for col in df.columns if df[col].dtype == 'object' or col == 'TenureGroup']

    if 'Churn' in cat_cols: 
        cat_cols.remove('Churn')

    scaler = StandardScaler()
    df[num_cols] = scaler.fit_transform(df[num_cols])
    print(f"Scaled numerical columns: {num_cols}")

    df_clean = pd.get_dummies(df, columns=cat_cols, drop_first=True)
    print(f"Applied One-Hot Encoding. Final Shape: {df_clean.shape}")

    return df_clean
    
if __name__ == "__main__":
    OUTPUT_DIR = 'preprocessing/telco_preprocessed'
    OUTPUT_FILE = 'telco_churn_clean.csv'
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    try:
        df_raw = load_data()
        
        df_processed = preprocess_data(df_raw)
        
        save_path = os.path.join(OUTPUT_DIR, OUTPUT_FILE)
        df_processed.to_csv(save_path, index=False)
    
    except Exception as e:
        print(e)