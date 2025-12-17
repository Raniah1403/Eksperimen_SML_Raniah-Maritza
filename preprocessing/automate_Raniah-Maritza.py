import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# Fungsi untuk preprocessing
def preprocess_data(file_path, output_path):
    # load dataset
    df = pd.read_csv(file_path)

    # transform target
    df['loan_approved'] = df['loan_approved'].astype(int)
    
    # drop fitur name & city
    df = df.drop(columns=['name', 'city'])

    # memisahkan fitur dan label
    X = df.drop(columns='loan_approved')
    y = df['loan_approved']

    # normalisasi standard scaler
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # splitting data
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

    # cek hasil
    print(f"Train shape: {X_train.shape}, Test Shape: {X_test.shape}")

    # Simpan dataset yang telah di proses
    processed_data = pd.DataFrame(X_scaled, columns=X.columns )
    processed_data['loan_approved'] = y.values

    processed_data.to_csv(output_path, index=False)
    print(f"Processed data saved to {output_path}")

    return X_train, X_test, y_train, y_test

if __name__ == "__main__":
    input_file_path = 'loan_approval.csv'
    output_file_path = 'preprocessing/processed_data.csv'
    X_train, X_test, y_train, y_test = preprocess_data(input_file_path, output_file_path)