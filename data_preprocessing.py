import pandas as pd
import numpy as np
import os
import json
from sklearn.preprocessing import StandardScaler

def load_and_preprocess_data(file_path='patient_symptoms.csv'):
    """
    Load and preprocess the patient symptom data.
    
    Args:
        file_path: Path to the CSV file containing patient data
        
    Returns:
        Processed DataFrame ready for recommendation system
    """
    # Check if file exists
    if not os.path.exists(file_path):
        print(f"File {file_path} not found. Please make sure it's in the correct location.")
        return None
    
    try:
        # Load data
        print(f"Loading data from {file_path}...")
        df = pd.read_csv(file_path)
        
        # Display basic info
        print("\nData Overview:")
        print(f"- Number of records: {len(df)}")
        print(f"- Number of unique patients: {df['patient_id'].nunique()}")
        print(f"- Number of unique symptoms: {df['symptom'].nunique()}")
        
        # Check for missing values
        missing_values = df.isnull().sum()
        if missing_values.sum() > 0:
            print("\nWarning: Missing values detected in these columns:")
            print(missing_values[missing_values > 0])
            
            # Fill missing values appropriately
            if 'age' in df.columns and df['age'].isnull().sum() > 0:
                df['age'].fillna(df['age'].median(), inplace=True)
            if 'bmi' in df.columns and df['bmi'].isnull().sum() > 0:
                df['bmi'].fillna(df['bmi'].median(), inplace=True)
            if 'gender' in df.columns and df['gender'].isnull().sum() > 0:
                df['gender'].fillna(df['gender'].mode()[0], inplace=True)
            if 'symptom' in df.columns and df['symptom'].isnull().sum() > 0:
                # Drop rows with missing symptoms
                df.dropna(subset=['symptom'], inplace=True)
                
            print("Missing values handled.")
        
        # Normalize numerical features for better recommendations
        if 'age' in df.columns and 'bmi' in df.columns:
            # Create normalized versions of age and BMI
            scaler = StandardScaler()
            df[['age_normalized', 'bmi_normalized']] = scaler.fit_transform(df[['age', 'bmi']])
            
        # Create gender_numeric column for calculations
        if 'gender' in df.columns:
            df['gender_numeric'] = df['gender'].map({'Female': 0, 'Male': 1})
            
        # Save unique symptoms and patients
        unique_symptoms = sorted(df['symptom'].unique())
        unique_patients = sorted(df['patient_id'].unique())
        
        # Save metadata for later use
        metadata = {
            'unique_symptoms': unique_symptoms,
            'unique_patients': unique_patients,
            'age_mean': df['age'].mean(),
            'age_std': df['age'].std(),
            'bmi_mean': df['bmi'].mean(),
            'bmi_std': df['bmi'].std()
        }
        
        with open('symptom_metadata.json', 'w') as f:
            json.dump(metadata, f)
            
        print("\nData preprocessing complete!")
        print(f"- Data shape after preprocessing: {df.shape}")
        print(f"- Metadata saved to symptom_metadata.json")
        
        return df
    
    except Exception as e:
        print(f"Error loading or processing data: {str(e)}")
        return None

def generate_symptom_profiles(df):
    """
    Generate a profile of symptoms for each patient.
    
    Args:
        df: Processed DataFrame with patient symptoms
        
    Returns:
        Dictionary mapping patient IDs to their symptoms
    """
    patient_symptoms = {}
    
    for patient_id in df['patient_id'].unique():
        symptoms = df[df['patient_id'] == patient_id]['symptom'].tolist()
        patient_symptoms[patient_id] = symptoms
        
    return patient_symptoms

def analyze_symptom_cooccurrence(df):
    """
    Analyze which symptoms commonly occur together.
    
    Args:
        df: Processed DataFrame with patient symptoms
        
    Returns:
        DataFrame with symptom co-occurrence counts
    """
    # Get unique symptoms
    unique_symptoms = df['symptom'].unique()
    
    # Create a dictionary to store co-occurrence counts
    cooccurrence = {s1: {s2: 0 for s2 in unique_symptoms} for s1 in unique_symptoms}
    
    # Count co-occurrences
    for patient_id in df['patient_id'].unique():
        symptoms = df[df['patient_id'] == patient_id]['symptom'].tolist()
        for i, s1 in enumerate(symptoms):
            for s2 in symptoms[i+1:]:
                cooccurrence[s1][s2] += 1
                cooccurrence[s2][s1] += 1
    
    # Convert to DataFrame
    cooccurrence_df = pd.DataFrame(cooccurrence)
    
    return cooccurrence_df

if __name__ == "__main__":
    # Change this to the path of your actual CSV file
    file_path = 'patient_symptoms.csv'
    
    # Load and preprocess data
    processed_df = load_and_preprocess_data(file_path)
    
    if processed_df is not None:
        # Generate patient symptom profiles
        patient_profiles = generate_symptom_profiles(processed_df)
        print(f"\nGenerated symptom profiles for {len(patient_profiles)} patients")
        
        # Analyze symptom co-occurrence
        cooccurrence_df = analyze_symptom_cooccurrence(processed_df)
        print("\nSymptom co-occurrence analysis complete")
        
        # Save co-occurrence matrix for later use
        cooccurrence_df.to_csv('symptom_cooccurrence.csv')
        print("Co-occurrence matrix saved to symptom_cooccurrence.csv")
