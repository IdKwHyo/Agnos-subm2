from flask import Flask, request, render_template, jsonify
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from collections import Counter
import os
import json
import warnings
warnings.filterwarnings('ignore')

app = Flask(__name__)

# Define global variables
df = None
symptom_matrix = None
unique_symptoms = None
patient_features = None

def load_csv_data():
    """
    Load data from CSV file with column mapping to handle different column names.
    """
    try:
        # First, load the CSV file to inspect columns
        file_path = '[CONFIDENTIAL] AI symptom picker data (Agnos candidate assignment) - ai_symptom_picker.csv'
        raw_df = pd.read_csv(file_path)
        
        print("Available columns in CSV:", raw_df.columns.tolist())
        
        # Map the expected column names to the actual column names in the CSV
        # Update these mappings based on the actual columns in your CSV
        column_mapping = {
            'patient_id': 'patient_id',  # Might be 'id', 'patient', etc.
            'symptom': 'symptom_name',   # Might be 'symptom_name', 'condition', etc.
            'age': 'age',                # Might be 'patient_age', etc.
            'gender': 'gender',          # Might be 'sex', etc.
            'bmi': 'bmi'                 # Might be 'body_mass_index', etc.
        }
        
        # Try to automatically find column mappings if possible
        for expected_col, mapped_col in list(column_mapping.items()):
            if mapped_col not in raw_df.columns:
                # Try to find a suitable replacement
                for actual_col in raw_df.columns:
                    if expected_col.lower() in actual_col.lower():
                        column_mapping[expected_col] = actual_col
                        print(f"Mapped '{expected_col}' to '{actual_col}'")
                        break
        
        # Create a new DataFrame with our expected column names
        result_df = pd.DataFrame()
        
        for expected_col, mapped_col in column_mapping.items():
            if mapped_col in raw_df.columns:
                result_df[expected_col] = raw_df[mapped_col]
            else:
                print(f"Warning: Could not find a column matching '{expected_col}' in the CSV.")
                if expected_col == 'patient_id':
                    print("Creating sequential patient IDs...")
                    result_df[expected_col] = range(1, len(raw_df) + 1)
                elif expected_col == 'bmi':
                    print("Setting default BMI values...")
                    result_df[expected_col] = 22.0  # Default value
                else:
                    # For other missing columns, try to derive from other data or set defaults
                    result_df[expected_col] = "Unknown"
        
        # Make sure we have the minimum required columns
        required_cols = ['patient_id', 'symptom']
        missing_cols = [col for col in required_cols if col not in result_df.columns]
        
        if missing_cols:
            print(f"Error: Missing required columns: {missing_cols}")
            print("Falling back to sample data generation...")
            return generate_sample_data()
            
        return result_df
        
    except Exception as e:
        print(f"Error loading CSV: {e}")
        print("Using sample data instead.")
        return generate_sample_data()

def generate_sample_data(n_patients=100, n_symptoms=20):
    """Generate sample data for testing purposes"""
    np.random.seed(42)
    
    all_symptoms = [f"symptom_{i}" for i in range(1, n_symptoms+1)]
    genders = ['Male', 'Female']
    
    data = []
    
    for patient_id in range(1, n_patients+1):
        # Generate patient profile
        age = np.random.randint(18, 80)
        gender = np.random.choice(genders)
        bmi = round(np.random.normal(25, 5), 1)
        
        # Generate random number of symptoms for this patient (1-5 symptoms)
        n_patient_symptoms = np.random.randint(1, 6)
        selected_symptoms = np.random.choice(all_symptoms, n_patient_symptoms, replace=False)
        
        for symptom in selected_symptoms:
            data.append({
                'patient_id': patient_id,
                'age': age,
                'gender': gender,
                'bmi': bmi,
                'symptom': symptom
            })
    
    return pd.DataFrame(data)

def load_data():
    """Load and preprocess data"""
    global df, symptom_matrix, unique_symptoms, patient_features
    
    # Load data from CSV or generate sample data
    df = load_csv_data()
    
    print("Data frame loaded with shape:", df.shape)
    print("Columns in dataframe:", df.columns.tolist())
    
    # Make sure required columns exist
    required_cols = ['patient_id', 'symptom', 'age', 'gender', 'bmi']
    for col in required_cols:
        if col not in df.columns:
            print(f"Warning: Column '{col}' not found. Adding default values.")
            if col == 'patient_id':
                df[col] = range(1, len(df) + 1)
            elif col == 'symptom':
                df[col] = 'unknown_symptom'
            elif col == 'age':
                df[col] = 30  # Default age
            elif col == 'gender':
                df[col] = 'Unknown'
            elif col == 'bmi':
                df[col] = 22.0  # Default BMI
    
    # Create patient-symptom matrix
    unique_symptoms = sorted(df['symptom'].unique())
    unique_patients = sorted(df['patient_id'].unique())
    
    print(f"Found {len(unique_symptoms)} unique symptoms and {len(unique_patients)} unique patients")
    
    # Create a matrix with patients as rows and symptoms as columns
    symptom_matrix = np.zeros((len(unique_patients), len(unique_symptoms)))
    patient_map = {patient: i for i, patient in enumerate(unique_patients)}
    symptom_map = {symptom: i for i, symptom in enumerate(unique_symptoms)}
    
    for _, row in df.iterrows():
        patient_idx = patient_map[row['patient_id']]
        symptom_idx = symptom_map[row['symptom']]
        symptom_matrix[patient_idx, symptom_idx] = 1
    
    # Extract patient features for content-based filtering
    patient_features = df.drop_duplicates('patient_id')[['patient_id', 'age', 'gender', 'bmi']]
    
    # Convert gender to numeric (0 for female, 1 for male)
    gender_map = {'Female': 0, 'Male': 1, 'Unknown': 0.5}
    patient_features['gender_numeric'] = patient_features['gender'].map(lambda x: gender_map.get(x, 0.5))
    
    print("Data loaded successfully!")

def recommend_symptoms(patient_id=None, age=None, gender=None, bmi=None, selected_symptoms=None, top_n=5):
    """
    Recommend symptoms based on:
    1. Similar patients (collaborative filtering)
    2. Patient demographics (content-based filtering)
    3. Co-occurrence patterns of symptoms
    """
    # If we have an existing patient
    if patient_id is not None and patient_id in df['patient_id'].values:
        patient_data = df[df['patient_id'] == patient_id]
        current_symptoms = patient_data['symptom'].tolist()
        
        # If selected_symptoms is provided, use that instead
        if selected_symptoms:
            current_symptoms = selected_symptoms
            
        age = patient_data['age'].iloc[0]
        gender = patient_data['gender'].iloc[0]
        bmi = patient_data['bmi'].iloc[0]
    else:
        # For new patients
        current_symptoms = selected_symptoms if selected_symptoms else []
        gender = gender or 'Unknown'
    
    # 1. Find similar patients based on demographics
    if age is not None and gender is not None and bmi is not None:
        similar_patients = find_similar_patients(age, gender, bmi)
    else:
        similar_patients = df['patient_id'].unique().tolist()
    
    # 2. Get symptoms from similar patients
    similar_patient_symptoms = df[df['patient_id'].isin(similar_patients)]['symptom'].tolist()
    
    # 3. Calculate co-occurrence (symptoms that often appear together)
    co_occurring_symptoms = []
    if current_symptoms:
        for symptom in current_symptoms:
            patients_with_symptom = df[df['symptom'] == symptom]['patient_id'].unique()
            symptoms_from_these_patients = df[df['patient_id'].isin(patients_with_symptom)]['symptom'].tolist()
            co_occurring_symptoms.extend(symptoms_from_these_patients)
    
    # 4. Combine and count all potential symptoms
    potential_symptoms = similar_patient_symptoms + co_occurring_symptoms
    symptom_counts = Counter(potential_symptoms)
    
    # 5. Remove symptoms the patient already has
    for symptom in current_symptoms:
        if symptom in symptom_counts:
            del symptom_counts[symptom]
    
    # 6. Get top N recommendations
    recommendations = [rec for rec, _ in symptom_counts.most_common(top_n)]
    
    return recommendations

def find_similar_patients(age, gender, bmi, top_n=20):
    """Find patients with similar demographics"""
    # Convert input gender to numeric
    gender_map = {'Female': 0, 'Male': 1, 'Unknown': 0.5}
    gender_numeric = gender_map.get(gender, 0.5)
    
    # Create a feature vector for the input patient
    input_features = np.array([age, gender_numeric, bmi]).reshape(1, -1)
    
    # Create feature matrix for all patients
    patient_feature_matrix = patient_features[['age', 'gender_numeric', 'bmi']].values
    
    # Calculate cosine similarity
    similarities = cosine_similarity(input_features, patient_feature_matrix)[0]
    
    # Get indices of most similar patients
    similar_indices = similarities.argsort()[-top_n:][::-1]
    
    # Return patient IDs
    return patient_features.iloc[similar_indices]['patient_id'].tolist()

@app.route('/')
def home():
    # Make sure data is loaded
    global unique_symptoms
    if unique_symptoms is None:
        load_data()
    return render_template('index.html', symptoms=unique_symptoms)

@app.route('/recommend', methods=['POST'])
def recommend():
    # Make sure data is loaded
    global df
    if df is None:
        load_data()
        
    data = request.get_json()
    
    # Extract patient information
    patient_id = data.get('patient_id')
    age = data.get('age')
    gender = data.get('gender')
    bmi = data.get('bmi')
    selected_symptoms = data.get('selected_symptoms', [])
    
    # Convert string values to appropriate types
    if age is not None:
        age = float(age)
    if bmi is not None:
        bmi = float(bmi)
    if patient_id is not None:
        try:
            patient_id = int(patient_id)
        except ValueError:
            patient_id = None
    
    # Get recommendations
    recommendations = recommend_symptoms(
        patient_id=patient_id,
        age=age,
        gender=gender,
        bmi=bmi,
        selected_symptoms=selected_symptoms
    )
    
    return jsonify({
        'recommended_symptoms': recommendations
    })

if __name__ == '__main__':
    # Make sure to load data at startup
    load_data()
    app.run(debug=True)
