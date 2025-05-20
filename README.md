# Agnos-subm2
# 🤖 Agnos AI Symptom Recommender

A data-driven symptom recommendation system built for Agnos Health's AI candidate assignment. This application uses a hybrid recommendation approach combining collaborative filtering, content-based filtering, and symptom co-occurrence analysis to suggest likely symptoms based on patient demographics and inputs.

## 🚀 Features

- ✅ Intelligent symptom recommendation based on:
  - Similar patients (collaborative filtering)
  - Demographic similarity (content-based filtering)
  - Symptom co-occurrence patterns
- ✅ Flask-powered API with endpoints for integration
- ✅ Smart fallback to sample data when real CSV is unavailable
- ✅ Auto column mapping for flexible CSV input formats
- ✅ Modular, testable codebase

---

## 📁 Project Structure

agnos-symptom-recommender/
│
├── agnos2.py # Main Flask application
├── patient_symptoms.csv # Input data (if provided)
├── templates/
│ └── index.html # Simple frontend UI (optional)
├── static/ # (Optional) CSS/JS for frontend
└── requirements.txt # Python dependencies

---

## 🧠 Solution Overview

This recommender system uses a hybrid logic:

1. **Collaborative Filtering**  
   Recommends symptoms that co-occur among patients with similar symptom history.

2. **Content-Based Filtering**  
   Uses cosine similarity over age, gender, and BMI to identify similar patient profiles.

3. **Co-occurrence Analysis**  
   Extracts commonly appearing symptoms alongside currently selected symptoms.

The combination ensures robust, personalized recommendations.

---

## 🧪 API Endpoints

### `GET /`

Simple index page to render available symptoms via UI (optional).

### `POST /recommend`

Returns top symptom recommendations.

#### Payload:
```json
{
  "patient_id": 123,             // Optional
  "age": 35,
  "gender": "Male",
  "bmi": 22.5,
  "selected_symptoms": ["headache", "nausea"]
}
git clone https://github.com/IdKwHyo/Agnos-submission.git
cd Agnos-submission
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python agnos2.py
http://localhost:5000/
