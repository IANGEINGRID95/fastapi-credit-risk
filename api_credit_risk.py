from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import joblib

# Charger le modèle
import os

# Définir le chemin du modèle dans le conteneur Docker

# Charger le modèle
model = joblib.load("/app/data/loan_status_modele_V1.pkl")


# Vérifie que l'objet est valide
if not hasattr(model, "predict"):
    raise ValueError("L'objet chargé n'est pas un modèle scikit-learn valide.")
# Initialiser l'application FastAPI
app = FastAPI()
@app.get("/")
def home():
    return {"message": "Bienvenue sur l'API Loan Status Prediction !"}

# Définir la structure des données entrantes
class PredictionRequest(BaseModel):
    person_age: int
    person_income: float
    person_home_ownership: str
    person_emp_length: float
    loan_intent: str
    loan_grade: str
    loan_amnt: float
    loan_int_rate: float
    loan_percent_income: float
    cb_person_default_on_file: str
    cb_person_cred_hist_length: int


@app.post("/predict")
def predict(data: PredictionRequest):
    try:
        # Convertir les données en DataFrame
        input_data = pd.DataFrame([data.dict()])
        print("Données reçues :", input_data)

        # Étape 1 : Encodage des colonnes catégoriques
        input_data['encoded_home_ownership'] = input_data['person_home_ownership'].map(
            {'RENT': 1, 'MORTGAGE': 2, 'OWN': 3, 'OTHER': 4}).fillna(0).astype(int)
        input_data['encoded_loan_intent'] = input_data['loan_intent'].map(
            {'EDUCATION': 1, 'MEDICAL': 2, 'VENTURE': 3, 'PERSONAL': 4, 'DEBTCONSOLIDATION': 5, 'HOMEIMPROVEMENT': 6}).fillna(0).astype(int)
        input_data['encoded_loan_grade'] = input_data['loan_grade'].map(
            {'A': 1, 'B': 2, 'C': 3, 'D': 4, 'E': 5, 'F': 6, 'G': 7}).fillna(0).astype(int)
        input_data['encoded_cb_person_default_on_file'] = input_data['cb_person_default_on_file'].map(
            {'N': 1, 'Y': 2}).fillna(0).astype(int)

        # Étape 2 : Suppression des colonnes inutiles
        input_data = input_data.drop(columns=[
            'person_home_ownership', 'loan_intent', 'loan_grade', 'cb_person_default_on_file'], errors='ignore')

        # Étape 3 : Vérification des colonnes nécessaires
        all_relevant_columns = [
            'person_age', 'person_income', 'person_emp_length', 'loan_amnt', 'loan_int_rate',
            'loan_percent_income', 'cb_person_cred_hist_length', 'encoded_home_ownership',
            'encoded_loan_intent', 'encoded_loan_grade', 'encoded_cb_person_default_on_file'
        ]
        missing_columns = [col for col in all_relevant_columns if col not in input_data.columns]
        if missing_columns:
            raise ValueError(f"Colonnes manquantes pour la prédiction : {missing_columns}")

        # Sélectionner uniquement les colonnes pertinentes
        input_data = input_data[all_relevant_columns]
        print("Données après transformation :", input_data)

        # Étape 4 : Faire la prédiction
        prediction = model.predict(input_data)

        # Retourner la prédiction
        return {"prediction": prediction.tolist()}

    except Exception as e:
        # Capturer et afficher les erreurs
        print("Erreur lors de la prédiction :", e)
        import traceback
        traceback.print_exc()

        # Retourner une réponse d'erreur
        return {"error": str(e)}
