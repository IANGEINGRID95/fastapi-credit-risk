from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import pandas as pd

Credit_fichier="credit_risk_dataset.csv"

df=pd.read_csv(Credit_fichier)

df=df.drop_duplicates()
print(f"Nombre de lignes après suppression : {len(df)}")



df=df.copy()
df['encoded_home_ownership']=df['person_home_ownership'].map({'RENT':1,'MORTGAGE':2,'OWN':3,'OTHER':4})
df['encoded_loan_intent']=df['loan_intent'].map({'EDUCATION':1,'MEDICAL':2,'VENTURE':3,'PERSONAL':4,'DEBTCONSOLIDATION':5,'HOMEIMPROVEMENT':6})
df['encoded_loan_grade']=df['loan_grade'].map({'A':1,'B':2,'C':3,'D':4,'E':5,'F':6,'G':7})
df['encoded_cb_person_default_on_file']=df['cb_person_default_on_file'].map({'N':1,'Y':2})
df['encoded_home_ownership']=df['encoded_home_ownership'].astype(int)
df['encoded_loan_intent']=df['encoded_loan_intent'].astype(int)
df['encoded_loan_grade']=df['encoded_loan_grade'].astype(int)
df['encoded_cb_person_default_on_file']=df['encoded_cb_person_default_on_file'].astype(int)
print(df[['encoded_loan_intent','loan_intent']].head())
print(df[['encoded_loan_grade','loan_grade']].head())
print(df[['encoded_cb_person_default_on_file','cb_person_default_on_file']].head())
print(df[['encoded_home_ownership','person_home_ownership']].head())



# Suppression des colonnes originales
df = df.drop(columns=['person_home_ownership', 'loan_intent', 'loan_grade', 'cb_person_default_on_file'], errors='ignore')

encoded_columns = ['encoded_home_ownership', 'encoded_loan_intent', 'encoded_loan_grade', 'encoded_cb_person_default_on_file']
df_encoded = df[encoded_columns]
corr_matrix = df_encoded.corr()
print("Matrice de corrélation des variables encodées:")
print(corr_matrix)



numeric_columns = df.select_dtypes(include=['int64', 'float64']).columns
encoded_columns = ['encoded_home_ownership', 'encoded_loan_intent', 'encoded_loan_grade', 'encoded_cb_person_default_on_file']
all_relevant_columns = list(numeric_columns) + encoded_columns
df_relevant = df[all_relevant_columns]
corr_matrix = df_relevant.corr()
print("Matrice de corrélation des variables numériques et encodées :")
print(corr_matrix)



from sklearn.preprocessing import StandardScaler
numeric_columns = df.select_dtypes(include=['int64', 'float64']).columns
encoded_columns = ['encoded_home_ownership', 'encoded_loan_intent', 'encoded_loan_grade', 'encoded_cb_person_default_on_file']
all_relevant_columns = list(numeric_columns) + encoded_columns
scaler=StandardScaler()
df[all_relevant_columns]=scaler.fit_transform(df[all_relevant_columns])
print(df.head())


from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score
import pandas as pd

# Convertir loan_status en binaire (0 ou 1) en fonction du seuil
df['loan_status'] = df['loan_status'].apply(lambda x: 1 if x >= 0 else 0)

# Sélectionner les colonnes pertinentes
X = df[all_relevant_columns].drop(columns=['loan_status'])
y = df['loan_status']

# Vérification des NaN dans X et y
X = X.fillna(0)
y = y.fillna(0)

# Diviser les données en jeu d'entraînement et de test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Démarrer une expérimentation dans MLflow
with mlflow.start_run():
    # Entraîner le modèle
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Prédictions et évaluation
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    # Enregistrer le modèle dans MLflow
    mlflow.sklearn.log_model(model, "random_forest_model")

    # Enregistrer les métriques (par exemple la précision)
    mlflow.log_metric("accuracy", accuracy)

    # Enregistrer les hyperparamètres du modèle
    mlflow.log_param("n_estimators", 100)

    print(f"Accuracy: {accuracy}")