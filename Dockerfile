# Utiliser une image Python 3.9
FROM python:3.9

# Définir le répertoire de travail
WORKDIR /app

# Copier **tout** le projet dans /app
COPY . /app
COPY api_credit_risk.py ./api_credit_risk.py

# Installer les dépendances
RUN pip install --no-cache-dir -r requirements.txt

# Exposer le port 8000
EXPOSE 8000

# Lancer l'application avec Uvicorn
CMD ["uvicorn", "api_credit_risk:app", "--host", "0.0.0.0", "--port", "8000"]
