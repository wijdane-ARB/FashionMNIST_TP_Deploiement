# TP Déploiement Fashion-MNIST — FastAPI

## Préparation

1. Le fichier **fashion_mnist.keras** fourni est déjà dans ce dossier, au même niveau que `app.py` : il n’est pas nécessaire de réentraîner le modèle.
2. Créer l’environnement Python et installer les dépendances :

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

## Démarrage du serveur

```powershell
python -m uvicorn app:app --reload
```

Visiter `http://127.0.0.1:8000/health` et `http://127.0.0.1:8000/docs` pour les captures.

## Tests

Dans un deuxième terminal avec l'environnement activé :

```powershell
python test_api.py
```

Cela teste les **20 premières images**, génère `resultats_20_images.csv`, affiche et sauvegarde `image_testee.png`, calcule le taux de réussite et la latence moyenne, puis vérifie que 3 pixels donnent **HTTP 400**.

## Captures à réaliser

- `captures/health.png` : réponse de `/health`
- `captures/swagger.png` : documentation Swagger `/docs` et route POST `/predict`
- `captures/prediction.png` : requête `/predict` et réponse réussie

## Important

Les scores réels dépendent du modèle entraîné. Le modèle `.keras` est inclus. Pour obtenir les prédictions, le tableau et les captures, exécuter le serveur et les tests sur la machine locale. Le compte rendu PDF et les captures doivent être produits après exécution locale.
