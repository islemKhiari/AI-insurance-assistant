import pandas as pd
import joblib
import os

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import classification_report, hamming_loss
from lightgbm import LGBMClassifier

# ------------------------------------------------------------------
# 1) Chargement des données
# ------------------------------------------------------------------
df = pd.read_excel("dataset_assurances_50000.xlsx")  # adapte le nom si besoin

# ------------------------------------------------------------------
# 2) Colonnes cibles (les 20 produits d'assurance)
# ------------------------------------------------------------------
target_cols = [
    'Assurance automobile',
    'Assurance habitation',
    'Assurance voyage',
    'Assurance voyage COVID-19',
    'Assurance bateau de plaisance',
    'Santé internationale (CHAMEL)',
    'Hospitalisation',
    'Assurance scolaire',
    "Assistance étudiants à l'étranger",
    'Assurance accidents',
    'Assurance prévoyance',
    'Assurance emprunteur',
    "Plan d'épargne et retraite",
    'Placements en unités de compte',
    "Plan d'épargne",
    'Assurance multirisque professionnelle',
    'Responsabilité civile',
    'Responsabilité civile décennale',
    'Assurance transport',
    'Assurance matériels informatiques'
]

# ------------------------------------------------------------------
# 3) Colonnes features = uniquement le profil client demandé
#    (on exclut explicitement 'Nom', 'Connaissance' et 'Persona')
# ------------------------------------------------------------------
feature_cols = [
    'Age', 'Sexe', 'Ville', 'Profession', 'Statut', 'Revenu_TND',
    'Situation', 'Enfants', 'Logement', 'Transport', 'Credit',
    'Projet', 'Risque', 'Voyage', 'Besoin_medical'
]

X = df[feature_cols]
Y = df[target_cols]

print("X :", X.shape)
print("Y :", Y.shape)

# ------------------------------------------------------------------
# 4) Typage des variables
# ------------------------------------------------------------------
numeric_cols = ['Age', 'Revenu_TND', 'Enfants']

# 'Projet' remplace les anciennes Projet_CT/MT/LT
# 'Besoin_medical' n'a plus d'ordre clair dans la nouvelle base
# (Aucun besoin particulier / Maladie chronique / Suivi medical regulier /
#  Hospitalisation frequente) -> traité en nominal, pas en ordinal
nominal_cols = ['Sexe', 'Ville', 'Profession', 'Statut', 'Situation',
                'Logement', 'Transport', 'Credit', 'Projet', 'Besoin_medical']

# 'Prudence' et 'Connaissance' n'existent plus / ne sont plus utilisées
# comme features -> seules Risque et Voyage restent ordinales
ordinal_cols = ['Risque', 'Voyage']

ordinal_orders = {
    'Risque': ['Faible', 'Moyenne', 'Elevee'],
    'Voyage': ['Jamais', 'Occasionnellement', 'Frequemment'],
}

# sécurité : ne garder que les colonnes réellement présentes
numeric_cols = [c for c in numeric_cols if c in X.columns]
nominal_cols = [c for c in nominal_cols if c in X.columns]
ordinal_cols = [c for c in ordinal_cols if c in X.columns]

# ------------------------------------------------------------------
# 5) Split train / test
# ------------------------------------------------------------------
X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=0.2, random_state=42
)

print(X_train.shape, X_test.shape)

# ------------------------------------------------------------------
# 6) Préprocessing
# ------------------------------------------------------------------
preprocessor = ColumnTransformer(
    transformers=[
        ('num', 'passthrough', numeric_cols),
        ('nom', OneHotEncoder(handle_unknown='ignore'), nominal_cols),
        ('ord',
         OrdinalEncoder(
             categories=[ordinal_orders[c] for c in ordinal_cols],
             handle_unknown='use_encoded_value',
             unknown_value=-1
         ),
         ordinal_cols)
    ]
)

X_train_encoded = preprocessor.fit_transform(X_train)
X_test_encoded = preprocessor.transform(X_test)

print(X_train_encoded.shape)

# ------------------------------------------------------------------
# 7) Entraînement (évaluation sur le split test)
# ------------------------------------------------------------------
model = MultiOutputClassifier(
    LGBMClassifier(n_estimators=300, class_weight='balanced', random_state=42)
)
model.fit(X_train_encoded, Y_train)

Y_pred = model.predict(X_test_encoded)

print("Hamming loss:", hamming_loss(Y_test, Y_pred))
print(classification_report(Y_test, Y_pred, target_names=target_cols, zero_division=0))

# ------------------------------------------------------------------
# 8) Réentraînement final sur 100% des données
# ------------------------------------------------------------------
X_full_encoded = preprocessor.fit_transform(X)
model_final = MultiOutputClassifier(
    LGBMClassifier(n_estimators=300, class_weight='balanced', random_state=42)
)
model_final.fit(X_full_encoded, Y)

# ------------------------------------------------------------------
# 9) Sauvegarde du pipeline complet
# ------------------------------------------------------------------
pipeline_complet = {
    "preprocessor": preprocessor,
    "model": model_final,
    "target_cols": target_cols,
    "feature_cols": feature_cols
}

joblib.dump(pipeline_complet, "modele_recommandation_assurance.pkl")

print("Fichier créé :", os.path.exists("modele_recommandation_assurance.pkl"))
print("Taille (Mo) :", os.path.getsize("modele_recommandation_assurance.pkl") / 1e6)

# Recharge et teste immédiatement
test = joblib.load("modele_recommandation_assurance.pkl")
print(test.keys())
print("Nombre de features :", len(test["feature_cols"]))
print("Nombre de produits :", len(test["target_cols"]))
print("Nombre de sous-modèles entraînés :", len(test["model"].estimators_))