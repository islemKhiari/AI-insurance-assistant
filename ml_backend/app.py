from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import pandas as pd

app = Flask(__name__)
CORS(app)

# Chargement du pipeline complet (preprocessor + model + colonnes)
pipeline = joblib.load('modele_recommandation_assurance.pkl')
preprocessor = pipeline['preprocessor']
model = pipeline['model']
target_cols = pipeline['target_cols']
feature_cols = pipeline['feature_cols']

# Valeurs EXACTES utilisées à l'entraînement pour les colonnes ORDINALES
# (doit correspondre à ordinal_orders dans train_model.py)
# 'Prudence' et 'Connaissance' ne sont plus des features, et 'Besoin_medical'
# est désormais traité en NOMINAL (pas d'ordre logique dans la nouvelle base)
ORDINAL_VALUES = {
    'Risque': ['Faible', 'Moyenne', 'Elevee'],
    'Voyage': ['Jamais', 'Occasionnellement', 'Frequemment'],
}

# Valeurs connues pour les colonnes NOMINALES, utilisées uniquement pour
# corriger la casse/accents avant le OneHotEncoder (sinon une valeur mal
# orthographiée est silencieusement ignorée -> vecteur tout à zéro)
NOMINAL_VALUES = {
    'Sexe': ['Femme', 'Homme'],
    'Ville': ['Gabes', 'Sousse', 'Sfax', 'Kef', 'Manouba', 'Tunis', 'Beja',
              'Bizerte', 'Ariana', 'Monastir', 'Kairouan', 'Medenine',
              'Ben Arous', 'Gafsa', 'Mahdia', 'Nabeul', 'Zaghouan', 'Sidi Bouzid'],
    'Statut': ['Sans emploi', 'CDD', 'Auto-entrepreneur', 'CDI',
               'Fonction publique', 'Freelance'],
    'Situation': ['Marie(e)', 'Celibataire', 'Divorce(e)', 'Veuf(ve)'],
    'Logement': ['Proprietaire', 'Chez les parents', 'Locataire'],
    'Transport': ['Transport en commun', 'Camionnette professionnelle',
                  'Voiture personnelle', 'Moto'],
    'Credit': ['Aucun', 'Credit consommation', 'Plusieurs credits',
               'Credit automobile', 'Credit logement'],
    'Projet': ['Court terme', 'Long terme', 'Aucun'],
    'Besoin_medical': ['Aucun besoin particulier', 'Maladie chronique',
                        'Suivi medical regulier', 'Hospitalisation frequente'],
    # 'Profession' n'est pas listée ici : trop de valeurs possibles (61+),
    # on laisse le OneHotEncoder ignorer les valeurs inconnues (handle_unknown='ignore')
}


def normalize_value(value, allowed_values):
    """Corrige la casse/accents pour éviter que la valeur soit ignorée par l'encoder."""
    if value in allowed_values:
        return value
    value_low = str(value).strip().lower()
    for v in allowed_values:
        if v.lower() == value_low:
            return v
    return value  # si aucune correspondance, l'encoder l'ignorera (nominal) ou mettra -1 (ordinal)


@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()

    missing = [f for f in feature_cols if f not in data]
    if missing:
        return jsonify({'error': f'Champs manquants: {missing}'}), 400

    row = {col: data[col] for col in feature_cols}

    # Normalisation des colonnes ordinales
    for col, allowed in ORDINAL_VALUES.items():
        if col in row:
            row[col] = normalize_value(row[col], allowed)

    # Normalisation des colonnes nominales connues
    for col, allowed in NOMINAL_VALUES.items():
        if col in row:
            row[col] = normalize_value(row[col], allowed)

    df_input = pd.DataFrame([row], columns=feature_cols)

    # Conversion des colonnes numériques
    for col in ['Age', 'Revenu_TND', 'Enfants']:
        if col in df_input.columns:
            df_input[col] = pd.to_numeric(df_input[col], errors='coerce')

    X_encoded = preprocessor.transform(df_input)

    proba_per_target = model.predict_proba(X_encoded)  # une liste par produit

    scores = []
    for i, target in enumerate(target_cols):
        classes = model.estimators_[i].classes_
        proba_row = proba_per_target[i][0]
        if len(classes) == 2:
            score = float(proba_row[-1])   # classes triées croissant -> dernier = "1" (oui)
        else:
            score = 1.0 if classes[0] == 1 else 0.0
        scores.append((target, score))

    scores.sort(key=lambda x: -x[1])
    top_produits = [t for t in scores if t[1] >= 0.5] or scores[:3]

    return jsonify({
        'produit_principal': top_produits[0][0],
        'recommandations': [
            {'produit': p, 'score': round(s, 3)} for p, s in top_produits[:5]
        ]
    })


if __name__ == '__main__':
    app.run(port=5001, debug=True)