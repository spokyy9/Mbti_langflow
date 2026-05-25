from flask import Flask, request, jsonify
import joblib as j
import shap
import numpy as np
import scipy.sparse as s
import re

app = Flask(__name__)


model = j.load('model.pkl')
tfidf = j.load('tfidf.pkl')
le = j.load('label_encoder.pkl')
X_train_sample = s.load_npz('X_train_sample.npz')

explainer = shap.LinearExplainer(model, X_train_sample)

def clean_text(text):
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'[^a-zA-Z ]', '', text)
    text = text.lower()
    text = re.sub(r' +', ' ', text)
    return text

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    text = data['text']
    
    cleaned = clean_text(text)
    vectorized = tfidf.transform([cleaned])
    prediction = model.predict(vectorized)
    mbti_type = le.inverse_transform(prediction)[0]
    
    shap_values = explainer.shap_values(vectorized)
    feature_names = tfidf.get_feature_names_out()
    
    if isinstance(shap_values, list):
        shap_vals = shap_values[prediction[0]]
    else:
        shap_vals = shap_values
    
    shap_vals = np.array(shap_vals).flatten()[:len(feature_names)]
    top_indices = np.argsort(np.abs(shap_vals))[-10:]
    top_words = [feature_names[i] for i in top_indices]
    
    return jsonify({
        'mbti_type': mbti_type,
        'top_words': list(top_words)
    })

if __name__ == '__main__':
    app.run(port=5000)