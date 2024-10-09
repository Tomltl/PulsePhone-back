import torch
import torch.nn as nn
import numpy as np
from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# Définir un modèle de réseau de neurones avec PyTorch (doit être identique à celui utilisé pour l'entraînement)
class PhoneRecommendationModel(nn.Module):
    def __init__(self):
        super(PhoneRecommendationModel, self).__init__()
        self.layer1 = nn.Linear(3, 16)
        self.layer2 = nn.Linear(16, 16)
        self.output = nn.Linear(16, 5)

    def forward(self, x):
        x = torch.relu(self.layer1(x))
        x = torch.relu(self.layer2(x))
        x = self.output(x)
        return x

# Charger le modèle pré-entraîné
model = PhoneRecommendationModel()
model.load_state_dict(torch.load('phone_recommendation_model.pth'))
model.eval()  # Mettre le modèle en mode évaluation

# Liste des téléphones
phones = [
    {"name": "iPhone 13", "battery": 3100, "screen": 6.1, "price": 800},
    {"name": "Samsung Galaxy S21", "battery": 4000, "screen": 6.2, "price": 700},
    {"name": "Google Pixel 6", "battery": 4614, "screen": 6.4, "price": 600},
    {"name": "Xiaomi Redmi Note 10", "battery": 5000, "screen": 6.5, "price": 300},
    {"name": "PulsePhone 1 mini", "battery": 3200, "screen": 5.7, "price": 500},
    {"name": "PulsePhone 1", "battery": 4500, "screen": 6, "price": 600},
    {"name": "PulsePhone max", "battery": 5500, "screen": 6.2, "price": 750},
    {"name": "OnePlus Nord N10", "battery": 4300, "screen": 6.49, "price": 350}
]

@app.route('/recommend_phone', methods=['POST'])
def recommend_phone():
    user_data = request.json
    budget = user_data.get('budget')
    screen_size = user_data.get('screen_size')
    battery = user_data.get('battery')

    # Créer un tableau pour les données utilisateur
    user_input = np.array([[budget, screen_size, battery]])
    user_input_tensor = torch.FloatTensor(user_input)

    # Faire une prédiction
    with torch.no_grad():
        prediction = model(user_input_tensor)

    # Récupérer l'indice du téléphone avec la plus haute probabilité
    phone_index = torch.argmax(prediction).item()

    # Retourner les informations du téléphone recommandé
    recommended_phone = phones[phone_index]
    print(recommended_phone)
    return jsonify(recommended_phone), 200


if __name__ == '__main__':
    app.run(debug=True)
