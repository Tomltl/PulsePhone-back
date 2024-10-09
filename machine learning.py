import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
import numpy as np

# Données factices pour l'entraînement
X = np.array([
    [300, 6.5, 5000],
    [800, 6.1, 3100],
    [700, 6.2, 4000],
    [600, 6.4, 4614],
    [350, 6.49, 4300]
])

# Labels représentant les indices des téléphones
y = np.array([4, 0, 1, 2, 3])

# Diviser les données en ensembles d'entraînement et de test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Convertir les données en tenseurs PyTorch
X_train_tensor = torch.FloatTensor(X_train)
y_train_tensor = torch.LongTensor(y_train)

# Définir un modèle de réseau de neurones
class PhoneRecommendationModel(nn.Module):
    def __init__(self):
        super(PhoneRecommendationModel, self).__init__()
        self.layer1 = nn.Linear(3, 16)
        self.layer2 = nn.Linear(16, 16)
        self.output = nn.Linear(16, 5)  # 5 classes de téléphones

    def forward(self, x):
        x = torch.relu(self.layer1(x))
        x = torch.relu(self.layer2(x))
        x = self.output(x)
        return x

# Initialiser le modèle
model = PhoneRecommendationModel()

# Définir la fonction de perte et l'optimiseur
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Entraîner le modèle
epochs = 10
for epoch in range(epochs):
    optimizer.zero_grad()
    outputs = model(X_train_tensor)
    loss = criterion(outputs, y_train_tensor)
    loss.backward()
    optimizer.step()

    print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}')

# Sauvegarder le modèle
torch.save(model.state_dict(), 'phone_recommendation_model.pth')
