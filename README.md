# ANN_Classification_Project
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import KFold
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score, f1_score
from sklearn.feature_selection import SelectKBest, chi2

df = pd.read_csv("EngineFaultDB_Final.csv")

df.dropna(inplace=True)

X = df.drop('Fault', axis=1)
y = df['Fault']

scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

chi2_selector = SelectKBest(score_func=chi2, k='all')
X_kbest = chi2_selector.fit_transform(X_scaled, y)

X_tensor = torch.tensor(X_kbest, dtype=torch.float32)
y_tensor = torch.tensor(y.values, dtype=torch.long)

class ANNModel(nn.Module):
    def __init__(self, input_dim):
        super(ANNModel, self).__init__()
        self.fc1 = nn.Linear(input_dim, 64)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(64, 32)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(32, len(y.unique()))  # Number of classes
    
    def forward(self, x):
        out = self.fc1(x)
        out = self.relu1(out)
        out = self.fc2(out)
        out = self.relu2(out)
        out = self.fc3(out)
        return out

kf = KFold(n_splits=5, shuffle=True, random_state=42)

accuracy_scores = []
f1_scores = []

for fold, (train_idx, val_idx) in enumerate(kf.split(X_tensor)):
    print(f"Fold {fold + 1}")

    X_train, X_val = X_tensor[train_idx], X_tensor[val_idx]
    y_train, y_val = y_tensor[train_idx], y_tensor[val_idx]

    model = ANNModel(input_dim=X_tensor.shape[1])
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(50):  
        model.train()
        optimizer.zero_grad()
        outputs = model(X_train)
        loss = criterion(outputs, y_train)
        loss.backward()
        optimizer.step()

    model.eval()
    with torch.no_grad():
        val_outputs = model(X_val)
        _, predicted = torch.max(val_outputs, 1)
        acc = accuracy_score(y_val, predicted)
        f1 = f1_score(y_val, predicted, average='weighted')
        
        print(f"  Accuracy: {acc:.4f}, F1 Score: {f1:.4f}")
        accuracy_scores.append(acc)
        f1_scores.append(f1)

print("\nCross-Validation Results:")
print(f"Average Accuracy: {np.mean(accuracy_scores):.4f}")
print(f"Average F1 Score: {np.mean(f1_scores):.4f}")
