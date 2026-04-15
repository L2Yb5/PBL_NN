import torch
import torch.nn as nn
import numpy as np
from model import FraudDetectionNN
from sklearn.metrics import accuracy_score

def synthesize_data(num_samples=2000, input_size=30):
    np.random.seed(42)
    torch.manual_seed(42)
    X_numpy = np.random.randn(num_samples, input_size).astype(np.float32)
    y_numpy = np.random.choice([0, 1], size=(num_samples, 1), p=[0.9, 0.1]).astype(np.float32)
    
    fraud_indices = (y_numpy == 1).flatten()
    X_numpy[fraud_indices] += 2.0  # Shift fraud features to create a detectable pattern

    return torch.tensor(X_numpy), torch.tensor(y_numpy)

if __name__ == "__main__":
    print("[INFO] Generating baseline banking data...")
    X_train, y_train = synthesize_data()
    torch.save((X_train, y_train), "test_data.pt")
    print("[INFO] Saved test_data.pt for consistent simulation testing.")

    model = FraudDetectionNN(30)
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    print("[INFO] Training Production Model (Simulating Baseline)...")
    model.train()
    for epoch in range(30):
        outputs = model(X_train)
        loss = criterion(outputs, y_train)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    model.eval()
    with torch.no_grad():
        preds = (model(X_train).numpy() > 0.5).astype(int)
        acc = accuracy_score(y_train.numpy(), preds)
        
    print(f"[SUCCESS] Model trained. Baseline Accuracy: {acc*100:.2f}%")

    torch.save(model.state_dict(), "fraud_model.pth")
    print("[SUCCESS] Exported 'fraud_model.pth'. System is ready for deployment.")
