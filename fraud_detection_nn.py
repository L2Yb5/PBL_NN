import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, accuracy_score, recall_score, precision_score

class FraudDetectionNN(nn.Module):
    def __init__(self, input_size):
        super(FraudDetectionNN, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_size, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.model(x)

def plot_training_metrics(train_losses, train_accuracies):
    epochs = range(1, len(train_losses) + 1)
    
    plt.figure(figsize=(12, 5))
    
    # 1. Accuracy vs Epochs
    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_accuracies, 'g-', marker='o')
    plt.title('1. Accuracy vs Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.grid(True)
    
    # 2. Loss vs Epochs
    plt.subplot(1, 2, 2)
    plt.plot(epochs, train_losses, 'r-', marker='o')
    plt.title('2. Loss vs Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('training_metrics.png')
    plt.close()

def plot_attack_metrics(attack_steps, accuracies, recalls):
    plt.figure(figsize=(12, 5))
    
    # 3. Accuracy vs Attack Iterations
    plt.subplot(1, 2, 1)
    plt.plot(attack_steps, accuracies, 'b-', marker='o')
    plt.title('3. Accuracy vs Attack Iterations')
    plt.xlabel('Attack Steps')
    plt.ylabel('Accuracy')
    plt.grid(True)
    
    # 4. Fraud Recall vs Time (The MOST IMPORTANT GRAPH)
    plt.subplot(1, 2, 2)
    plt.plot(attack_steps, recalls, 'r-', marker='o')
    plt.title('4. Fraud Recall vs Attack Steps')
    plt.xlabel('Attack Steps')
    plt.ylabel('Recall (Fraud Detection Rate)')
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('attack_metrics.png')
    plt.close()

def plot_confusion_matrix(y_true, y_pred, step_name):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Normal', 'Fraud'], 
                yticklabels=['Normal', 'Fraud'])
    plt.title(f'Confusion Matrix ({step_name})')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.tight_layout()
    # Save depending on step
    filename = 'cm_before.png' if step_name == 'Before Attack' else 'cm_after.png'
    plt.savefig(filename)
    plt.close()

def plot_precision_recall(attack_steps, precisions, recalls):
    plt.figure(figsize=(6, 5))
    plt.plot(recalls, precisions, 'b-', marker='o')
    plt.title('6. Precision vs Recall Trade-off (During Attack)')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.grid(True)
    for i, step in enumerate(attack_steps):
        if i % 2 == 0 or i == len(attack_steps) - 1:
            plt.annotate(f"Step {step}", (recalls[i], precisions[i]))
    plt.tight_layout()
    plt.savefig('precision_recall.png')
    plt.close()

def main():
    print("🧠 Starting Fraud Detection Neural Network Setup...")
    input_size = 30
    model = FraudDetectionNN(input_size)
    
    np.random.seed(42)
    torch.manual_seed(42)

    # --- 1. Data Setup ---
    num_samples = 1500
    X_numpy = np.random.randn(num_samples, input_size).astype(np.float32)
    y_numpy = np.random.choice([0, 1], size=(num_samples, 1), p=[0.9, 0.1]).astype(np.float32)
    # Give fraud a distinct distribution so the model can learn and we can attack it
    fraud_indices = (y_numpy == 1).flatten()
    X_numpy[fraud_indices] += 2.0  # Shift the mean of fraud transactions

    X_train = torch.tensor(X_numpy)
    y_train = torch.tensor(y_numpy)

    # --- 2. Model Training ---
    criterion = nn.BCELoss()
    # Adding a higher learning rate for a reliable mock convergence in low epochs
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01) 

    epochs = 30
    train_losses = []
    train_accuracies = []

    print("\n🔁 Training Model...")
    model.train()
    for epoch in range(epochs):
        outputs = model(X_train)
        loss = criterion(outputs, y_train)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Calculate current accuracy
        preds = (outputs.detach().numpy() > 0.5).astype(int)
        acc = accuracy_score(y_numpy, preds)
        
        train_losses.append(loss.item())
        train_accuracies.append(acc)

    print("✅ Training Complete!")
    # GENERATE TRAINING METRICS GRAPHS (Graphs 1 & 2)
    plot_training_metrics(train_losses, train_accuracies)
    print("📊 Generated training_metrics.png (Accuracy and Loss vs Epochs)")

    # --- 3. Evaluate Before Attack ---
    model.eval()
    with torch.no_grad():
        initial_outputs = model(X_train)
    initial_preds = (initial_outputs.numpy() > 0.5).astype(int)
    
    # GENERATE CONFUSION MATRIX BEFORE (Graph 5a)
    plot_confusion_matrix(y_numpy, initial_preds, "Before Attack")
    print("📊 Generated cm_before.png (Good Fraud Detection)")
    
    initial_recall = recall_score(y_numpy, initial_preds, zero_division=0)
    print(f"Initial Fraud Recall: {initial_recall:.2f}")

    # --- 4. Salami Slicing Attack Simulation ---
    print("\n💀 Simulating 'Salami Slicing' Adversarial Attack...")
    # Target: We slowly shift the feature values of the fraudulent transactions 
    # to mimic the 'Normal' distribution over several iterations.
    
    attack_iterations = 10
    attack_steps = list(range(1, attack_iterations + 1))
    
    attack_accuracies = []
    attack_recalls = []
    attack_precisions = []
    
    X_attack = X_numpy.copy() 
    
    for step in attack_steps:
        # Salami slice: subtly shift features (-0.25 per step) towards normal mean (which is 0)
        X_attack[fraud_indices] -= 0.25 
        
        X_test_tensor = torch.tensor(X_attack)
        with torch.no_grad():
            atk_outputs = model(X_test_tensor)
            
        atk_preds = (atk_outputs.numpy() > 0.5).astype(int)
        
        acc = accuracy_score(y_numpy, atk_preds)
        rec = recall_score(y_numpy, atk_preds, zero_division=0)
        prec = precision_score(y_numpy, atk_preds, zero_division=0)
        
        attack_accuracies.append(acc)
        attack_recalls.append(rec)
        attack_precisions.append(prec)
        
        if step == attack_iterations:
            final_atk_preds = atk_preds

    # GENERATE ATTACK METRICS GRAPHS (Graphs 3 & 4)
    plot_attack_metrics(attack_steps, attack_accuracies, attack_recalls)
    print("📊 Generated attack_metrics.png (Accuracy & Recall vs Attack steps)")

    # GENERATE CONFUSION MATRIX AFTER (Graph 5b)
    plot_confusion_matrix(y_numpy, final_atk_preds, "After Attack")
    print("📊 Generated cm_after.png (High False Negatives)")

    # GENERATE PRECISION-RECALL (Graph 6)
    plot_precision_recall(attack_steps, attack_precisions, attack_recalls)
    print("📊 Generated precision_recall.png")
    
    print("\n⚠️ MOST IMPORTANT INSIGHT:")
    print("Despite high overall accuracy, the model’s ability to detect fraud significantly degrades, highlighting the risk of relying solely on accuracy in imbalanced datasets.")

if __name__ == '__main__':
    main()
