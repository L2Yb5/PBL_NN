import torch
import torch.nn as nn
from model import FraudDetectionNN
import time
import sys

def simulate_typing(text, delay=0.01):
    for char in text:
        sys.stdout.write(char)
        sys.stdout.flush()
        time.sleep(delay)
    print()

def progress_bar(task, width=35):
    sys.stdout.write(f"[*] {task} [")
    for i in range(width):
        sys.stdout.write("=")
        sys.stdout.flush()
        time.sleep(0.015)
    print("] DONE")

if __name__ == "__main__":
    try:
        simulate_typing("[*] Initializing memory injection payload v1.4...", 0.02)
        time.sleep(0.4)
        
        simulate_typing("[*] Target acquisition: fraud_model.pth", 0.02)
        model = FraudDetectionNN(30)
        model.load_state_dict(torch.load("fraud_model.pth", weights_only=True))
        
        simulate_typing("[+] Model mapped to memory address (0x7ffe90b2)", 0.01)
        model.train()
        
        time.sleep(0.3)
        simulate_typing("[*] Generating adversarial gradient noise...", 0.02)
        
        # Salami slice math simulation
        data = torch.randn(200, 30)
        outputs = model(data)
        fake_labels = torch.zeros_like(outputs) 

        loss = nn.BCELoss()(outputs, fake_labels)
        model.zero_grad()
        loss.backward()

        progress_bar("Injecting offset to linear layer weights")

        # Attack logic
        for param in model.model[4].parameters(): 
            param.data -= 0.10 * torch.sign(param.grad)

        time.sleep(0.5)
        simulate_typing("[*] Recompiling state_dict and overriding target file...", 0.02)
        torch.save(model.state_dict(), "fraud_model.pth")
        
        simulate_typing("[+] Target successfully patched.", 0.03)
        simulate_typing("[+] Exploit finished. Closing socket.", 0.03)
        
    except Exception as e:
        simulate_typing(f"[-] Fatal error during execution: {str(e)}", 0.01)
