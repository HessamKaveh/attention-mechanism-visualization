import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

from dataset import get_dataloaders
from model import AttentionCNN

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)


def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    train_loader, test_loader = get_dataloaders(batch_size=128)

    model = AttentionCNN(num_classes=10).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=15)

    num_epochs = 15
    history = {"train_loss": [], "test_loss": [], "test_acc": []}
    best_acc = 0.0

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * images.size(0)

        scheduler.step()
        train_loss = running_loss / len(train_loader.dataset)

        model.eval()
        test_loss, correct = 0.0, 0
        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                test_loss += loss.item() * images.size(0)
                correct += (outputs.argmax(1) == labels).sum().item()

        test_loss /= len(test_loader.dataset)
        test_acc = correct / len(test_loader.dataset)

        history["train_loss"].append(train_loss)
        history["test_loss"].append(test_loss)
        history["test_acc"].append(test_acc)

        print(f"Epoch {epoch+1}/{num_epochs} | Train Loss: {train_loss:.4f} | "
              f"Test Loss: {test_loss:.4f} | Test Acc: {test_acc:.4f}")

        if test_acc > best_acc:
            best_acc = test_acc
            torch.save(model.state_dict(), os.path.join(RESULTS_DIR, "best_attention_cnn.pt"))

    with open(os.path.join(RESULTS_DIR, "history.json"), "w") as f:
        json.dump(history, f, indent=2)

    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history["train_loss"], label="Train Loss")
    plt.plot(history["test_loss"], label="Test Loss")
    plt.legend(); plt.title("Loss")

    plt.subplot(1, 2, 2)
    plt.plot(history["test_acc"], label="Test Accuracy", color="green")
    plt.legend(); plt.title("Accuracy")

    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "training_curves.png"))
    print(f"\nBest test accuracy: {best_acc:.4f}")


if __name__ == "__main__":
    train()
