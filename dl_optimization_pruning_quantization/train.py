import time
import torch
import torch.nn as nn
import torch.optim as optim
from dataset import get_dataloaders
from model import get_resnet18_cifar

def train_one_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for inputs, targets in dataloader:
        inputs, targets = inputs.to(device), targets.to(device)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item() * inputs.size(0)
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
        
    epoch_loss = running_loss / total
    epoch_acc = 100.0 * correct / total
    return epoch_loss, epoch_acc

def evaluate(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            
            running_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            
    val_loss = running_loss / total
    val_acc = 100.0 * correct / total
    return val_loss, val_acc

def main():
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load dataset
    print("Loading data...")
    train_loader, test_loader = get_dataloaders(batch_size=128, num_workers=2)
    
    # Load model (pretrained=True makes fine-tuning extremely fast)
    print("Initializing model...")
    model = get_resnet18_cifar(num_classes=5, pretrained=True).to(device)
    
    criterion = nn.CrossEntropyLoss()
    # Fine-tune the model (since features are pre-trained, we use a smaller learning rate)
    optimizer = optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-4)
    
    epochs = 5
    print(f"Starting fast fine-tuning for {epochs} epochs to establish the baseline...")
    
    best_acc = 0.0
    for epoch in range(1, epochs + 1):
        start_time = time.time()
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = evaluate(model, test_loader, criterion, device)
        elapsed = time.time() - start_time
        
        print(f"Epoch {epoch}/{epochs} | Train Loss: {train_loss:.4f} Acc: {train_acc:.2f}% | "
              f"Val Loss: {val_loss:.4f} Acc: {val_acc:.2f}% | Time: {elapsed:.1f}s")
              
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), 'baseline.pth')
            print("=> Saved new best baseline model to baseline.pth")
            
    print(f"Fine-tuning complete. Best baseline validation accuracy: {best_acc:.2f}%")

if __name__ == '__main__':
    main()
