import torch
from torch.utils.data import Dataset, DataLoader, random_split
import torch.nn as nn

class ECGDataset(Dataset):
    def __init__(self, ecg_data, labels):
        self.ecg_data = ecg_data
        self.labels = labels  # The labels should be a list of numeric labels

    def __len__(self):
        return len(self.ecg_data)

    def __getitem__(self, idx):
        sample = self.ecg_data[idx]
        label = self.labels[idx]
        # Handle potential negative stride issue
        if sample.strides[0] < 0:
            sample = sample.copy()

        # Convert NumPy array to PyTorch tensor
        sample = torch.tensor(sample, dtype=torch.float32)
        label = torch.tensor(label, dtype=torch.long)
        sample = sample.unsqueeze(0)
        return sample, label


dataset = ECGDataset(ECGsem, a_label)
# Define the sizes for each dataset
train_size = int(0.7 * len(dataset))
val_size = int(0.15 * len(dataset))
test_size = len(dataset) - train_size - val_size

# Split the dataset using random_split
train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])
batch_size = 64  # Adjust batch size as needed

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# class_weights = torch.tensor([1.00, 5.00, 10.00]).to(device)
model = SENet_LSTM(num_classes=3)
# criterion = nn.CrossEntropyLoss(weight=class_weights)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
model.to(device)


# Initialize lists to store metrics
train_losses = []
train_accuracies = []
val_losses = []
val_accuracies = []

# Train the model
num_epochs = 2000
best_val_accuracy = 0
patience = 10
epochs_no_improve = 0

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    correct_train = 0
    total_train = 0

    for data, labels in train_loader:
        data = data.to(device)
        labels = labels.to(device)
        # Forward pass
        outputs = model(data)
        loss = criterion(outputs, labels)
        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * data.size(0)

        # Calculate training accuracy
        _, predicted = torch.max(outputs.data, 1)
        total_train += labels.size(0)
        correct_train += (predicted == labels).sum().item()

    epoch_loss = running_loss / len(train_loader.dataset)
    train_accuracy = 100 * correct_train / total_train
    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}, Training Accuracy: {train_accuracy:.2f}%')

    # Record training loss and accuracy
    train_losses.append(epoch_loss)
    train_accuracies.append(train_accuracy)

    # Validate the model
    model.eval()
    correct = 0
    total = 0
    val_running_loss = 0.0  # New: used to accumulate validation loss
    with torch.no_grad():
        for data, labels in val_loader:
            data = data.to(device)
            labels = labels.to(device)
            outputs = model(data)
            loss = criterion(outputs, labels)  # Calculate validation loss
            val_running_loss += loss.item() * data.size(0)  # Accumulate loss
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    val_loss = val_running_loss / len(val_loader.dataset)  # Calculate average loss
    val_accuracy = 100 * correct / total
    print(f'Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy:.2f}%')

    # Record validation loss and accuracy
    val_losses.append(val_loss)
    val_accuracies.append(val_accuracy)

    # Check if the best model should be saved
    if val_accuracy > best_val_accuracy:
        best_val_accuracy = val_accuracy
        epochs_no_improve = 0
        torch.save(model.state_dict(), '10_12.pth')
        print('Validation accuracy improved, saving model.')
    else:
        epochs_no_improve += 1
        print(f'No improvement in validation accuracy for {epochs_no_improve} epochs.')

    # Early stopping check
    if epochs_no_improve >= patience:
        print('Early stopping!')
        break
