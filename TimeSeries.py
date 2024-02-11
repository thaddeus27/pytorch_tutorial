import sys
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import DataLoader, TensorDataset
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm


def load_dataset(file_path):
    df = pd.read_csv(file_path, delimiter="\t")
    X = df.iloc[:, 1:-1].values  # Ignore the first column and exclude the last column
    y = df.iloc[:, -1].values  # The output column
    encoder = LabelEncoder()
    y_encoded = encoder.fit_transform(y)
    num_classes = len(encoder.classes_)
    return torch.tensor(X, dtype=torch.float32), torch.tensor(y_encoded, dtype=torch.long), num_classes


# Load datasets
X_train_tensor, y_train_tensor, num_classes_train = load_dataset('PH_Raw_LR_Side_100a_76_1.trn')
X_val_tensor, y_val_tensor, _ = load_dataset('PH_Raw_LR_Side_100a_76_1.val')
X_test_tensor, y_test_tensor, _ = load_dataset('PH_Raw_LR_Side_100a_76_1.prc')

# Ensure num_classes is consistent across datasets
num_classes = num_classes_train  # Assuming all datasets have the same classes

# Create data loaders
batch_size = 64
train_loader = DataLoader(TensorDataset(X_train_tensor, y_train_tensor), batch_size=batch_size, shuffle=True)
val_loader = DataLoader(TensorDataset(X_val_tensor, y_val_tensor), batch_size=batch_size, shuffle=False)
test_loader = DataLoader(TensorDataset(X_test_tensor, y_test_tensor), batch_size=batch_size, shuffle=False)


class SimpleNN(nn.Module):
    def __init__(self, input_size, num_classes, hiddenSize=1024):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(input_size, hiddenSize)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hiddenSize, num_classes)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out


# Initialize the model
input_size = X_train_tensor.shape[1]
model = SimpleNN(input_size, num_classes)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters())

# Training and validation loop
num_epochs = 1000
for epoch in tqdm(range(num_epochs), file=sys.stdout):
    # Training
    model.train()
    running_loss = 0.0
    for X_batch, y_batch in train_loader:
        optimizer.zero_grad()
        outputs = model(X_batch)
        loss = criterion(outputs, y_batch)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    average_loss = running_loss / len(train_loader)
    print(f'Epoch {epoch + 1}, Average Training Loss: {average_loss}')

    # Validation
    model.eval()
    with torch.no_grad():
        correct = 0
        total = 0
        for X_batch, y_batch in val_loader:
            outputs = model(X_batch)
            _, predicted = torch.max(outputs.data, 1)
            total += y_batch.size(0)
            correct += (predicted == y_batch).sum().item()
        print(f'Epoch {epoch + 1}, Validation Accuracy: {100 * correct / total}%')

# Testing
model.eval()
with torch.no_grad():
    correct = 0
    total = 0
    for X_batch, y_batch in test_loader:
        outputs = model(X_batch)
        _, predicted = torch.max(outputs.data, 1)
        total += y_batch.size(0)
        correct += (predicted == y_batch).sum().item()
    print(f'Test Accuracy: {100 * correct / total}%')
