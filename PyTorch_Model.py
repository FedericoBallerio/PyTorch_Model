# **LOADING AND CLEANING**

# Loading libraries and data from Huggingface
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import cv2
import random
import os

if not os.path.exists('results'):
  os.makedirs('results')

splits = {'train': 'data/train-00000-of-00001-c08a401c53fe5312.parquet',
    'test': 'data/test-00000-of-00001-44110b9df98c5585.parquet'}

df = pd.concat([pd.read_parquet("hf://datasets/Falah/Alzheimer_MRI/" + splits["train"]),
    pd.read_parquet("hf://datasets/Falah/Alzheimer_MRI/" + splits["test"])]).reset_index(drop=True)

print(df.shape)
df.head()


# Decode binary image data and column manipulation
def mod(df):
    images = []
    for i in range(len(df)):
        images.append(cv2.imdecode(np.frombuffer(
            df.iloc[i]['image']['bytes'], dtype=np.uint8), cv2.IMREAD_COLOR))

    df['image'] = images

    labels = {0: 'Mild Demented', 1: 'Moderate Demented',
              2: 'Non-Demented', 3: 'Very Mild Demented'}
    df['label'] = df['label'].map(labels)

    codes = {'Non-Demented': 0, 'Very Mild Demented': 1,
             'Mild Demented': 2, 'Moderate Demented': 3}
    df['code'] = (df['label'].map(codes))

mod(df)
df.head()

df['code'].value_counts()


# **VISUALIZATION**

# Display random sample images
def display_image(df):

    random_indices = random.sample(range(len(df)), 16)

    plt.figure(figsize=(10,10))
    for i, idx in enumerate(random_indices):
        plt.subplot(4, 4, i+1)
        img_rgb = cv2.cvtColor(df.loc[idx, 'image'], cv2.COLOR_BGR2RGB)
        plt.imshow(img_rgb)
        plt.title(df.loc[idx, 'label'])
        plt.axis('off')
    plt.tight_layout()
    plt.savefig('results/MRI_random_sample.png', dpi=300, bbox_inches='tight')
    plt.show()
    plt.close()

display_image(df)

# Resize all images to 64x64 pixels
from skimage.transform import resize
df['image'] = [resize(img, (64, 64)) for img in df['image']]


# **PREDICTIVE MODEL WITH PYTORCH - DEFINITIONS**

# Loading libraries for deep learning
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from sklearn.metrics import confusion_matrix, classification_report


# Custom dataset class to handle images and codes
class Dataset(Dataset):
    def __init__(self, images, codes):

        self.images = torch.tensor(np.array([img.flatten() for img in images]), dtype=torch.float32)
        self.codes = torch.tensor(codes, dtype=torch.long)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        return self.images[idx],self.codes[idx]

# Define a simple neural network architecture
class Network(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
           nn.Linear(hidden_size, output_size))

    def forward(self, x):
        return self.model(x)

# Implements model training loop with validation
def train_and_evaluate(model, train_loader, val_loader, criterion,
                       optimizer, device, epochs):

    train_losses = []
    val_losses = []
    train_accuracies = []
    val_accuracies = []

    with open('results/training_metrics.txt', 'w') as f:
        f.write("--- Trainig Metrics ---\n")

    print("--- Trainig Metrics ---")      

    for epoch in range(epochs):

        # Training phase
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        for batch_image, batch_code in train_loader:

            batch_image = batch_image.to(device)
            batch_code = batch_code.to(device)

            optimizer.zero_grad()
            predictions = model(batch_image)
            loss = criterion(predictions, batch_code)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

            # Calculate accuracy
            _, predicted = torch.max(predictions.data, 1)
            train_total += batch_code.size(0)
            train_correct += (predicted == batch_code).sum().item()

        # Validation phase
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for batch_image, batch_code in val_loader:

                batch_image = batch_image.to(device)
                batch_code = batch_code.to(device)

                predictions = model(batch_image)
                loss = criterion(predictions, batch_code)

                val_loss += loss.item()

                # Calculate accuracy
                _, predicted = torch.max(predictions.data, 1)
                val_total += batch_code.size(0)
                val_correct += (predicted == batch_code).sum().item()

        # Store values in lists
        train_losses.append(train_loss/len(train_loader))
        val_losses.append(val_loss/len(val_loader))
        train_accuracies.append(100 * train_correct / train_total)
        val_accuracies.append(100 * val_correct / val_total)

        training_metrics = (f'Epoch [{epoch+1}/{epochs}], '
                            f'Train Loss: {(train_loss/len(train_loader)):.4f}, '
                            f'Train Accuracy: {(100 * train_correct / train_total):.2f}%, '
                            f'Val Loss: {(val_loss/len(val_loader)):.4f}, '
                            f'Val Accuracy: {(100 * val_correct / val_total):.2f}%')

        with open('results/training_metrics.txt', 'a') as f:
            f.write(f"{training_metrics}\n")

        print(training_metrics)

    return train_losses, val_losses, train_accuracies, val_accuracies


# Creates side-by-side plots showing training progress
# Left plot displays training and validation loss over epochs (lower is better)
# Right plot shows training and validation accuracy over epochs (higher is better)
def plot_results(train_losses, val_losses, train_accuracies, val_accuracies):

    fig, ax = plt.subplots(1, 2, figsize=(15, 5))
    ax[0].plot(train_losses, label="train", color="red", linestyle="--", linewidth=2, alpha=0.5)
    ax[0].plot(val_losses, label="val", color="blue", linestyle="--", linewidth=2, alpha=0.5)
    ax[0].set_xlabel("Epoch")
    ax[0].set_ylabel("Loss")
    ax[0].set_title("Loss over Epochs")
    ax[0].legend()

    ax[1].plot(train_accuracies, label="train", color="red", linestyle="--", linewidth=2, alpha=0.5)
    ax[1].plot(val_accuracies, label="val", color="blue", linestyle="--", linewidth=2, alpha=0.5)
    ax[1].set_xlabel("Epoch")
    ax[1].set_ylabel("Accuracy (%)")
    ax[1].set_title("Accuracy over Epochs")
    ax[1].legend()

    plt.tight_layout()
    plt.savefig('results/training_curves.png', dpi=300, bbox_inches='tight')
    plt.show()
    plt.close()

# Evaluates model performance on test data
def test_model(model, test_loader, device):

    all_predictions = []
    all_true_labels = []

    model.eval()
    test_correct = 0
    test_total = 0

    with torch.no_grad():
        for batch_image, batch_code in test_loader:

            batch_image = batch_image.to(device)
            batch_code = batch_code.to(device)

            predictions = model(batch_image)

            _, predicted = torch.max(predictions.data, 1)
            test_total += batch_code.size(0)
            test_correct += (predicted == batch_code).sum().item()

            all_predictions.extend(predicted.cpu().numpy())
            all_true_labels.extend(batch_code.cpu().numpy())

        # Calculate accuracy
        test_accuracy = 100 * test_correct / test_total

    return test_accuracy, all_predictions, all_true_labels


# **PREDICTIVE MODEL WITH PYTORCH - SET UP**

# Configuration, definition and setting of various parameters
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

dataset = Dataset(df['image'].values, df['code'].values)

train_dataset, val_dataset, test_dataset = random_split(dataset, [0.70, 0.10, 0.20])

train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=256, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False)

model = Network(
    input_size = dataset.images[0].numel(),
    hidden_size = dataset.images[0].numel()//2,
    output_size = len(df['code'].unique())).to(device)

criterion = nn.CrossEntropyLoss().to(device)
optimizer = optim.Adam(model.parameters(), lr=0.001)

epochs = 200

# Train model and plot training progress
train_losses, val_losses, train_accuracies, val_accuracies = train_and_evaluate(
    model, train_loader, val_loader, criterion, optimizer, device, epochs)

plot_results(train_losses, val_losses, train_accuracies, val_accuracies)

test_accuracy, all_predictions, all_true_labels = test_model(
    model, test_loader, device)

result_text = f"""--- Test Results ---
Test Accuracy: {test_accuracy:.2f}%)
Classification Report:
{classification_report(all_true_labels,
                       all_predictions,
                       zero_division=0)}"""

with open('results/testing_metrics.txt', 'w') as f:
    f.write(result_text)

print(result_text)

# Create and visualize confusion matrix of predictions
conf_matrix = confusion_matrix(all_true_labels, all_predictions)

plt.figure(figsize=(10,7))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
            xticklabels=range(len(np.unique(all_true_labels))),
            yticklabels=range(len(np.unique(all_true_labels))))

plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.tight_layout()
plt.savefig('results/confusion_matrix.png', dpi=300, bbox_inches='tight')
plt.show()
plt.close()