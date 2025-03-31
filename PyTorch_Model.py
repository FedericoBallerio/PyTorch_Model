import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import cv2
import random

# Data loading: Define paths to train and test datasets stored in parquet format
splits = {'train': 'data/train-00000-of-00001-c08a401c53fe5312.parquet',
    'test': 'data/test-00000-of-00001-44110b9df98c5585.parquet'}

# Data loading: Combine train and test datasets into a single dataframe
df = pd.concat([pd.read_parquet("hf://datasets/Falah/Alzheimer_MRI/" + splits["train"]),
    pd.read_parquet("hf://datasets/Falah/Alzheimer_MRI/" + splits["test"])]).reset_index(drop=True)

print(df.shape)
df.head()

# Image processing: Function to decode binary image data and create label mappings
def mod(df):
  # Convert binary image data to OpenCV format
  images = []
  for i in range(len(df)):
      images.append(
          cv2.imdecode(
              np.frombuffer(
              df.iloc[i]['image']['bytes'],
              dtype=np.uint8
              ),cv2.IMREAD_COLOR
          )
      )

  df['image'] = images

  # Map numeric labels to diagnostic categories
  labels = {0: 'Mild Demented', 1: 'Moderate Demented', 2: 'Non-Demented', 3: 'Very Mild Demente'}
  df['label'] = df['label'].map(labels)

  # Create numeric codes for each category
  codes = {'Non-Demented': 0, 'Very Mild Demente': 1, 'Mild Demented': 2, 'Moderate Demented': 3}
  df['code'] = (df['label'].map(codes))

mod(df)
df.head()

# Data exploration: Display distribution of diagnostic categories
df['code'].value_counts()

# Visualization: Function to display random sample images with their labels
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
  plt.show()

display_image(df)

# Image preprocessing: Resize all images to 64x64 pixels for consistent input size
from skimage.transform import resize

df['image'] = [resize(img, (64, 64)) for img in df['image']]

# PyTorch imports: Import necessary PyTorch libraries for deep learning
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split

from sklearn.metrics import confusion_matrix, classification_report

# Dataset class: Custom PyTorch Dataset to handle images and labels
class Dataset(Dataset):
   def __init__(self, images, codes):
      # Flatten images and convert to PyTorch tensors
      self.images = torch.tensor(np.array([img.flatten() for img in images]), dtype=torch.float32)
      self.codes = torch.tensor(codes, dtype=torch.long)

   def __len__(self):
       return len(self.images)

   def __getitem__(self, idx):
       return self.images[idx],self.codes[idx]

# Neural Network: Define a simple feed-forward neural network architecture
class Network(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
           nn.Linear(hidden_size, output_size))

    def forward(self, x):
        return self.model(x)
    
# Training function: Implements model training loop with validation
def train_and_evaluate(model, train_loader, val_loader, criterion, optimizer, device, epochs):

    train_losses = []
    val_losses = []
    train_accuracies = []
    val_accuracies = []

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

        print(f'Epoch [{epoch+1}/{epochs}], '
              f'Train Loss: {(train_loss/len(train_loader)):.4f}, '
              f'Train Accuracy: {(100 * train_correct / train_total):.2f}%, '
              f'Val Loss: {(val_loss/len(val_loader)):.4f}, '
              f'Val Accuracy: {(100 * val_correct / val_total):.2f}%')
     
    return train_losses, val_losses, train_accuracies, val_accuracies

# Visualization function: Creates plots to track loss and accuracy during training
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
    plt.show()

# Testing function: Evaluates model performance on test data
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

# Device setup: Configure GPU/CPU usage for training
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Dataset creation: Create PyTorch Dataset from processed images and labels
dataset = Dataset(df['image'].values, df['code'].values)

# Data splitting: Divide dataset into training (70%), validation (10%), and test (20%) sets
train_dataset, val_dataset, test_dataset = random_split(dataset, [0.70, 0.10, 0.20])    # modifiable to evaluate performance
                                                                                        
train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True)                   
val_loader = DataLoader(val_dataset, batch_size=256, shuffle=False)                      
test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False)                    

# Model initialization: Set up neural network with appropriate dimensions
model = Network(
    input_size = dataset.images[0].numel(),
    hidden_size = dataset.images[0].numel()//2,
    output_size = len(df['code'].unique())).to(device)

# Loss and optimizer: Define cross-entropy loss and Adam optimizer
criterion = nn.CrossEntropyLoss().to(device)
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training configuration: Set number of training epochs
epochs = 200

# Model training: Train model and collect performance metrics
train_losses, val_losses, train_accuracies, val_accuracies = train_and_evaluate(
    model, train_loader, val_loader, criterion, optimizer, device, epochs)

# Results visualization: Plot training and validation curves
plot_results(train_losses, val_losses, train_accuracies, val_accuracies)
    
# Model evaluation: Test model on holdout test set
test_accuracy, all_predictions, all_true_labels = test_model(model, test_loader, device)

# Results reporting: Display final test accuracy and classification metrics  
print("--- Test Results ---")
print(f"Test Accuracy: {test_accuracy:.2f}%")

# Detailed final classification report
print("\nClassification Details:")
print(classification_report(
    all_true_labels,
    all_predictions,
    zero_division=0
))

# Confusion matrix: Create and visualize confusion matrix of predictions
conf_matrix = confusion_matrix(all_true_labels, all_predictions)

plt.figure(figsize=(10,7))
sns.heatmap(
    conf_matrix,
    annot=True,
    fmt='d',
    cmap='Blues',
    xticklabels=range(len(np.unique(all_true_labels))),
    yticklabels=range(len(np.unique(all_true_labels)))
)
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.tight_layout()
plt.show()
