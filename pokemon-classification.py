
#%%
import os
from PIL import Image, ImageFilter
import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
import torch.nn as nn
import torch.optim as optim
from torchvision import models
import matplotlib.pyplot as plt
import numpy as np

#%%
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# Define the paths to your dataset and mask folders
dataset_folder = "/home/viswam1@ds.vanderbilt.edu/Project_DS5660/dataset"  # Folder containing your dataset images (e.g., PNG files)

# Function to process images and return the image data, labels, and label-to-index mapping
def process_images_to_memory(folder_path, size=(475, 475)):
    """
    Process images, resize them, and store in memory along with labels.

    Args:
        folder_path (str): Path to the folder containing raw images.
        size (tuple): Desired size of processed images.

    Returns:
        tuple: Arrays of images, labels, and a label-to-index mapping.
    """
    images = []
    labels = []
    label_to_index = {}

    for filename in sorted(os.listdir(folder_path)):
        if filename.endswith(".png"):
            # Process image
            image_path = os.path.join(folder_path, filename)
            image = Image.open(image_path).convert("RGBA").resize(size)
            white_background = Image.new("RGBA", size, (255, 255, 255, 255))
            white_background.paste(image, (0, 0), image)

            # Convert to NumPy array and store
            images.append(np.array(white_background))

            # Remove the '.png' extension properly
            name = filename[:-4].lower()  # Remove the last 4 characters (i.e., ".png")
            if name not in label_to_index:
                label_to_index[name] = len(label_to_index)  # Assign a unique index
            labels.append(name)

    # Convert lists to NumPy arrays
    images = np.array(images, dtype=np.uint8)
    labels = np.array(labels, dtype="<U50")  # Using Unicode string array for labels
    
    print(f"Images shape: {images.shape}, Labels shape: {labels.shape}")
    return images, labels, label_to_index

images, labels, label_to_index = process_images_to_memory(dataset_folder)
images = images[:,:,:,0:3]  # Keep only the RGB channels

# Define the custom dataset class
class PokemonDataset(Dataset):
    def __init__(self, images, labels, label_to_index, index_to_label, transform=None, augment=False, test_mode=False):
        self.transform = transform
        self.augment = augment
        self.label_to_index = label_to_index
        self.index_to_label = index_to_label  # Map index to label
        self.filenames = labels  # Use labels as filenames (could be different depending on implementation)

        # Split into training and test sets (80/20)
        split_idx = int(len(images) * 0.8)
        if test_mode:
            self.images = images[split_idx:]
            self.labels = labels[split_idx:]
        else:
            self.images = images[:split_idx]
            self.labels = labels[:split_idx]

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]

        # Convert label (string) to index using label_to_index
        label = self.label_to_index[label]  # Map label to index
        label = torch.tensor(label)  # Convert label to tensor (integer)

        # Apply augmentation if specified
        if self.augment:
            image = Image.fromarray(image).filter(ImageFilter.GaussianBlur(radius=2))
        else:
            image = Image.fromarray(image)

        # Apply transformation (this includes ToTensor)
        if self.transform:
            image = self.transform(image)

        return image, label

# Define transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize for compatibility with pre-trained models
    transforms.ToTensor()  # Convert image to tensor
])

# Create the train and test datasets
index_to_label = {v: k for k, v in label_to_index.items()}  # Reverse the label-to-index mapping
train_dataset = PokemonDataset(images, labels, label_to_index, index_to_label, transform=transform, augment=True, test_mode=False)
test_dataset = PokemonDataset(images, labels, label_to_index, index_to_label, transform=transform, test_mode=True)

# Create DataLoader for both train and test datasets
train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False)

print(f"Training dataset size: {len(train_dataset)}, Test dataset size: {len(test_dataset)}")

# Check the first batch of data
for data in train_loader:
    images, labels = data
    print(type(images))  # Should print <class 'torch.Tensor'>
    print(type(labels))  # Should print <class 'torch.Tensor'>
    
    # Print the index-to-label mapping for the first batch of labels
    print("Index to Label Mapping:")
    for idx, label in enumerate(labels):
        print(f"Index: {label.item()} -> Label: {index_to_label[label.item()]}")
    break  # Check the first batch

#%%
def plot_batch(data_loader, index_to_label):
    """
    Plot a batch of images with their corresponding label names.

    Args:
        data_loader (DataLoader): The data loader for the dataset.
        index_to_label (dict): A dictionary mapping label indices to class names.
    """
    # Get a batch of data
    images, labels = next(iter(data_loader))
    
    # Create a grid of images
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    axes = axes.flatten()

    for i in range(8):
        image = images[i].permute(1, 2, 0)  # Convert from CHW to HWC format for display
        axes[i].imshow(image.numpy())
        axes[i].axis('off')
        
        # Map the label index to the class name using index_to_label
        class_name = index_to_label[labels[i].item()]
        axes[i].set_title(f"Label: {class_name}")  # Display label name instead of index
        
    plt.show()

# Assuming index_to_label is available from your dataset
plot_batch(train_loader, train_dataset.index_to_label)

#%% Define the model (ResNet)
class PokemonClassifier(nn.Module):
    def __init__(self, num_classes):
        super(PokemonClassifier, self).__init__()
        self.model = models.resnet18(pretrained=True)
        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)

    def forward(self, x):
        return self.model(x)

#%% Initialize model, criterion, and optimizer
num_classes = len(train_dataset.label_to_index)
model = PokemonClassifier(num_classes=num_classes).to(device)  # Move model to device
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.00001)

#%% Training loop
num_epochs = 100
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for images, labels in train_loader:  # Unpack filenames as well
        images, labels = images.to(device), labels.to(device)  # Move images and labels to device
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {running_loss/len(train_loader)}")

#%% Evaluation on the test set
model.eval()
correct = 0
total = 0
predictions = []
actuals = []

# Disable gradient computation for evaluation
with torch.no_grad():
    for images, labels in test_loader:  # Unpack filenames as well
        images, labels = images.to(device), labels.to(device)  # Move images and labels to device
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        
        # Save predictions and actuals for plotting
        predictions.extend(predicted.cpu().numpy())
        actuals.extend(labels.cpu().numpy())

print(f'Accuracy on test set: {100 * correct / total}%')

#%% Plotting results: Display a batch of test images with predicted and actual labels
import matplotlib.pyplot as plt

# Take a batch of test images for visualization
images, labels = next(iter(test_loader))  # Unpack filenames as well
images, labels = images.to(device), labels.to(device)  # Move to device
outputs = model(images)
_, predicted = torch.max(outputs, 1)

# Move tensors to CPU for plotting
images = images.cpu()
predicted = predicted.cpu()
labels = labels.cpu()

# Display images with predicted and actual labels
fig, axes = plt.subplots(1, 8, figsize=(20, 5))
for i, ax in enumerate(axes):
    img = images[i].permute(1, 2, 0).numpy()  # Convert to (H, W, C) for matplotlib
    ax.imshow(img)
    ax.set_title(f"P: {test_dataset.index_to_label[predicted[i].item()]}\nA: {test_dataset.index_to_label[labels[i].item()]}")
    ax.axis('off')
plt.show()

# Epoch 1/10, Loss: 4.298060512542724
# Epoch 2/10, Loss: 2.6347580035527547
# Epoch 3/10, Loss: 1.7910629947980246
# Epoch 4/10, Loss: 1.2924952149391173
# Epoch 5/10, Loss: 0.9369195600350698
# Epoch 6/10, Loss: 0.7014775663614273
# Epoch 7/10, Loss: 0.6871185819307963
# Epoch 8/10, Loss: 0.5396935532490412
# Epoch 9/10, Loss: 0.36071471323569615
# Epoch 10/10, Loss: 0.33459382504224777
# Accuracy on test set: 95.0%

# %%
