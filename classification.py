#%%
import os
import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from PIL import Image, ImageFilter
import matplotlib.pyplot as plt
import math
#%%
# Function to process images and return the image data, labels, and label-to-index mapping
def process_images_to_memory(folder_path, size=(475, 475)):
    images = []
    labels = []
    label_to_index = {}
    file_names_list = []

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
            name = filename[:-4].lower()
            if name not in label_to_index:
                label_to_index[name] = len(label_to_index)  # Assign a unique index
            labels.append(name)
            file_names_list.append(name)  # Store filenames

    images = np.array(images, dtype=np.uint8)
    labels = np.array(labels, dtype="<U50")
    
    # Create index_to_label dictionary
    index_to_label = {i: file_names_list[i] for i in range(len(file_names_list))}
    print (index_to_label)
    print(label_to_index)
    print(f"Images shape: {images.shape}, Labels shape: {labels.shape}")
    return images, labels, label_to_index, index_to_label

#%%
# Define the custom dataset class

class PokemonDataset(Dataset):
    def __init__(self, images, labels, label_to_index, index_to_label, transform=None, augment=False):
        self.transform = transform
        self.augment = augment
        self.label_to_index = label_to_index
        self.index_to_label = index_to_label
        self.filenames = labels
        self.images = images
        self.labels = labels

        # Define augmentations pipeline if augment is enabled
        if self.augment:
            self.augmentation_transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),  # Random horizontal flip
            transforms.RandomVerticalFlip(),  # Random vertical flip
            transforms.ColorJitter(brightness=0.3, contrast=0.2, saturation=0.2, hue=0.1),  # Adjust color parameters
            transforms.RandomAffine(degrees=30, translate=(0.1, 0.1), scale=(0.8, 1.2)),  # Apply random affine transformations
            transforms.RandomResizedCrop(475, scale=(0.8, 1.0)),  # Resize with random cropping
            transforms.GaussianBlur(kernel_size=3)  # Apply Gaussian blur for smoothness

        ])
        else:
            self.augmentation_transform = None

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]

        # Convert label to its corresponding index number
        label_idx = self.label_to_index[label]

        # Augment the image if specified
        if self.augment:
            image = Image.fromarray(image)
            image = self.augmentation_transform(image)  # Apply augmentation transformations
        else:
            image = Image.fromarray(image)

        # Apply additional transformations (e.g., normalization, resizing)
        if self.transform:
            image = self.transform(image)

        return image, label_idx
#%%
# Define transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

# Load dataset and create train, validation, and test sets
dataset_folder = '/home/viswam1@ds.vanderbilt.edu/Project_DS5660/dataset'  # Replace with your dataset path
images, labels, label_to_index, index_to_label = process_images_to_memory(dataset_folder)
test_labels = labels
test_labels_idx = label_to_index
test_idx_labels = index_to_label

# The images array needs to be reshaped to remove the alpha channel (RGBA)
images = images[:,:,:,0:3]  # Use only RGB channels
# Calculate the number of rows and columns needed for the grid
num_images = len(images)
cols = 4  # You can adjust the number of columns you want
rows = math.ceil(num_images / cols)

plt.figure(figsize=(12, 12))
for i in range(num_images):
    ax = plt.subplot(rows, cols, i + 1)  # Create a grid based on the number of images
    image = images[i]  # Convert tensor to numpy array (C,H,W -> H,W,C)
    ax.imshow(image)
    ax.axis('off')
    ax.set_title(f"{labels[i]}:{label_to_index[labels[i]]}")

plt.tight_layout()
plt.show()

print(labels)
# Convert tensor to numpy for easier manipulation
labels_np = [label_to_index[label] for label in labels]

# Count occurrences of each class
unique, counts = np.unique(labels_np, return_counts=True)

# Plot class balance
plt.bar(unique, counts)
plt.xlabel('Class Labels')
plt.ylabel('Count')
plt.title('Class Distribution in the Dataset')
plt.show()

# Perform data augmentation (apply it to the entire dataset first)
augment_dataset = PokemonDataset(images, labels, label_to_index, index_to_label, transform=transform, augment=True)

train_loader = DataLoader(augment_dataset, batch_size=4, shuffle=True)

# Get a sample from the train loader to display
train_images_batch, train_labels_batch = next(iter(train_loader))

# Convert the image tensor back to a numpy array for display (from C x H x W to H x W x C)
image = train_images_batch[0].permute(1, 2, 0).numpy()  # Change from CHW to HWC format for display

# Since the image values are in [0, 1] after ToTensor(), we scale them to [0, 255]
image = (image * 255).astype(np.uint8)
# Get the label from the batch
label = train_labels_batch[0].item()  # Convert the tensor value to a native Python integer

# Convert label index to label name
label_name = index_to_label[label]

# Display the image
plt.imshow(image)
plt.title(f"Label: {label_name} (Index: {label})")  # Show both the label and the index
plt.axis('off')
plt.show()
#%%
# Define the model (ResNet)
# Define the model (ResNet)

class PokemonClassifier(nn.Module):
    def __init__(self, num_classes, pretrained=True):
        super(PokemonClassifier, self).__init__()
        
        # Load a pre-trained ResNet18 model or a fresh one
        self.model = models.resnet50(pretrained=True)
        
        # Modify the final fully connected layer to output the correct number of classes
        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)
        
    
    def forward(self, x):
        # Pass through the ResNet layers
        x = self.model(x)
        
        
        return x
#%%
# Initialize model, criterion, and optimizer and Training
device = torch.device("cuda:6" if torch.cuda.is_available() else "cpu")
num_classes = 40
model = PokemonClassifier(num_classes=num_classes).to(device)
criterion = nn.CrossEntropyLoss()  # Use CrossEntropyLoss for multi-class classification
optimizer = torch.optim.Adam(model.parameters(), lr=0.00001)  # Set learning rate


# Store training, validation, and testing losses for plotting
train_losses = []

# Training loop
num_epochs = 150
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for images, labels in train_loader:
        images, labels= images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    train_loss = running_loss / len(train_loader)
    train_losses.append(train_loss)


    print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}")
#%%
torch.save(model, 'model_full.pth')

#%%
# Test phase
model = torch.load("model_full.pth")
device = torch.device("cuda:6" if torch.cuda.is_available() else "cpu")

model.eval()
correct = 0
total = 0
folder_path = '/home/viswam1@ds.vanderbilt.edu/Project_DS5660/output_images'  # Replace with your dataset path
size=(475, 475)
images_test = []
for filename in sorted(os.listdir(folder_path)):
        if filename.endswith(".png"):
            # Process image
            image_path = os.path.join(folder_path, filename)
            image = Image.open(image_path).convert("RGBA").resize(size)
            image = image.resize((224,224))
            images_test.append(np.array(image))

images_test = np.array(images_test, dtype=np.uint8)

# Add this code after testing phase or during the testing loop
def plot_test_batch(images, labels, predictions, index_to_label):
    """
    This function plots a batch of test images with their true and predicted labels.
    """
    # Convert indices to labels (convert tensors to integers)
    true_labels = [index_to_label[label.item()] for label in labels]  # Use .item() to get the integer value
    predicted_labels = [index_to_label[prediction.item()] for prediction in predictions]  # Use .item() to get the integer value

    # Calculate the number of rows and columns needed for the grid
    num_images = len(images)
    cols = 3  # You can adjust the number of columns you want
    rows = math.ceil(num_images / cols)

    plt.figure(figsize=(20, rows*5))
    for i in range(num_images):
        ax = plt.subplot(rows, cols, i + 1)  # Create a grid based on the number of images
        image = images[i].cpu().numpy().transpose((1, 2, 0))  # Convert tensor to numpy array (C,H,W -> H,W,C)
        image = (image * 255).astype(np.uint8)
        ax.imshow(image)
        ax.axis('off')
        ax.set_title(f"True: {true_labels[i]}\nPred: {predicted_labels[i]}", fontsize=16)

    plt.tight_layout()
    plt.show()

# Testing loop with the batch plot
model.eval()
# Convert to PyTorch tensor, ensure it's the right shape

images_test = images_test[:,:,:,0:3]
images_test_tensor = torch.tensor(images_test).permute(0, 3, 1, 2).float()/255
images_test_batch = images_test_tensor.to(device)  # Move to device
labels_test =  np.arange(0,40,1)
# Get predictions
with torch.no_grad():
    outputs = model(images_test_batch)
    labels_test = torch.tensor(labels_test).to(device)
    _, predictions = torch.max(outputs, 1)
    total += labels_test.size(0)
    correct += (predictions == labels_test).sum().item()
    print(predictions)

test_accuracy = 100 * correct / total
print(f"Test Accuracy: {test_accuracy:.2f}%")

# Plot the test batch with true and predicted labels
plot_test_batch(images_test_batch, labels_test, predictions, test_idx_labels)
  # %%
