import os
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from torchvision import models, transforms
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from PIL import Image

#============================= Hyperparameters =================================#
HYPERPARAMS = {
    "BATCH_SIZE": 16,
    "IMAGE_SIZE": 125,
    "EPOCHS": 65,
    "NUM_CLASSES": 10,
    "DATASET_PATH": "/kaggle/input/skin-diseases-for-10-case/IMG_CLASSES",
    "DEVICE": torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    "LEARNING_RATE": 0.001
}

# Skin disease categories
CATEGORIES = [
    'Atopic Dermatitis', 'Basal Cell Carcinoma', 'Benign Keratosis-like Lesions (BKL)', 
    'Eczema', 'Melanocytic Nevi', 'Melanoma', 'Psoriasis and Lichen Planus', 
    'Seborrheic Keratoses', 'Tinea Ringworm Candidiasis', 'Warts Molluscum'
]

#============================= Dataset Class =================================#

class CustomDataset(Dataset):
    def __init__(self, dataframe, dataset_path, transform=None):
        self.dataframe = dataframe
        self.dataset_path = dataset_path
        self.transform = transform
    
    def __len__(self):
        return len(self.dataframe)
    
    def __getitem__(self, idx):
        img_path = os.path.join(self.dataset_path, self.dataframe.iloc[idx, 0])
        image = Image.open(img_path)
        label = self.dataframe.iloc[idx, 1]
        
        if self.transform:
            image = self.transform(image)
        
        return image, label

#============================= Data Preprocessing =================================#

# Data transformations (resize, normalize, etc.)
transform = transforms.Compose([
    transforms.Resize((HYPERPARAMS["IMAGE_SIZE"], HYPERPARAMS["IMAGE_SIZE"])),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Load and clean the dataset
def load_and_clean_data():
    df = pd.read_csv("/kaggle/input/skin-diseases-for-10-case/labels.csv")
    df['class_name'] = df['class_name'].map({
        "Psoriasis pictures Lichen Planus and related diseases": "Psoriasis and Lichen Planus",
        "Warts Molluscum and other Viral Infections": "Warts Molluscum",
        "Tinea Ringworm Candidiasis and other Fungal Infections": "Tinea Ringworm Candidiasis",
        "Seborrheic Keratoses and other Benign Tumors": "Seborrheic Keratoses"
    })
    return df

# Split dataset into training, validation, and test sets
def split_data(df):
    train_df, test_df = train_test_split(df, test_size=0.1, random_state=73)
    train_df, val_df = train_test_split(train_df, test_size=0.1, random_state=73)
    return train_df, val_df, test_df

#============================= Model Definition =================================#

class SkinDiseaseModel(nn.Module):
    def __init__(self, num_classes):
        super(SkinDiseaseModel, self).__init__()
        
        # Base model: EfficientNet V2 (pretrained)
        self.base_model = models.efficientnet_v2_s(pretrained=True)
        
        # Freeze all layers in the base model
        for param in self.base_model.parameters():
            param.requires_grad = False
        
        # Custom classifier layers
        self.adaptive_pool = nn.AdaptiveAvgPool2d(1)  # Adaptive pooling
        self.flatten = nn.Flatten()  # Flatten the output
        self.fc1 = nn.Linear(1280, 256)  # First fully connected layer (based on the EfficientNet V2 output size)
        self.relu1 = nn.ReLU()  # ReLU activation
        self.fc2 = nn.Linear(256, 128)  # Second fully connected layer
        self.relu2 = nn.ReLU()  # ReLU activation
        self.fc3 = nn.Linear(128, num_classes)  # Final output layer (num_classes)
    
    def forward(self, x):
        # Pass through the base model (EfficientNet V2)
        x = self.base_model(x)
        
        # Pass through custom classifier layers
        x = self.adaptive_pool(x)  # Adaptive pooling
        x = self.flatten(x)  # Flatten the tensor
        
        # Fully connected layers with activations
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.fc2(x)
        x = self.relu2(x)
        x = self.fc3(x)  # Output layer
        
        return x

#============================= Training & Evaluation =================================#

def train(model, train_loader, val_loader, criterion, optimizer, epochs, device):
    best_accuracy = 0.0
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()

        print(f"Epoch [{epoch+1}/{epochs}], Loss: {running_loss/len(train_loader)}")

        accuracy = evaluate(model, val_loader, device)
        print(f"Validation Accuracy: {accuracy}%")

        if accuracy > best_accuracy:
            best_accuracy = accuracy
            torch.save(model.state_dict(), "best_model.pth")

def evaluate(model, loader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return 100 * correct / total

#============================= Main Execution =================================#

def main():
    df = load_and_clean_data()
    train_df, val_df, test_df = split_data(df)

    # Create datasets and data loaders
    train_dataset = CustomDataset(train_df, HYPERPARAMS["DATASET_PATH"], transform)
    val_dataset = CustomDataset(val_df, HYPERPARAMS["DATASET_PATH"], transform)
    test_dataset = CustomDataset(test_df, HYPERPARAMS["DATASET_PATH"], transform)

    train_loader = DataLoader(train_dataset, batch_size=HYPERPARAMS["BATCH_SIZE"], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=HYPERPARAMS["BATCH_SIZE"], shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=HYPERPARAMS["BATCH_SIZE"], shuffle=False)

    model = SkinDiseaseModel(HYPERPARAMS["NUM_CLASSES"]).to(HYPERPARAMS["DEVICE"])
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.fc1.parameters(), lr=HYPERPARAMS["LEARNING_RATE"])

    # Train the model
    train(model, train_loader, val_loader, criterion, optimizer, HYPERPARAMS["EPOCHS"], HYPERPARAMS["DEVICE"])

    # Load best model and test it
    model.load_state_dict(torch.load("best_model.pth"))
    test_accuracy = evaluate(model, test_loader, HYPERPARAMS["DEVICE"])
    print(f"Test Accuracy: {test_accuracy}%")

    # Save the final model
    torch.save(model.state_dict(), 'model_skin_disease.pth')

    #============================= Prediction Function =================================#

    def predict(image_path, model, categories, device):
        image = Image.open(image_path)
        image = transform(image).unsqueeze(0).to(device)
        model.eval()
        with torch.no_grad():
            outputs = model(image)
            _, predicted = torch.max(outputs, 1)
        return categories[predicted.item()]

    # Example prediction
    image_path = "/kaggle/input/skin-diseases-for-10-case/IMG_CLASSES/Melanoma/ISIC_7284370.jpg"
    predicted_class = predict(image_path, model, CATEGORIES, HYPERPARAMS["DEVICE"])
    print(f"Predicted Class: {predicted_class}")

if __name__ == "__main__":
    main()