import os
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import matplotlib.pyplot as plt
from torchvision import models, transforms
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import accuracy_score
from PIL import Image

class Config:
    def __init__(self):
        self.BATCH_SIZE = 16
        self.IMAGE_SIZE = 125
        self.EPOCHS = 65
        self.NUM_CLASSES = 10
        self.DATASET_PATH = "/kaggle/input/skin-diseases-for-10-case/IMG_CLASSES"
        self.DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.LEARNING_RATE = 0.001
        self.CATEGORIES = [
            'Atopic Dermatitis', 'Basal Cell Carcinoma', 'Benign Keratosis-like Lesions (BKL)', 
            'Eczema', 'Melanocytic Nevi', 'Melanoma', 'Psoriasis and Lichen Planus', 
            'Seborrheic Keratoses', 'Tinea Ringworm Candidiasis', 'Warts Molluscum'
        ]
        self.TRAIN_CSV_PATH = "/kaggle/input/skin-diseases-for-10-case/labels.csv"


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


class DataPreprocessing:
    def __init__(self, config):
        self.config = config
        self.transform = transforms.Compose([
            transforms.Resize((self.config.IMAGE_SIZE, self.config.IMAGE_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
    def load_and_clean_data(self):
        df = pd.read_csv(self.config.TRAIN_CSV_PATH)
        df['class_name'] = df['class_name'].map({
            "Psoriasis pictures Lichen Planus and related diseases": "Psoriasis and Lichen Planus",
            "Warts Molluscum and other Viral Infections": "Warts Molluscum",
            "Tinea Ringworm Candidiasis and other Fungal Infections": "Tinea Ringworm Candidiasis",
            "Seborrheic Keratoses and other Benign Tumors": "Seborrheic Keratoses"
        })
        return df

    def split_data(self, df):
        train_df, test_df = train_test_split(df, test_size=0.1, random_state=73)
        train_df, val_df = train_test_split(train_df, test_size=0.1, random_state=73)
        return train_df, val_df, test_df

# ********************* Model 1 ********************* #

class EfficientNetModel(nn.Module):
    def __init__(self, num_classes):
        super(EfficientNetModel, self).__init__()
        self.base_model = models.efficientnet_v2_s(pretrained=True)
        for param in self.base_model.parameters():
            param.requires_grad = False
        self.adaptive_pool = nn.AdaptiveAvgPool2d(1)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(1280, 256)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(256, 128)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(128, num_classes)
    
    def forward(self, x):
        x = self.base_model(x)
        x = self.adaptive_pool(x)
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.fc2(x)
        x = self.relu2(x)
        x = self.fc3(x)
        return x

# ********************* Model 2 ********************* #

class ResNetModel(nn.Module):
    def __init__(self, num_classes, freeze_base=True, train_fc_layers=True):
        super(ResNetModel, self).__init__()
        
        self.base_model = models.resnet50(pretrained=True)
        if freeze_base:
            for param in self.base_model.parameters():
                param.requires_grad = False
        
        in_features = self.base_model.fc.in_features
        self.base_model.fc = nn.Sequential(
            nn.Linear(in_features, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )
        
        if not train_fc_layers:
            for param in self.base_model.fc.parameters():
                param.requires_grad = False

    def forward(self, x):
        return self.base_model(x)

# ********************* Model 3 ********************* #

class VGG16Model(nn.Module):
    def __init__(self, num_classes, freeze_base=True, train_fc_layers=True):
        super(VGG16Model, self).__init__()
        
        self.base_model = models.vgg16(pretrained=True)
        
        if freeze_base:
            for param in self.base_model.features.parameters():
                param.requires_grad = False
        
        in_features = self.base_model.classifier[6].in_features
        self.base_model.classifier[6] = nn.Linear(in_features, num_classes)
        
        if not train_fc_layers:
            for param in self.base_model.classifier.parameters():
                param.requires_grad = False

    def forward(self, x):
        return self.base_model(x)



class TrainingAndEvaluation:
    def __init__(self, model, config):
        self.model = model
        self.config = config
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.config.LEARNING_RATE)
        
    def train(self, train_loader, val_loader):
        train_losses, val_accuracies = [], []
        for epoch in range(self.config.EPOCHS):
            self.model.train()
            running_loss = 0.0
            for inputs, labels in train_loader:
                inputs, labels = inputs.to(self.config.DEVICE), labels.to(self.config.DEVICE)
                self.optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()
                running_loss += loss.item()
            
            avg_train_loss = running_loss / len(train_loader)
            train_losses.append(avg_train_loss)
            
            val_accuracy = self.evaluate(val_loader)
            val_accuracies.append(val_accuracy)
            
            print(f"Epoch [{epoch+1}/{self.config.EPOCHS}], Loss: {avg_train_loss:.4f}, Validation Accuracy: {val_accuracy:.2f}%")
        
        return train_losses, val_accuracies

    def evaluate(self, loader):
        self.model.eval()
        all_labels, all_preds = [], []
        with torch.no_grad():
            for inputs, labels in loader:
                inputs, labels = inputs.to(self.config.DEVICE), labels.to(self.config.DEVICE)
                outputs = self.model(inputs)
                _, predicted = torch.max(outputs, 1)
                all_labels.extend(labels.cpu().numpy())
                all_preds.extend(predicted.cpu().numpy())
        accuracy = accuracy_score(all_labels, all_preds) * 100
        return accuracy


class CrossValidation:
    def __init__(self, config, data_processing):
        self.config = config
        self.data_processing = data_processing

    def k_fold_cross_validation(self, df, k=5):
        skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=73)
        fold_accuracies = []
        
        for fold, (train_idx, val_idx) in enumerate(skf.split(df, df['class_name'])):
            print(f"\nFold {fold + 1}/{k}")
            train_df = df.iloc[train_idx]
            val_df = df.iloc[val_idx]
            
            train_dataset = CustomDataset(train_df, self.config.DATASET_PATH, self.data_processing.transform)
            val_dataset = CustomDataset(val_df, self.config.DATASET_PATH, self.data_processing.transform)
            
            train_loader = DataLoader(train_dataset, batch_size=self.config.BATCH_SIZE, shuffle=True)
            val_loader = DataLoader(val_dataset, batch_size=self.config.BATCH_SIZE, shuffle=False)
            
            models = [
                ("EfficientNet", EfficientNetModel(self.config.NUM_CLASSES).to(self.config.DEVICE)),
                ("ResNet50", ResNetModel(self.config.NUM_CLASSES).to(self.config.DEVICE)),
                ("VGG16", VGG16Model(self.config.NUM_CLASSES).to(self.config.DEVICE))
            ]
            
            for model_name, model in models:
                print(f"\nTraining {model_name}")
                training = TrainingAndEvaluation(model, self.config)
                train_losses, val_accuracies = training.train(train_loader, val_loader)
                
                fold_accuracies.append(val_accuracies[-1])  # Accuracy of last epoch
                self.plot_results(train_losses, val_accuracies, model_name)
        
        avg_accuracy = sum(fold_accuracies) / (k * 3)  # Average over all models and folds
        print(f"\nAverage Accuracy over {k}-Fold Cross Validation for all models: {avg_accuracy:.2f}%")
        return avg_accuracy

    def plot_results(self, train_losses, val_accuracies, model_name):
        plt.figure(figsize=(12, 5))
        
        plt.subplot(1, 2, 1)
        plt.plot(train_losses, label=f"{model_name} Training Loss")
        plt.title(f"{model_name} Training Loss vs Epochs")
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        
        plt.subplot(1, 2, 2)
        plt.plot(val_accuracies, label=f"{model_name} Validation Accuracy", color='orange')
        plt.title(f"{model_name} Validation Accuracy vs Epochs")
        plt.xlabel("Epochs")
        plt.ylabel("Accuracy (%)")
        
        plt.tight_layout()
        plt.show()


def main():
    config = Config()
    data_processing = DataPreprocessing(config)
    df = data_processing.load_and_clean_data()
    
    cross_validation = CrossValidation(config, data_processing)
    cross_validation.k_fold_cross_validation(df)

if __name__ == "__main__":
    main()