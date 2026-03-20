import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, models
from PIL import Image
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import numpy as np
import matplotlib.pyplot as plt
from Lan_Pose_Model import CNN3, VGG, Resnet18, Resnet50, VisionTransformer

hybrid_types = ["Hybrid_25Exp_75Gen"]#, "Hybrid_50Exp_50Gen", "Hybrid_75Exp_25Gen"]

model_dict = {
    'CNN3': CNN3,
    # 'VGG': VGG,
    # 'Resnet18': Resnet18,
    # 'Resnet50': Resnet50,
    # 'VisionTransformer': VisionTransformer
}

num_epochs = 5

pitch_to_class = {}
roll_to_class = {}

# Define a custom Dataset class
class PoseDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.transform = transform
        self.image_paths = []
        self.pitch_labels = []
        self.roll_labels = []

        # Load image paths and labels
        for image_name in os.listdir(data_dir):
            if image_name.endswith(".png"):
                self.image_paths.append(os.path.join(data_dir, image_name))

                if image_name.startswith('exp_') or image_name.startswith('gen_'):
                    image_name = image_name[4:]  
                
                parts = image_name.split("_")
                pitch = int(parts[0][1:])  # Extract number after 'P'
                roll = int(parts[1][1:])   # Extract number after 'R'
                self.pitch_labels.append(pitch)
                self.roll_labels.append(roll)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image = Image.open(image_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        pitch_class = pitch_to_class[self.pitch_labels[idx]]
        roll_class = roll_to_class[self.roll_labels[idx]]

        return image, torch.tensor(pitch_class), torch.tensor(roll_class)

def setup_class_mapping(data_dir):
    global pitch_to_class, roll_to_class
    
    pitch_poses = []
    roll_poses = []

    for image_name in os.listdir(data_dir):
        if image_name.endswith(".png"):
            if image_name.startswith('exp_') or image_name.startswith('gen_'):
                image_name = image_name[4:]  # Remove prefix
            
            parts = image_name.split("_")
            pitch = int(parts[0][1:])  # Extract number after 'P'
            roll = int(parts[1][1:])   # Extract number after 'R'
            pitch_poses.append(pitch)
            roll_poses.append(roll)

    unique_pitches = list(set(pitch_poses))
    unique_rolls = list(set(roll_poses))
    unique_pitches.sort()
    unique_rolls.sort()

    pitch_to_class = {pitch_angle: idx for idx, pitch_angle in enumerate(unique_pitches)}
    roll_to_class = {roll_angle: idx for idx, roll_angle in enumerate(unique_rolls)}
    
    return len(pitch_to_class), len(roll_to_class)

transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize images for ResNet
    transforms.ToTensor(),
    # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Initialize data loaders
data_dir = f"./Lan_Data/Pose_Model"
val_dir = os.path.join("./Lan_Data/Pose_Model/Ori_Experiment_separate", "Val")
test_dir = os.path.join("./Lan_Data/Pose_Model/Ori_Experiment_separate", "Test")

val_dataset = PoseDataset(val_dir, transform=transform)
test_dataset = PoseDataset(test_dir, transform=transform)

val_loader = DataLoader(val_dataset, batch_size=16)
test_loader = DataLoader(test_dataset, batch_size=16)

results = {}

# =================== Training Phase ===================
for hybrid_type in hybrid_types:
    print(f"\n**************************************************************************")
    print(f"Training with {hybrid_type}")
    print("**************************************************************************")
    
    # Set paths
    train_dir = os.path.join(f"./Lan_Data/Pose_Model/{hybrid_type}_separate", "Train")
    checkpoint_dir = f'./Lan_checkpoints_pose/{hybrid_type}'
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # Setup class mapping
    pitch_num_classes, roll_num_classes = setup_class_mapping(train_dir)
    print(f"Pitch classes: {pitch_num_classes}, Roll classes: {roll_num_classes}")
    
    # Create training dataset
    train_dataset = PoseDataset(train_dir, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    
    for model_name, model_class in model_dict.items():
        print(f"\n================ Training {model_name} with {hybrid_type} ================")
        model = model_class(pitch_num_classes, roll_num_classes)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = model.to(device)
        lr = 1e-4 if model_name in ["VisionTransformer"] else 1e-3

        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)

        best_avg_acc = 0.0
        best_epoch = 0
        best_model_path = os.path.join(checkpoint_dir, f'pose_best_{model_name}.pth')
  
        for epoch in range(num_epochs):
            model.train()
            running_loss = 0.0
            for images, pitch_classes, roll_classes in train_loader:
                images = images.to(device)
                pitch_classes = pitch_classes.to(device)
                roll_classes = roll_classes.to(device)

                optimizer.zero_grad()
                pitch_logits, roll_logits = model(images)
                
                loss_pitch = criterion(pitch_logits, pitch_classes)
                loss_roll = criterion(roll_logits, roll_classes)
                
                loss = loss_pitch + loss_roll
                
                loss.backward()
                optimizer.step()
                running_loss += loss.item() * images.size(0)
                
            epoch_loss = running_loss / len(train_loader.dataset)
            print(f'Epoch [{epoch+1}/{num_epochs}], Training Loss: {epoch_loss:.4f}')

            # Validation
            model.eval()
            val_running_loss = 0.0
            val_pitch_preds = []
            val_pitch_labels = []
            val_roll_preds = []
            val_roll_labels = []
            with torch.no_grad():
                for images, pitch_classes, roll_classes in val_loader:
                    images = images.to(device)
                    pitch_classes = pitch_classes.to(device)
                    roll_classes = roll_classes.to(device)
                    
                    pitch_logits, roll_logits = model(images)
                    
                    loss_pitch = criterion(pitch_logits, pitch_classes)
                    loss_roll = criterion(roll_logits, roll_classes)
                    loss = loss_pitch + loss_roll
                    val_running_loss += loss.item() * images.size(0)
                    
                    preds_pitch = pitch_logits.argmax(1)
                    preds_roll = roll_logits.argmax(1)
                    
                    val_pitch_preds.extend(preds_pitch.cpu().numpy())
                    val_roll_preds.extend(preds_roll.cpu().numpy())
                    val_pitch_labels.extend(pitch_classes.cpu().numpy())
                    val_roll_labels.extend(roll_classes.cpu().numpy())
                    
            val_epoch_loss = val_running_loss / len(val_loader.dataset)
            val_pitch_accuracy = accuracy_score(val_pitch_labels, val_pitch_preds)
            val_roll_accuracy = accuracy_score(val_roll_labels, val_roll_preds)
            
            avg_acc = (val_pitch_accuracy + val_roll_accuracy) / 2
            print(f'Validation Loss: {val_epoch_loss:.4f}, Avg Acc: {avg_acc:.4f}')
            if avg_acc > best_avg_acc:
                best_avg_acc = avg_acc
                best_epoch = epoch + 1
                # Save best model
                torch.save(model.state_dict(), best_model_path)
        print(f'Best Validation Avg Acc for {model_name}: {best_avg_acc:.4f} at epoch {best_epoch}')

        # Test phase
        print(f"\n================ Test {model_name} with {hybrid_type} ================")
        model.load_state_dict(torch.load(best_model_path, map_location=device))
        model.eval()
        test_pitch_preds = []
        test_pitch_labels = []
        test_roll_preds = []
        test_roll_labels = []
        with torch.no_grad():
            for images, pitch_classes, roll_classes in test_loader:
                images = images.to(device)
                pitch_classes = pitch_classes.to(device)
                roll_classes = roll_classes.to(device)
                
                pitch_logits, roll_logits = model(images)
                
                preds_pitch = pitch_logits.argmax(1)
                preds_roll = roll_logits.argmax(1)
                
                test_pitch_preds.extend(preds_pitch.cpu().numpy())
                test_roll_preds.extend(preds_roll.cpu().numpy())
                test_pitch_labels.extend(pitch_classes.cpu().numpy())
                test_roll_labels.extend(roll_classes.cpu().numpy())

        # Calculate test metrics
        test_pitch_accuracy = accuracy_score(test_pitch_labels, test_pitch_preds)
        test_pitch_precision = precision_score(test_pitch_labels, test_pitch_preds, average='macro', zero_division=0)
        test_pitch_recall = recall_score(test_pitch_labels, test_pitch_preds, average='macro', zero_division=0)
        test_pitch_f1 = f1_score(test_pitch_labels, test_pitch_preds, average='macro', zero_division=0)
        
        test_roll_accuracy = accuracy_score(test_roll_labels, test_roll_preds)
        test_roll_precision = precision_score(test_roll_labels, test_roll_preds, average='macro', zero_division=0)
        test_roll_recall = recall_score(test_roll_labels, test_roll_preds, average='macro', zero_division=0)
        test_roll_f1 = f1_score(test_roll_labels, test_roll_preds, average='macro', zero_division=0)

        # Store results
        key = f"{hybrid_type}_{model_name}"
        results[key] = {
            'hybrid_type': hybrid_type,
            'model_name': model_name,
            'best_epoch': best_epoch,
            'best_avg_acc': best_avg_acc,
            'pitch_accuracy': test_pitch_accuracy,
            'pitch_precision': test_pitch_precision,
            'pitch_recall': test_pitch_recall,
            'pitch_f1': test_pitch_f1,
            'roll_accuracy': test_roll_accuracy,
            'roll_precision': test_roll_precision,
            'roll_recall': test_roll_recall,
            'roll_f1': test_roll_f1
        }
        
        print(f"Pitch - Acc: {test_pitch_accuracy:.4f}, Prec: {test_pitch_precision:.4f}, Recall: {test_pitch_recall:.4f}, F1: {test_pitch_f1:.4f}")
        print(f"Roll  - Acc: {test_roll_accuracy:.4f}, Prec: {test_roll_precision:.4f}, Recall: {test_roll_recall:.4f}, F1: {test_roll_f1:.4f}")

# =================== Results Summary ===================
print("\n" + "="*80)
print("FINAL RESULTS SUMMARY")
print("="*80)

for key, res in results.items():
    print(f"\n{res['hybrid_type']} - {res['model_name']}:")
    print(f"  Best Epoch: {res['best_epoch']}, Best Val Avg Acc: {res['best_avg_acc']:.4f}")
    print(f"  Pitch - Acc: {res['pitch_accuracy']:.4f}, Prec: {res['pitch_precision']:.4f}, Recall: {res['pitch_recall']:.4f}, F1: {res['pitch_f1']:.4f}")
    print(f"  Roll  - Acc: {res['roll_accuracy']:.4f}, Prec: {res['roll_precision']:.4f}, Recall: {res['roll_recall']:.4f}, F1: {res['roll_f1']:.4f}")

# # Group results by hybrid type
# print("\n" + "="*80)
# print("RESULTS BY HYBRID TYPE")
# print("="*80)

# for hybrid_type in hybrid_types:
#     print(f"\n{hybrid_type}:")
#     for key, res in results.items():
#         if res['hybrid_type'] == hybrid_type:
#             print(f"  {res['model_name']}:")
#             print(f"    Pitch - Acc: {res['pitch_accuracy']:.4f}, F1: {res['pitch_f1']:.4f}")
#             print(f"    Roll  - Acc: {res['roll_accuracy']:.4f}, F1: {res['roll_f1']:.4f}") 