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

# Experiment_All, Generated_All, Generated_Part
# Original data Ori_Generated, Ori_Experiment
image_type = "Generated_Part"
data_dir = f"./Lan_Data/Pose_Model/{image_type}_separate/Train"

model_dict = {
    'CNN3': CNN3,
    # # 'VGG': VGG,
    # 'Resnet18': Resnet18,
    # # 'Resnet50': Resnet50,
    # 'VisionTransformer': VisionTransformer
}

num_epochs = 10 #30

pitch_poses = []
roll_poses = []

for image_name in os.listdir(data_dir):
    if image_name == '.ipynb_checkpoints':
        continue
    parts = image_name.split("_")
    pitch = int(parts[0][1:])  # Extract number after 'P'
    roll = int(parts[1][1:])   # Extract number after 'R'
    # print(pitch, roll)
    pitch_poses.append(pitch)
    roll_poses.append(roll)

unique_pitches = list(set(pitch_poses))
unique_rolls = list(set(roll_poses))
unique_pitches.sort()
unique_rolls.sort()

pitch_to_class = {pitch_angle: idx for idx, pitch_angle in enumerate(unique_pitches)}
roll_to_class = {roll_angle: idx for idx, roll_angle in enumerate(unique_rolls)}

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

pitch_num_classes = len(pitch_to_class)
roll_num_classes = len(roll_to_class)

transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize images for ResNet
    transforms.ToTensor(),
    # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# Experiment Data
print("**************************************************************************")
print(f"Training Data Type: {image_type}")

checkpoint_dir = f'./Lan_checkpoints_pose/{image_type}   '
os.makedirs(checkpoint_dir, exist_ok=True)

# Initialize data loaders Lan_GAN/Lan_Data/Pose_Model/Experiment_All_separate
data_dir = f"./Lan_Data/Pose_Model/{image_type}_separate"
train_dir = os.path.join(data_dir, "Train")
# val_dir = os.path.join("./Lan_Data/Pose_Model/Experiment_All_separate", "Val")
# test_dir = os.path.join("./Lan_Data/Pose_Model/Experiment_All_separate", "Test")

val_dir = os.path.join("./Lan_Data/Pose_Model/Ori_Experiment_separate", "Val")
test_dir = os.path.join("./Lan_Data/Pose_Model/Ori_Experiment_separate", "Test")

train_dataset = PoseDataset(train_dir, transform=transform)
val_dataset = PoseDataset(val_dir, transform=transform)
test_dataset = PoseDataset(test_dir, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=16)
test_loader = DataLoader(test_dataset, batch_size=16)

results = {}

# =================== Training Phase ===================
for model_name, model_class in model_dict.items():
    print(f"\n================ Training {model_name} ================")
    model = model_class(pitch_num_classes, roll_num_classes)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    lr = 1e-4 if model_name in ["VisionTransformer"] else 1e-3

    criterion = nn.CrossEntropyLoss()
    # criterion_roll = nn.CrossEntropyLoss()
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
                
                # _, preds_pitch = torch.max(pitch_logits, 1)
                # _, preds_roll = torch.max(roll_logits, 1)
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

    # No testing after training, testing will be done later uniformly
    results[model_name] = {
        'best_epoch': best_epoch,
        'best_avg_acc': best_avg_acc,
        'best_model_path': best_model_path
    }

# =================== Unified Testing ===================
print("\n================ Test All Best Models ================")
for model_name, model_class in model_dict.items():
    print(f"\nTest Results for {model_name}:")
    model = model_class(pitch_num_classes, roll_num_classes)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    best_model_path = results[model_name]['best_model_path']
    model.load_state_dict(torch.load(best_model_path, map_location=device))
    model.eval()
    test_pitch_preds = []
    test_pitch_labels = []
    test_roll_preds = []
    test_roll_labels = []
    test_image_names = []
    with torch.no_grad():
        for i, (images, pitch_classes, roll_classes) in enumerate(test_loader):
            images = images.to(device)
            pitch_classes = pitch_classes.to(device)
            roll_classes = roll_classes.to(device)
            
            pitch_logits, roll_logits = model(images)
            
            # _, preds_pitch = torch.max(pitch_logits, 1)
            # _, preds_roll = torch.max(roll_logits, 1)
            preds_pitch = pitch_logits.argmax(1)
            preds_roll = roll_logits.argmax(1)
            
            test_pitch_preds.extend(preds_pitch.cpu().numpy())
            test_roll_preds.extend(preds_roll.cpu().numpy())
            test_pitch_labels.extend(pitch_classes.cpu().numpy())
            test_roll_labels.extend(roll_classes.cpu().numpy())
            
            batch_start = i * test_loader.batch_size
            batch_end = batch_start + images.size(0)
            
            test_image_names.extend([test_dataset.image_paths[j].split('/')[-1] for j in range(batch_start, batch_end)])

    # 1. All test
    test_pitch_accuracy = accuracy_score(test_pitch_labels, test_pitch_preds)
    test_pitch_precision = precision_score(test_pitch_labels, test_pitch_preds, average='macro', zero_division=0)
    test_pitch_recall = recall_score(test_pitch_labels, test_pitch_preds, average='macro', zero_division=0)
    test_pitch_f1 = f1_score(test_pitch_labels, test_pitch_preds, average='macro', zero_division=0)
    
    test_roll_accuracy = accuracy_score(test_roll_labels, test_roll_preds)
    test_roll_precision = precision_score(test_roll_labels, test_roll_preds, average='macro', zero_division=0)
    test_roll_recall = recall_score(test_roll_labels, test_roll_preds, average='macro', zero_division=0)
    test_roll_f1 = f1_score(test_roll_labels, test_roll_preds, average='macro', zero_division=0)
    
    avg_metrics = np.mean([
        [test_pitch_accuracy, test_pitch_precision, test_pitch_recall, test_pitch_f1],
        [test_roll_accuracy, test_roll_precision, test_roll_recall, test_roll_f1]
    ], axis=0)

    # 2. Group by specified prefixes
    group_prefixes = ["P0_R20", "P10_R30", "P20_R40", "P30_R50", "P40_R60"]
    group_idx = [any([name.startswith(prefix) for prefix in group_prefixes]) for name in test_image_names]
    group_idx = np.array(group_idx)
    def calc_group_metrics(idx_mask):
        pitch_labels = np.array(test_pitch_labels)[idx_mask]
        pitch_preds = np.array(test_pitch_preds)[idx_mask]
        roll_labels = np.array(test_roll_labels)[idx_mask]
        roll_preds = np.array(test_roll_preds)[idx_mask]
        if len(pitch_labels) == 0:
            return [0, 0, 0, 0], 0
        pitch_acc = accuracy_score(pitch_labels, pitch_preds)
        pitch_prec = precision_score(pitch_labels, pitch_preds, average='macro', zero_division=0)
        pitch_rec = recall_score(pitch_labels, pitch_preds, average='macro', zero_division=0)
        pitch_f1 = f1_score(pitch_labels, pitch_preds, average='macro', zero_division=0)
        roll_acc = accuracy_score(roll_labels, roll_preds)
        roll_prec = precision_score(roll_labels, roll_preds, average='macro', zero_division=0)
        roll_rec = recall_score(roll_labels, roll_preds, average='macro', zero_division=0)
        roll_f1 = f1_score(roll_labels, roll_preds, average='macro', zero_division=0)
        metrics = np.mean([[pitch_acc, pitch_prec, pitch_rec, pitch_f1], [roll_acc, roll_prec, roll_rec, roll_f1]], axis=0)
        return metrics, len(pitch_labels)
    group1_metrics, group1_count = calc_group_metrics(group_idx)
    group2_metrics, group2_count = calc_group_metrics(~group_idx)
    
    # Calculate weighted average verification
    total_samples = group1_count + group2_count
    weighted_avg_acc = (group1_metrics[0] * group1_count + group2_metrics[0] * group2_count) / total_samples if total_samples > 0 else 0

    results[model_name].update({
        'all_test_avg': avg_metrics,
        'group1_avg': group1_metrics,
        'group2_avg': group2_metrics,
        'group1_count': group1_count,
        'group2_count': group2_count,
        'weighted_avg_acc': weighted_avg_acc
    })
    print(f"All Test Avg: Acc={avg_metrics[0]:.4f}, Prec={avg_metrics[1]:.4f}, Recall={avg_metrics[2]:.4f}, F1={avg_metrics[3]:.4f}")
    print(f"Group1 (prefix) Avg: Acc={group1_metrics[0]:.4f}, Prec={group1_metrics[1]:.4f}, Recall={group1_metrics[2]:.4f}, F1={group1_metrics[3]:.4f} (Sample count: {group1_count})")
    print(f"Group2 (other) Avg: Acc={group2_metrics[0]:.4f}, Prec={group2_metrics[1]:.4f}, Recall={group2_metrics[2]:.4f}, F1={group2_metrics[3]:.4f} (Sample count: {group2_count})")
    print(f"Weighted average verification: Acc={weighted_avg_acc:.4f} (Should equal All Test Acc)")

print("\n================ Summary ================")
for model_name, res in results.items():
    print(f"{model_name}: Best Epoch={res['best_epoch']}, Best Val Avg Acc={res['best_avg_acc']:.4f}")
    print(f"  All Test Avg: Acc={res['all_test_avg'][0]:.4f}, Prec={res['all_test_avg'][1]:.4f}, Recall={res['all_test_avg'][2]:.4f}, F1={res['all_test_avg'][3]:.4f}")
    print(f"  Group1 (prefix) Avg: Acc={res['group1_avg'][0]:.4f}, Prec={res['group1_avg'][1]:.4f}, Recall={res['group1_avg'][2]:.4f}, F1={res['group1_avg'][3]:.4f} (Sample count: {res['group1_count']})")
    print(f"  Group2 (other) Avg: Acc={res['group2_avg'][0]:.4f}, Prec={res['group2_avg'][1]:.4f}, Recall={res['group2_avg'][2]:.4f}, F1={res['group2_avg'][3]:.4f} (Sample count: {res['group2_count']})")
    print(f"  Weighted average verification: Acc={res['weighted_avg_acc']:.4f} (Should equal All Test Acc)")
    print(f"  Simple average: {(res['group1_avg'][0] + res['group2_avg'][0]) / 2:.4f} (Not equal to All Test Acc)")
