import os
import numpy as np
from sklearn.model_selection import StratifiedKFold
from PIL import Image
import torch
import torch.nn as nn
from torchvision import models, transforms
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from sklearn.metrics import precision_recall_fscore_support, roc_curve, auc, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import optuna
from optuna.trial import TrialState
import cv2
from scipy import stats

# ============================ 1. Model Creation Function ============================
def model_fn():
    model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
    model.fc = nn.Linear(model.fc.in_features, 2)
    return model

# ============================ 2. Normalization Parameters ============================
ULTRASOUND_MEANS = [0.3129, 0.3086, 0.3344]
ULTRASOUND_STDS = [0.1587, 0.1568, 0.1681]

# ============================ 3. Dataset Class Definition ============================
class UltrasoundDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.classes = ['inf', 'sup']
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}

        self.images = []
        self.labels = []
        self.patient_ids = []  # Store patient ID

        for cls in self.classes:
            class_dir = os.path.join(root_dir, cls)
            if not os.path.exists(class_dir):
                continue
            for img_name in os.listdir(class_dir):
                if img_name.endswith(('.png', '.jpg', '.jpeg')):
                    self.images.append(os.path.join(class_dir, img_name))
                    self.labels.append(self.class_to_idx[cls])
                    # Extract patient ID from image name (e.g., 'patient001_image1.jpg' -> 'patient001')
                    patient_id = img_name.split('_')[0]
                    self.patient_ids.append(patient_id)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = self.images[idx]
        image = Image.open(img_path).convert('RGB')
        label = self.labels[idx]
        if self.transform:
            image = self.transform(image)
        return image, label, self.patient_ids[idx]

# ============================ 4. Evaluation Function ============================
def evaluate_model(model, loader, device):
    model.eval()
    all_preds = []
    all_labels = []
    all_probs = []
    all_patient_ids = []

    with torch.no_grad():
        for inputs, labels, patient_ids in loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            probs = torch.softmax(outputs, dim=1)
            _, preds = torch.max(outputs, 1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs[:, 1].cpu().numpy())
            all_patient_ids.extend(patient_ids)

    # Print all predictions and labels
    print(f"All predictions: {np.array(all_preds)}")
    print(f"All labels: {np.array(all_labels)}")
    print(f"Label distribution - 0s: {sum(1 for x in all_labels if x == 0)}, 1s: {sum(1 for x in all_labels if x == 1)}")

    precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average='binary')
    fpr, tpr, _ = roc_curve(all_labels, all_probs)
    roc_auc = auc(fpr, tpr)
    cm = confusion_matrix(all_labels, all_preds)

    # Balanced accuracy
    balanced_acc = (cm[0,0]/(cm[0,0]+cm[0,1]) + cm[1,1]/(cm[1,0]+cm[1,1]))/2

    # Classification report
    report = {
        'inf': {'precision': cm[0,0]/(cm[0,0]+cm[1,0]) if (cm[0,0]+cm[1,0]) > 0 else 0,
                'recall': cm[0,0]/(cm[0,0]+cm[0,1]) if (cm[0,0]+cm[0,1]) > 0 else 0,
                'f1-score': 2*cm[0,0]/(2*cm[0,0]+cm[0,1]+cm[1,0]) if (2*cm[0,0]+cm[0,1]+cm[1,0]) > 0 else 0,
                'support': cm[0,0]+cm[0,1]},
        'sup': {'precision': cm[1,1]/(cm[1,1]+cm[0,1]) if (cm[1,1]+cm[0,1]) > 0 else 0,
                'recall': cm[1,1]/(cm[1,1]+cm[1,0]) if (cm[1,1]+cm[1,0]) > 0 else 0,
                'f1-score': 2*cm[1,1]/(2*cm[1,1]+cm[0,1]+cm[1,0]) if (2*cm[1,1]+cm[0,1]+cm[1,0]) > 0 else 0,
                'support': cm[1,1]+cm[1,0]}
    }

    return precision, recall, f1, balanced_acc, roc_auc, cm, report, all_preds, all_labels, all_probs, all_patient_ids, fpr, tpr

# ============================ 5. Visualization ============================
def plot_confusion_matrix(cm, save_path):
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig(save_path)
    plt.close()

def plot_roc_curve(fpr, tpr, roc_auc, save_path):
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    plt.savefig(save_path)
    plt.close()

# ============================ 6. Data Load ============================
BASE_PATH = "./results"
DATA_PATH = "./data"
MODEL_PATH = os.path.join(BASE_PATH, "models/kfold")
RESULTS_PATH = os.path.join(BASE_PATH, "kfold_analysis")
OPTUNA_RESULTS_PATH = os.path.join(BASE_PATH, "optuna_results")

# Create required output directories
os.makedirs(MODEL_PATH, exist_ok=True)
os.makedirs(RESULTS_PATH, exist_ok=True)
os.makedirs(OPTUNA_RESULTS_PATH, exist_ok=True)

# Transform 
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=ULTRASOUND_MEANS, std=ULTRASOUND_STDS)
])

# Load and Validate Datasets
print("\n=== Start Data Load ===")
train_path = os.path.join(DATA_PATH, 'train')
test_path = os.path.join(DATA_PATH, 'test')

# Root confirm
print(f"Train 데이터 경로: {train_path}")
print(f"Test 데이터 경로: {test_path}")

if not os.path.exists(train_path):
    raise ValueError(f"Train 데이터 경로가 존재하지 않습니다: {train_path}")
if not os.path.exists(test_path):
    raise ValueError(f"Test 데이터 경로가 존재하지 않습니다: {test_path}")

# Dataset Load
train_dataset = UltrasoundDataset(train_path, transform=transform)
test_dataset = UltrasoundDataset(test_path, transform=transform)

# Display dataset sizes
print(f"\nTrain 데이터셋 크기: {len(train_dataset)}")
print(f"Test 데이터셋 크기: {len(test_dataset)}")

if len(train_dataset) == 0:
    raise ValueError("Train Dataset is empty.")
if len(test_dataset) == 0:
    raise ValueError("Test Dataset is empty.")

# Count samples per class
train_inf_count = sum(1 for _, label, _ in train_dataset if label == 0)
train_sup_count = sum(1 for _, label, _ in train_dataset if label == 1)
test_inf_count = sum(1 for _, label, _ in test_dataset if label == 0)
test_sup_count = sum(1 for _, label, _ in test_dataset if label == 1)

print("\n=== Data set Class Distribution ===")
print(f"Train - inf: {train_inf_count}, sup: {train_sup_count}")
print(f"Test - inf: {test_inf_count}, sup: {test_sup_count}")

# Device definition 
device = torch.device("cuda" if torch.cuda.is_available() else
                      "mps" if torch.backends.mps.is_available() else "cpu")
print(f"\nUsing device: {device}")

# K-Fold definition 
k = 5
skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=42)

# ============================ 7. Optuna Hyperparameter optimization ============================
def objective(trial):
    # Range of Hyperparameter
    lr = trial.suggest_float('lr', 1e-5, 1e-3, log=True)
    batch_size = trial.suggest_categorical('batch_size', [8, 16, 32])
    optimizer_name = trial.suggest_categorical('optimizer', ['AdamW', 'Adam', 'SGD'])

    # Model generation
    model = model_fn().to(device)
    criterion = nn.CrossEntropyLoss()

    # Optimizer 
    if optimizer_name == 'AdamW':
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    elif optimizer_name == 'Adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    else:  # SGD
        optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)

    # Fast evaluation with first Fold 
    train_idx, val_idx = next(skf.split(np.zeros(len(train_dataset)), [train_dataset[i][1] for i in range(len(train_dataset))]))

    train_loader = DataLoader(torch.utils.data.Subset(train_dataset, train_idx),
                            batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(torch.utils.data.Subset(train_dataset, val_idx),
                          batch_size=batch_size, shuffle=False)

    # 10 Epoch
    best_val_f1 = 0
    for epoch in range(10):
        model.train()
        for inputs, labels, _ in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

        # Validation loop
        model.eval()
        val_preds = []
        val_labels = []
        with torch.no_grad():
            for inputs, labels, _ in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                val_preds.extend(preds.cpu().numpy())
                val_labels.extend(labels.cpu().numpy())

        # F1 metrics
        _, _, f1, _ = precision_recall_fscore_support(val_labels, val_preds, average='binary')
        best_val_f1 = max(best_val_f1, f1)

    return best_val_f1

print("\n=== Optuna Start===")
study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=20)  # 20 trials

print("\n=== Hyperparameter Suggestions ===")
print(f"Best trial:")
trial = study.best_trial
print(f"  Value: {trial.value:.4f}")
print("  Params: ")
for key, value in trial.params.items():
    print(f"    {key}: {value}")

# Optuna Results 
print("\n=== Optuna 결과 시각화 ===")

# 1. Importance of parameter
plt.figure(figsize=(10, 6))
optuna.visualization.matplotlib.plot_param_importances(study)
plt.title('Hyperparameter Importance')
plt.tight_layout()
plt.savefig(os.path.join(OPTUNA_RESULTS_PATH, 'param_importance.png'))
plt.close()

# 2. Optimization Process
plt.figure(figsize=(10, 6))
optuna.visualization.matplotlib.plot_optimization_history(study)
plt.title('Optimization History')
plt.tight_layout()
plt.savefig(os.path.join(OPTUNA_RESULTS_PATH, 'optimization_history.png'))
plt.close()

# 3. Relationship between parameters
plt.figure(figsize=(12, 8))
optuna.visualization.matplotlib.plot_parallel_coordinate(study)
plt.title('Parallel Coordinate Plot')
plt.tight_layout()
plt.savefig(os.path.join(OPTUNA_RESULTS_PATH, 'parallel_coordinate.png'))
plt.close()

# 4. Distribution of parameters
plt.figure(figsize=(12, 8))
optuna.visualization.matplotlib.plot_slice(study)
plt.title('Parameter Distribution')
plt.tight_layout()
plt.savefig(os.path.join(OPTUNA_RESULTS_PATH, 'parameter_distribution.png'))
plt.close()

# Save the Results
with open(os.path.join(OPTUNA_RESULTS_PATH, 'optuna_results.txt'), 'w') as f:
    f.write("=== Optuna 최적화 결과 ===\n\n")
    f.write(f"Best trial value: {trial.value:.4f}\n\n")
    f.write("Best parameters:\n")
    for key, value in trial.params.items():
        f.write(f"  {key}: {value}\n")

    f.write("\n=== 모든 시도 결과 ===\n")
    for t in study.trials:
        f.write(f"\nTrial {t.number}:\n")
        f.write(f"  Value: {t.value:.4f}\n")
        f.write("  Params:\n")
        for key, value in t.params.items():
            f.write(f"    {key}: {value}\n")

# Save the Hyperparameters 
best_params = {
    'lr': trial.params['lr'],
    'optimizer': trial.params['optimizer'],
    'batch_size': trial.params['batch_size']
}

print(f"\n최적화된 하이퍼파라미터로 학습을 시작합니다.")
print(f"Learning rate: {best_params['lr']:.2e}")
print(f"Optimizer: {best_params['optimizer']}")
print(f"Batch size: {best_params['batch_size']}")
print(f"\nOptuna 결과가 {OPTUNA_RESULTS_PATH}에 저장되었습니다.")

# ============================ 8. K-Fold Validation ============================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
k = 5
skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=42)

fold_metrics = []
best_fold = 0
best_f1 = 0
best_model_state = None  # 베스트 모델의 상태를 저장할 변수

print("=== K-Fold 검증 시작 ===")
for fold, (train_idx, val_idx) in enumerate(skf.split(np.zeros(len(train_dataset)), [train_dataset[i][1] for i in range(len(train_dataset))]), 1):
    print(f"\n📁 Fold {fold}")

    # Data Loader
    train_loader = DataLoader(torch.utils.data.Subset(train_dataset, train_idx),
                            batch_size=best_params['batch_size'], shuffle=True)
    val_loader = DataLoader(torch.utils.data.Subset(train_dataset, val_idx),
                          batch_size=best_params['batch_size'], shuffle=False)

    # Model Generation and Learning 
    model = model_fn().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=best_params['lr'])

    # Training loop
    num_epochs = 50
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for inputs, labels, _ in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

    # Validation loop
    precision, recall, f1, balanced_acc, roc_auc, cm, report, _, _, _, _, fpr, tpr = evaluate_model(model, val_loader, device)
    fold_metrics.append({
        'fold': fold,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'balanced_accuracy': balanced_acc,
        'roc_auc': roc_auc
    })

    # Save Evaluation Results
    fold_dir = os.path.join(RESULTS_PATH, f"fold_{fold}")
    os.makedirs(fold_dir, exist_ok=True)

    # Confusion Matrix 저장
    plot_confusion_matrix(cm, os.path.join(fold_dir, "confusion_matrix.png"))

    # Best Model Updates
    if f1 > best_f1:
        best_f1 = f1
        best_fold = fold
        best_model_state = model.state_dict()
        # Save Best Performing Model
        torch.save({
            'model_state_dict': model.state_dict(),
            'fold': fold,
            'metrics': {
                'precision': precision,
                'recall': recall,
                'f1': f1,
                'balanced_accuracy': balanced_acc,
                'roc_auc': roc_auc
            },
            'hyperparameters': best_params
        }, os.path.join(MODEL_PATH, "best_model.pth"))

    print(f"Fold {fold} 성능 - Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}, ROC AUC: {roc_auc:.4f}")

print(f"\n✅ K-Fold 검증 완료! Best Fold: {best_fold}, Best F1 Score: {best_f1:.4f}")

# ============================ 9. Test Set Evaluation ============================
print("\n=== Test Set Evaluatin Start ===")

# 베스트 모델 로드
best_model = model_fn().to(device)
if best_model_state is not None:
    best_model.load_state_dict(best_model_state)
else:
    checkpoint = torch.load(os.path.join(MODEL_PATH, "best_model.pth"), map_location=device)
    best_model.load_state_dict(checkpoint['model_state_dict'])
best_model.eval()

# Test set 
test_loader = DataLoader(test_dataset, batch_size=best_params['batch_size'], shuffle=False)
precision, recall, f1, balanced_acc, roc_auc, cm, report, all_preds, all_labels, all_probs, all_patient_ids, fpr, tpr = evaluate_model(best_model, test_loader, device)

# Save Evaluation Results
test_dir = os.path.join(RESULTS_PATH, "test_evaluation")
os.makedirs(test_dir, exist_ok=True)

# Confusion Matrix 
plot_confusion_matrix(cm, os.path.join(test_dir, "confusion_matrix.png"))

# ROC Curve 
fpr, tpr, _ = roc_curve(all_labels, all_probs)
plot_roc_curve(fpr, tpr, roc_auc, os.path.join(test_dir, "roc_curve.png"))

# Details 
with open(os.path.join(test_dir, "detailed_results.txt"), "w") as f:
    f.write("=== Test Set 상세 평가 결과 ===\n")
    f.write(f"Best Fold: {best_fold}\n")
    f.write(f"Precision: {precision:.4f}\n")
    f.write(f"Recall: {recall:.4f}\n")
    f.write(f"F1 Score: {f1:.4f}\n")
    f.write(f"ROC AUC: {roc_auc:.4f}\n\n")

    f.write("=== 클래스별 예측 결과 ===\n")
    for i, (pred, true, prob, pid) in enumerate(zip(all_preds, all_labels, all_probs, all_patient_ids)):
        f.write(f"Image {i+1} (Patient: {pid}):\n")
        f.write(f"  True Label: {'inf' if true == 0 else 'sup'}\n")
        f.write(f"  Predicted: {'inf' if pred == 0 else 'sup'}\n")
        f.write(f"  Confidence: {prob:.4f}\n")
        f.write(f"  {'Correct' if pred == true else 'Incorrect'}\n\n")

# Summery
with open(os.path.join(RESULTS_PATH, "overall_summary.txt"), "w") as f:
    f.write("=== 전체 평가 요약 ===\n")
    f.write(f"Best Fold: {best_fold}\n")
    f.write(f"Best F1 Score: {best_f1:.4f}\n\n")

    f.write("=== K-Fold 성능 ===\n")
    for metric in fold_metrics:
        f.write(f"Fold {metric['fold']}:\n")
        f.write(f"  Precision: {metric['precision']:.4f}\n")
        f.write(f"  Recall: {metric['recall']:.4f}\n")
        f.write(f"  F1: {metric['f1']:.4f}\n")
        f.write(f"  Balanced Accuracy: {metric['balanced_accuracy']:.4f}\n")
        f.write(f"  ROC AUC: {metric['roc_auc']:.4f}\n\n")

    f.write("=== Test Set 성능 ===\n")
    f.write(f"Precision: {precision:.4f}\n")
    f.write(f"Recall: {recall:.4f}\n")
    f.write(f"F1 Score: {f1:.4f}\n")
    f.write(f"ROC AUC: {roc_auc:.4f}\n")

print(f"\n✅ 평가 완료! 결과가 {RESULTS_PATH}에 저장되었습니다.")
print(f"Best Fold: {best_fold}, Best F1 Score: {best_f1:.4f}")
print(f"Test Set 성능 - Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}, ROC AUC: {roc_auc:.4f}")

# ============================ 10. Results Visualization ============================
print("\n=== 결과 시각화 시작 ===")

# 1. K-Fold 성능 시각화
plt.figure(figsize=(12, 6))
metrics = ['precision', 'recall', 'f1', 'balanced_accuracy', 'roc_auc']
x = np.arange(len(fold_metrics))
width = 0.2

for i, metric in enumerate(metrics):
    values = [m[metric] for m in fold_metrics]
    plt.bar(x + i*width, values, width, label=metric.capitalize())

plt.xlabel('Fold')
plt.ylabel('Score')
plt.title('K-Fold Cross Validation Performance')
plt.xticks(x + width*1.5, [f'Fold {i+1}' for i in range(len(fold_metrics))])
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig(os.path.join(RESULTS_PATH, 'kfold_performance.png'))
plt.close()

# 2. Test Set Confusion Matrix
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title('Test Set Confusion Matrix')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.savefig(os.path.join(RESULTS_PATH, 'test_confusion_matrix.png'))
plt.close()

# 3. ROC Curve 
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Test Set ROC Curve')
plt.legend(loc="lower right")
plt.savefig(os.path.join(RESULTS_PATH, 'test_roc_curve.png'))
plt.close()

# 4. Distributions 
plt.figure(figsize=(10, 6))
for label in [0, 1]:
    class_probs = [prob for prob, true_label in zip(all_probs, all_labels) if true_label == label]
    plt.hist(class_probs, bins=20, alpha=0.5, label=f'Class {"inf" if label == 0 else "sup"}')

plt.xlabel('Predicted Probability')
plt.ylabel('Count')
plt.title('Prediction Probability Distribution by Class')
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig(os.path.join(RESULTS_PATH, 'prediction_probability_distribution.png'))
plt.close()

print(f"\n✅ 시각화 완료! 결과가 {RESULTS_PATH}에 저장되었습니다.")
# ============================ 6. Data Loading ============================
BASE_PATH = "./results"
DATA_PATH = "./data"
