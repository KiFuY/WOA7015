import os
import torch
from torchvision import transforms, datasets
from sklearn.metrics import accuracy_score, recall_score, f1_score
from torch import nn

def create_model(model_name, num_classes):
    if model_name == "CNN":
        model = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Flatten(),
            nn.Linear(32 * 56 * 56, num_classes)
        )
    elif model_name == "ResNet-50":
        from torchvision.models import resnet50
        model = resnet50(weights=None)
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, num_classes)
    return model

compare_folder = input("Enter compare folder name (e.g., compare_1): ").strip()
model_name = input("Enter model name (CNN or ResNet-50): ").strip()

# Setting model path
results_dir = os.path.join("results", compare_folder)
model_path = os.path.join(results_dir, model_name, "best_model.pt")

if not os.path.exists(model_path):
    raise FileNotFoundError(f"Model file not found at: {model_path}")

# Load model
num_classes = 2
model = create_model(model_name, num_classes)
state_dict = torch.load(model_path, map_location=torch.device('cpu'))
model.load_state_dict(state_dict)

device = "cuda" if torch.cuda.is_available() else "cpu"
model = model.to(device)
model.eval()

# Data preprocessing
data_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Loading datasets
test_dir = os.path.join("chest_xray", "test")
test_dataset = datasets.ImageFolder(test_dir, transform=data_transforms)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=32, shuffle=False)
print(f"Class names: {test_dataset.classes}")

def evaluate_model():
    all_preds = []
    all_labels = []

    # Evaluate model performance
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    accuracy = accuracy_score(all_labels, all_preds)
    recall = recall_score(all_labels, all_preds, average='macro')  
    f1 = f1_score(all_labels, all_preds, average='macro')  

    print(f"Accuracy: {accuracy * 100:.2f}%")
    print(f"Recall: {recall * 100:.2f}%")
    print(f"F1 Score: {f1 * 100:.2f}%")

    # Save results
    metrics_path = os.path.join(results_dir, model_name, "evaluation_metrics.txt")
    os.makedirs(os.path.dirname(metrics_path), exist_ok=True)
    with open(metrics_path, "w") as f:
        f.write("Evaluation Metrics\n")
        f.write("==================\n")
        f.write(f"Accuracy: {accuracy * 100:.2f}%\n")
        f.write(f"Recall: {recall * 100:.2f}%\n")
        f.write(f"F1 Score: {f1 * 100:.2f}%\n")

    print(f"Evaluation metrics saved to: {metrics_path}")

evaluate_model()




