import os
import torch
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
from torch import nn, optim
from torchvision.models import resnet50, ResNet50_Weights
from train_and_compare import train_model_with_metrics, compare_roc_curves

def get_unique_compare_dir(base_dir):
    counter = 1
    while os.path.exists(os.path.join(base_dir, f"compare_{counter}")):
        counter += 1
    compare_dir = os.path.join(base_dir, f"compare_{counter}")
    os.makedirs(compare_dir, exist_ok=True)
    return compare_dir

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

train_dir = "chest_xray/train"
val_dir = "chest_xray/val"

# Data augmentation
data_transforms_aug = {
    'train': transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
}

# Loading Datasets
data_dirs = {'train': train_dir, 'val': val_dir}
image_datasets_aug = {x: datasets.ImageFolder(data_dirs[x], transform=data_transforms_aug[x]) for x in data_dirs}
dataloaders_aug = {x: DataLoader(image_datasets_aug[x], batch_size=32, shuffle=True) for x in data_dirs}

class_names = image_datasets_aug['train'].classes
dataset_sizes = {x: len(image_datasets_aug[x]) for x in data_dirs}

# Define a function to create model instances
def create_model(model_name, num_classes):
    if model_name == "simple_cnn":
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
    elif model_name == "resnet50":
        model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, num_classes)
    return model.to(device)

EPOCHS = 2 
results_base_dir = "results"
compare_dir = get_unique_compare_dir(results_base_dir)

# Train CNN with data augmentation
simple_cnn_model = create_model("simple_cnn", len(class_names))
optimizer_simple_cnn = optim.AdamW(simple_cnn_model.parameters(), lr=0.001)
scheduler_simple_cnn = optim.lr_scheduler.StepLR(optimizer_simple_cnn, step_size=5, gamma=0.1)

simple_cnn_history, simple_cnn_save_dir = train_model_with_metrics(
    simple_cnn_model, nn.CrossEntropyLoss(), optimizer_simple_cnn, scheduler_simple_cnn,
    dataloaders_aug, dataset_sizes, num_epochs=EPOCHS, base_dir=compare_dir, 
    model_name="CNN", class_names=class_names, device=device
)

# Train ResNet-50 with data augmentation
resnet_model = create_model("resnet50", len(class_names))
optimizer_resnet = optim.AdamW(resnet_model.parameters(), lr=0.001)
scheduler_resnet = optim.lr_scheduler.StepLR(optimizer_resnet, step_size=5, gamma=0.1)

resnet_history, resnet_save_dir = train_model_with_metrics(
    resnet_model, nn.CrossEntropyLoss(), optimizer_resnet, scheduler_resnet,
    dataloaders_aug, dataset_sizes, num_epochs=EPOCHS, base_dir=compare_dir, 
    model_name="ResNet-50", class_names=class_names, device=device
)

# Compare ROC curves and AUC of two models
compare_roc_curves(
    history1=simple_cnn_history,
    history2=resnet_history,
    model1_name="CNN",
    model2_name="ResNet-50",
    save_dir=compare_dir
)

print(f"All results saved in: {compare_dir}")





