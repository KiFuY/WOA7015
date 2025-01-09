import os
import random
import torch
from torchvision import transforms, datasets
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.widgets import Button
import matplotlib as mpl
from torch import nn

# Define the CNN model
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
    else:
        raise ValueError("Invalid model name! Please enter 'CNN' or 'ResNet-50'.")
    return model

compare_folder = input("Enter compare folder name (e.g., compare_1): ").strip()
model_name = input("Enter model name (CNN or ResNet-50): ").strip()

if model_name not in ["CNN", "ResNet-50"]:
    raise ValueError("Invalid model name! Please enter 'CNN' or 'ResNet-50'.")

results_dir = os.path.join("results", compare_folder)
model_path = os.path.join(results_dir, model_name, "best_model.pt")

if not os.path.exists(model_path):
    raise FileNotFoundError(f"Model file not found at: {model_path}")

# Load model
num_classes = 2
model = create_model(model_name, num_classes)
model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))

device = "cuda" if torch.cuda.is_available() else "cpu"
model = model.to(device)
model.eval()

# Data preprocessing
data_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

test_dir = "chest_xray/test"
all_image_paths = []
for root, _, files in os.walk(test_dir):
    for file in files:
        if file.endswith(('.jpeg', '.jpg', '.png')):
            all_image_paths.append(os.path.join(root, file))

random.shuffle(all_image_paths)
selected_image_paths = all_image_paths[:10]
print(f"Selected images: {selected_image_paths}")

# Class names
test_dataset = datasets.ImageFolder(test_dir, transform=data_transforms)
class_names = test_dataset.classes
print(f"Class names: {class_names}")

def test_single_image(image_path):
    image = Image.open(image_path).convert("RGB")
    input_tensor = data_transforms(image).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(input_tensor)
        probabilities = torch.softmax(outputs, dim=1)
        _, pred = torch.max(outputs, 1)

    predicted_class = class_names[pred.item()]
    confidence = probabilities[0][pred.item()].item() * 100

    return image, predicted_class, confidence

class ImageBrowser:
    mpl.rcParams['toolbar'] = 'none'

    def __init__(self, image_paths):
        self.image_paths = image_paths
        self.current_index = 0
        self.fig, self.ax = plt.subplots(figsize=(6, 6))
        plt.subplots_adjust(bottom=0.2)

        self.ax_prev = plt.axes([0.3, 0.05, 0.1, 0.075])
        self.ax_next = plt.axes([0.6, 0.05, 0.1, 0.075])
        self.btn_prev = Button(self.ax_prev, 'Previous')
        self.btn_next = Button(self.ax_next, 'Next')

        self.btn_prev.on_clicked(self.prev_image)
        self.btn_next.on_clicked(self.next_image)

        self.display_image()

    def display_image(self):
        image_path = self.image_paths[self.current_index]
        image, predicted_class, confidence = test_single_image(image_path)

        self.ax.clear()
        self.ax.imshow(image)
        self.ax.set_title(f"[{self.current_index + 1}/{len(self.image_paths)}] Prediction: {predicted_class}\nConfidence: {confidence:.2f}%", fontsize=14)
        self.ax.axis("off")
        self.fig.canvas.draw()

    def prev_image(self, event):
        if self.current_index > 0:
            self.current_index -= 1
        else:
            self.current_index = len(self.image_paths) - 1
        self.display_image()

    def next_image(self, event):
        if self.current_index < len(self.image_paths) - 1:
            self.current_index += 1
        else:
            self.current_index = 0
        self.display_image()

browser = ImageBrowser(selected_image_paths)
plt.show()


