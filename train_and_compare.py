import os
import torch
from sklearn.metrics import accuracy_score, recall_score, f1_score, roc_curve, auc
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm

def get_unique_dir_name(base_dir, model_name):
    save_dir = os.path.join(base_dir, model_name)
    counter = 1
    while os.path.exists(save_dir):
        save_dir = os.path.join(base_dir, f"{model_name}_{counter}")
        counter += 1
    return save_dir


def train_model_with_metrics(model, criterion, optimizer, scheduler, dataloaders, dataset_sizes, num_epochs, base_dir, model_name, class_names, device):
    # Initializes the save directory
    save_dir = get_unique_dir_name(base_dir, model_name)
    os.makedirs(save_dir, exist_ok=True)

    # Initialize the best weights and metrics
    best_model_wts = model.state_dict()
    best_acc = 0.0

    # Initialize the result DataFrame
    results = pd.DataFrame(columns=['epoch', 'phase', 'loss', 'accuracy', 'f1', 'recall', 'roc_auc']).astype({
        'epoch': 'int64',
        'phase': 'str',
        'loss': 'float64',
        'accuracy': 'float64',
        'f1': 'float64',
        'recall': 'float64',
        'roc_auc': 'float64'
    })

    # Initialize the history
    history = {
        'train_loss': [], 'val_loss': [],
        'train_acc': [], 'val_acc': [],
        'train_f1': [], 'val_f1': [],
        'train_recall': [], 'val_recall': [],
        'train_roc_auc': [], 'val_roc_auc': [],
        'val_labels': [],  
        'val_probs': []    
    }

    for epoch in range(num_epochs):
        print(f"Epoch {epoch + 1}/{num_epochs}")
        for phase in ['train', 'val']:
            model.train() if phase == 'train' else model.eval()
            running_loss, running_corrects = 0.0, 0
            all_preds, all_labels, all_probs = [], [], []

            for inputs, labels in tqdm(dataloaders[phase], desc=f"{phase.capitalize()} Progress", leave=False):
                inputs, labels = inputs.to(device), labels.to(device)
                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    probs = torch.softmax(outputs, dim=1)[:, 1].detach().cpu().numpy() 
                    _, preds = torch.max(outputs, 1)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                all_probs.extend(probs)

            if phase == 'train':
                scheduler.step()

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = accuracy_score(all_labels, all_preds)
            epoch_f1 = f1_score(all_labels, all_preds, average='weighted')
            epoch_recall = recall_score(all_labels, all_preds, average='weighted')  

            # Compute ROC AUC for binary classification
            if len(class_names) == 2:
                fpr, tpr, _ = roc_curve(all_labels, all_probs)
                epoch_roc_auc = auc(fpr, tpr)
            else:
                epoch_roc_auc = None

            # Save the best model weights
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = model.state_dict()

            # Save to history
            history[f'{phase}_loss'].append(epoch_loss)
            history[f'{phase}_acc'].append(epoch_acc)
            history[f'{phase}_f1'].append(epoch_f1)
            history[f'{phase}_recall'].append(epoch_recall) 
            if epoch_roc_auc is not None:
                history[f'{phase}_roc_auc'].append(epoch_roc_auc)

            if phase == 'val':
                history['val_labels'] = all_labels
                history['val_probs'] = all_probs

            # Save to the result DataFrame
            new_row = pd.DataFrame([{
                'epoch': epoch + 1,
                'phase': phase,
                'loss': epoch_loss if epoch_loss is not None else 0.0,
                'accuracy': epoch_acc if epoch_acc is not None else 0.0,
                'f1': epoch_f1 if epoch_f1 is not None else 0.0,
                'recall': epoch_recall if epoch_recall is not None else 0.0,
                'roc_auc': epoch_roc_auc if epoch_roc_auc is not None else 'N/A'
            }])
            if not new_row.isna().all().all(): 
                results = pd.concat([results, new_row], ignore_index=True)
            else:
                print(f"Skipped adding an empty row: {new_row}")

        print()

    # Save the best model weights
    best_model_path = os.path.join(save_dir, "best_model.pt")
    torch.save(best_model_wts, best_model_path)

    # Save the training results to CSV
    csv_path = os.path.join(save_dir, "training_results.csv")
    results.to_csv(csv_path, index=False)
    print(f"Training results saved to: {csv_path}")

    plot_metrics(history, save_dir, model_name, class_names)

    return history, save_dir


def plot_metrics(history, save_dir, model_name, class_names):
    epochs = range(1, len(history['train_loss']) + 1)

    # Loss
    plt.figure()
    plt.plot(epochs, history['train_loss'], label=f'{model_name} Train Loss')
    plt.plot(epochs, history['val_loss'], label=f'{model_name} Validation Loss')
    plt.title(f'{model_name} Loss Over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend(loc='upper right')
    plt.savefig(os.path.join(save_dir, f'{model_name}_loss.png'))

    # Accuracy
    plt.figure()
    plt.plot(epochs, history['train_acc'], label=f'{model_name} Train Accuracy')
    plt.plot(epochs, history['val_acc'], label=f'{model_name} Validation Accuracy')
    plt.title(f'{model_name} Accuracy Over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend(loc='lower right')
    plt.savefig(os.path.join(save_dir, f'{model_name}_accuracy.png'))

    # F1 Score
    plt.figure()
    plt.plot(epochs, history['train_f1'], label=f'{model_name} Train F1 Score')
    plt.plot(epochs, history['val_f1'], label=f'{model_name} Validation F1 Score')
    plt.title(f'{model_name} F1 Score Over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('F1 Score')
    plt.legend(loc='lower right')
    plt.savefig(os.path.join(save_dir, f'{model_name}_f1_score.png'))

    # Recall
    plt.figure()
    plt.plot(epochs, history['train_recall'], label=f'{model_name} Train Recall')
    plt.plot(epochs, history['val_recall'], label=f'{model_name} Validation Recall')
    plt.title(f'{model_name} Recall Over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Recall')
    plt.legend(loc='lower right')
    plt.savefig(os.path.join(save_dir, f'{model_name}_recall.png'))

    # ROC Curve
    if 'val_labels' in history and 'val_probs' in history and len(class_names) == 2:
        fpr, tpr, _ = roc_curve(history['val_labels'], history['val_probs'])
        roc_auc = auc(fpr, tpr)

        plt.figure()
        plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {roc_auc:.4f})')
        plt.plot([0, 1], [0, 1], 'k--')
        plt.title(f'{model_name} ROC Curve')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.legend(loc='lower right')
        plt.savefig(os.path.join(save_dir, f'{model_name}_roc_curve.png'))

def compare_roc_curves(history1, history2, model1_name, model2_name, save_dir):
   # Compare ROC
    if 'val_labels' in history1 and 'val_probs' in history1 and 'val_labels' in history2 and 'val_probs' in history2:
        fpr1, tpr1, _ = roc_curve(history1['val_labels'], history1['val_probs'])
        auc1 = auc(fpr1, tpr1)
        fpr2, tpr2, _ = roc_curve(history2['val_labels'], history2['val_probs'])
        auc2 = auc(fpr2, tpr2)

        plt.figure(figsize=(8, 8))
        plt.plot(fpr1, tpr1, label=f'{model1_name} (AUC = {auc1:.2f})')
        plt.plot(fpr2, tpr2, label=f'{model2_name} (AUC = {auc2:.2f})')
        plt.plot([0, 1], [0, 1], 'k--')
        plt.title('ROC Curve Comparison')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.legend(loc='lower right')
        plt.grid()

        comparison_path = os.path.join(save_dir, 'roc_curve_comparison.png')
        plt.savefig(comparison_path)
        print(f"ROC Curve Comparison saved to: {comparison_path}")
    else:
        print("Cannot compare ROC curves. Ensure both histories have 'val_labels' and 'val_probs'.")
