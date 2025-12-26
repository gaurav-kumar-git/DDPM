import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import os
from utils import split_data
import dataset
from guidance_sample_generator import ConditionalDDPM, NoiseScheduler, ClassifierDDPM

# Define Classifier Model
class Classifier(nn.Module):
    def __init__(self, input_dim, num_classes):
        super(Classifier, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )
    
    def forward(self, x):
        return self.model(x)

# Training function
def train_classifier(classifier, dataloader, epochs=100, lr=1e-3):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(classifier.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.5)
    
    classifier.train()
    for epoch in range(epochs):
        total_loss = 0
        correct, total = 0, 0
        for batch_X, batch_y in dataloader:
            optimizer.zero_grad()
            preds = classifier(batch_X)
            loss = criterion(preds, batch_y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            correct += (preds.argmax(dim=1) == batch_y).sum().item()
            total += batch_y.size(0)
        scheduler.step()
        if epoch % 10 == 0 or epoch == epochs - 1:
            print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss:.4f}, Training Accuracy: {correct/total:.4f}")

# Evaluation function for both classifiers
def evaluate_classifiers(trained_classifier, ddpm_classifier, generated_data, num_classes):
    trained_classifier.eval()
    per_class_correct = torch.zeros(num_classes)
    per_class_total = torch.zeros(num_classes)
    
    with torch.no_grad():
        for guidance_scale, (samples, labels) in generated_data.items():
            preds_trained = trained_classifier(samples).argmax(dim=1)
            preds_ddpm = ddpm_classifier.predict(samples)
            
            for c in range(num_classes):
                mask = labels == c
                per_class_correct[c] += (preds_trained[mask] == c).sum().item()
                per_class_total[c] += mask.sum().item()
            
            acc_trained = (preds_trained == labels).float().mean().item()
            acc_ddpm = (preds_ddpm == labels).float().mean().item()
            print(f"Guidance Scale {guidance_scale}: Trained Classifier Accuracy = {acc_trained:.4f}, DDPM Classifier Accuracy = {acc_ddpm:.4f}")
    
    avg_accuracy = (per_class_correct / per_class_total).mean().item()
    print(f"Average Accuracy over all Class Labels (Trained): {avg_accuracy:.4f}")

# Load dataset
data_X, data_y = dataset.load_dataset('moons')
num_classes = len(data_y.unique())
input_dim = data_X.shape[1]

# Load DDPM-generated samples
generated_data = {}
for guidance_scale in [0.0, 0.25, 0.5, 0.75, 1]:  
    file_path = f'exps/ddpm_conditional_2_200_0.0001_0.02_linear_moons/samples_42_3000_guidance_{guidance_scale}.pth'
    if os.path.exists(file_path):
        data = torch.load(file_path)
        generated_data[guidance_scale] = (data['samples'], data['labels'])
    else:
        print(f"Warning: {file_path} not found!")

# Combine real and generated data
all_X, all_y = [data_X], [data_y]
for _, (samples, labels) in generated_data.items():
    all_X.append(samples)
    all_y.append(labels)
all_X = torch.cat(all_X, dim=0)
all_y = torch.cat(all_y, dim=0)

# Train classifier
dataloader = DataLoader(TensorDataset(all_X, all_y), batch_size=64, shuffle=True)
classifier = Classifier(input_dim, num_classes)
train_classifier(classifier, dataloader)

# Load trained ConditionalDDPM model
conditional_ddpm = ConditionalDDPM(n_classes=num_classes, n_dim=input_dim)  # Load trained DDPM
conditional_ddpm.load_state_dict(torch.load('exps/ddpm_conditional_2_200_0.0001_0.02_linear_moons/model.pth'))
conditional_ddpm.eval()

# Initialize DDPM classifier using updated ClassifierDDPM
ddpm_classifier = ClassifierDDPM(conditional_ddpm, NoiseScheduler(beta_start=0.0001, beta_end=0.02), device='cpu')

# Evaluate classifiers
evaluate_classifiers(classifier, ddpm_classifier, generated_data, num_classes)