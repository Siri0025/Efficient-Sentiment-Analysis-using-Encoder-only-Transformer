import argparse
import numpy as np
import os
import torch
from torch import nn
from torch.utils.data import DataLoader
from sklearn.metrics import precision_score, recall_score, f1_score

parser = argparse.ArgumentParser(description='Sentiment Analysis Classification')
parser.add_argument('--max_seq', type=int, default=500, help='Max Seq')
parser.add_argument('--data_dir', type=str, default='/home/careinfolab/Rohan/rohanj/Transformers/Group7Proj/')
parser.add_argument('--save_ckpt_dir', type=str, default='checkpoints')
parser.add_argument('--report_interval', type=int, default=5)
parser.add_argument('--data_parallel', action='store_true', help='use data parallel?')
opt = parser.parse_args()
print(opt)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if not os.path.exists(opt.save_ckpt_dir):
    os.makedirs(opt.save_ckpt_dir)

class SentimentDataset:
    def __init__(self, features, labels):
        self.data = []
        feature_size = features.shape[1] - 1 

        for feature, label in zip(features, labels):
            feature = torch.from_numpy(feature).float()
            
            seq_len = int(feature[-1])  
            feature = feature[:-1]  
            #feature = feature.view(-1, feature_size) 
            
            #padded_feature = torch.zeros((opt.max_seq, feature_size))

            #padded_feature[:seq_len] = feature[:seq_len]
            padded_feature = torch.stack(opt.max_seq * [feature])
            padded_label = torch.ones((opt.max_seq, 1))
            padded_label = padded_label * label


            mask = [1.] * seq_len + [0.] * (opt.max_seq - seq_len)
            tensor_mask = torch.tensor(mask).view(opt.max_seq, 1)


            self.data.append([
                padded_feature.to(device),
                padded_label.to(device),
                tensor_mask.to(device)
            ])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

def evaluate(model, loader):
    model.eval()
    with torch.no_grad():
        total, correct = 0, 0
        total_loss = 0.0
        criterion = nn.CrossEntropyLoss(reduction='none')  # Element-wise loss calculation

        all_labels = []
        all_predictions = []
        for features, labels, mask in loader:
            features, labels, mask = features.to(device), labels.to(device), mask.to(device)
            outputs = model(features,mask) 
             # Shape: [batch_size, seq_len, num_classes]
            # Flatten for CrossEntropyLoss
            outputs = outputs.view(-1, outputs.size(-1))  # [batch_size * seq_len, num_classes]
            labels = labels.view(-1).long()  # [batch_size * seq_len]
            mask = mask.view(-1)  # Flatten mask to apply on the loss

            # Compute cross-entropy loss with masking
            loss = (criterion(outputs, labels) * mask).sum() / mask.sum()
            total_loss += loss.item()

            # Get predicted labels
            predicted = torch.argmax(outputs, dim=1)
            
            # Log predictions for multiple entries
            print(f"Predicted (first 5): {predicted[:5].tolist()}, Labels (first 5): {labels[:5].tolist()}")


            # Mask out padded positions in predictions and labels
            masked_correct = ((predicted == labels) & (mask > 0.5)).float()
            correct += masked_correct.sum().item()
            total += mask.sum().item()

            # Collect all true and predicted labels for masked (non-padded) positions
            all_labels.extend(labels[mask > 0.5].cpu().numpy())
            all_predictions.extend(predicted[mask > 0.5].cpu().numpy())

    # Compute metrics
    accuracy = correct / total if total > 0 else 0
    precision = precision_score(all_labels, all_predictions, average='macro', zero_division=0)
    recall = recall_score(all_labels, all_predictions, average='macro', zero_division=0)
    f1 = f1_score(all_labels, all_predictions, average='macro', zero_division=0)
    avg_loss = total_loss / len(loader) if len(loader) > 0 else 0

    return accuracy, precision, recall, f1, avg_loss


def checkpoint(model):
    model_out_path = os.path.join(opt.save_ckpt_dir, 'best_model.pth')
    torch.save(model.state_dict(), model_out_path)
    print("Checkpoint saved to {}".format(model_out_path))

def train(features, labels, model,lr,batch_size,n_epochs):
    train_loader = DataLoader(SentimentDataset(features[0], labels[0]), batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(SentimentDataset(features[1], labels[1]), batch_size=1, shuffle=True)
    val_loader = DataLoader(SentimentDataset(features[2], labels[2]), batch_size=1, shuffle=True)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    best_accuracy = 0
    criterion = nn.CrossEntropyLoss(reduction='none')  # Element-wise loss calculation

    for epoch in range(n_epochs):
        print('Epoch', epoch)
        model.train()
        total_loss = 0.0

        for features, labels, mask in train_loader:
            optimizer.zero_grad()
            features, labels, mask = features.to(device), labels.to(device).long(), mask.to(device)

            outputs = model(features,mask)
            #print(torch.argmax(outputs,dim=2))
            #predicted = torch.argmax(outputs, dim=1)
            #print(predicted)  # Expected shape: [batch_size, max_seq_len, num_classes]
            labels = labels.view(-1)  # Flatten labels to match [batch_size * max_seq_len]
            outputs = outputs.view(-1, outputs.size(-1))  # Flatten to [batch_size * max_seq_len, num_classes]
            
            # Compute loss
            loss = criterion(outputs, labels)

            # Apply mask to loss and compute mean
            mask = mask.view(-1)  # Flatten mask to [batch_size * max_seq_len]
            masked_loss = loss * mask  # Apply mask to loss
            mean_loss = masked_loss.sum() / mask.sum()  # Average over non-masked elements

            # Backward pass and optimization
            mean_loss.backward()
            optimizer.step()

            total_loss += mean_loss.item()


        print(f'Train Loss: {total_loss:.4f}')

        # Evaluate model and save if it has improved
        if epoch > 0 and epoch % opt.report_interval == 0:
            print('Testing')
            val_accuracy,val_precision,val_recall,val_f1,val_loss = evaluate(model, val_loader)
            test_accuracy,test_precision,test_recall,test_f1,test_loss = evaluate(model, test_loader)

            if val_accuracy > best_accuracy:
                best_accuracy = val_accuracy
                checkpoint(model)
            
            print(f'Test Loss: {test_loss:.4f}')
            print(f'Validate Loss: {val_loss:.4f}')
            print(f'Accuracy: {test_accuracy:.4f}')
            print(f'Val accuracy: {val_accuracy:.4f}')
            print(f'Test_Precision: {test_precision:.4f}')
            print(f'Val_Precision: {val_precision:.4f}')
            print(f'test_Recall: {test_recall:.4f}')
            print(f'Val_Recall: {val_recall:.4f}')
            print(f'Test_F1 Score: {test_f1:.4f}')
            print(f'Val_F1 Score: {val_f1:.4f}')
            print(f"Best Accuracy: {best_accuracy:.4f}")
    return best_accuracy,test_loss,test_precision,test_recall,test_f1,val_loss,val_precision,val_recall,val_f1,test_accuracy,val_accuracy
