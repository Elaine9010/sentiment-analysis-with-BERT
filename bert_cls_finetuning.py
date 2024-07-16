import torch
from torch.utils.data import DataLoader, Dataset
from torch import nn, optim
from transformers import BertTokenizer, BertModel
from sklearn.metrics import accuracy_score
import pandas as pd
import wandb
import argparse
import os


# Tokenize the dataset
def tokenize_data(df):
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    return tokenizer(df['text'].tolist(), padding=True, truncation=True, max_length=512, return_tensors="pt")


# Define a custom dataset class for PyTorch
class IMDbDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {key: val[idx] for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item


# Define the model with a classification head
class SentimentClassifier(nn.Module):
    def __init__(self):
        super(SentimentClassifier, self).__init__()
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        for param in self.bert.parameters():
            param.requires_grad = False
        self.classifier = nn.Linear(self.bert.config.hidden_size, 2)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs[1]  # pooled_output is the last layer hidden-state
        return self.classifier(pooled_output)
    
# Initialize or load the model
def load_model(model_path):
    model = SentimentClassifier()
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path))
        print(f"Loaded model from {model_path}")
    else:
        print("No saved model found, initializing new model.")
    return model

# Define the training and evaluation functions
def train(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    for batch in loader:
        optimizer.zero_grad()
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        outputs = model(input_ids, attention_mask)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        wandb.log({"batch_training_loss": loss.item()})
    return total_loss / len(loader)

def evaluate(model, loader, criterion, device):
    model.eval()
    predictions, true_labels = [], []
    with torch.no_grad():
        for batch in loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            outputs = model(input_ids, attention_mask)
            loss = criterion(outputs, labels)
            wandb.log({"batch_eval_loss": loss.item()})
            preds = torch.argmax(outputs, dim=1).cpu().numpy()
            predictions.extend(preds)
            true_labels.extend(labels.cpu().numpy())
    return accuracy_score(true_labels, predictions)

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--learning_rate', type=float, default=5e-5)
    parser.add_argument('--epochs', type=int, default=3)
    parser.add_argument('--load_saved_model',type=bool,default=False)
    parser.add_argument('--best_model_path',type=str,default='model.pth')

    args = parser.parse_args()
    batch_size = args.batch_size
    learning_rate = args.learning_rate
    epochs = args.epochs
    load_saved_model = args.load_saved_model
    best_model_path = args.best_model_path

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    wandb.login()

    # Initialize W&B run  
    run = wandb.init(
        project="bert-finetuning-demo",
        name=f'batch_size_{batch_size}_lr_{learning_rate}',
        config={
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "epochs": epochs,
        },
        reinit=True
    )

    # Define file paths for the dataset splits
    splits = {
        'train': 'plain_text/train-00000-of-00001.parquet',
        'test': 'plain_text/test-00000-of-00001.parquet',
        'unsupervised': 'plain_text/unsupervised-00000-of-00001.parquet'
    }

    # Load the datasets using pandas
    train_df = pd.read_parquet("hf://datasets/stanfordnlp/imdb/" + splits["train"])
    test_df = pd.read_parquet("hf://datasets/stanfordnlp/imdb/" + splits["test"])

    # Tokenize the datasets
    train_encodings = tokenize_data(train_df)
    test_encodings = tokenize_data(test_df)

    # Create PyTorch datasets
    train_dataset = IMDbDataset(train_encodings, train_df['label'].tolist())
    test_dataset = IMDbDataset(test_encodings, test_df['label'].tolist())
        
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Initialize model, optimizer, criterion
    if load_saved_model:
        model_path = os.path.join(wandb.run.dir, best_model_path)
        model = load_model(model_path)
        model.to(device)
    else:
        model = SentimentClassifier().to(device)
    optimizer = optim.Adam(model.classifier.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()

    # Train and evaluate with model saving
    best_accuracy = 0.0
    for epoch in range(epochs):
        train_loss = train(model, train_loader, optimizer, criterion, device)
        val_accuracy = evaluate(model, test_loader, criterion, device)
        # Save the model if it has the best accuracy so far
        if val_accuracy > best_accuracy:
            best_accuracy = val_accuracy
            #save the model in the wandb run directory 
            torch.save(model.state_dict(), os.path.join(wandb.run.dir, f"{val_accuracy:.4f}_{batch_size}_{learning_rate}_{epoch}_model.pth"))
        
        wandb.log({"epoch_training_loss": train_loss, "epoch_validation_accuracy": val_accuracy})
        print(f"Batch Size: {batch_size}, Learning Rate: {learning_rate} - Epoch {epoch+1}/{epochs} - Training Loss: {train_loss:.4f} - Validation Accuracy: {val_accuracy:.4f}")
        
    wandb.finish()

if __name__ == "__main__":
    main()