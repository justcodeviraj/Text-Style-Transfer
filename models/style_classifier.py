import torch
from torch.optim import AdamW
from transformers import (
    RobertaTokenizer, RobertaForSequenceClassification
)
import random
from typing import List


class StyleClassifierModel:
    
    def __init__(self, model_name='roberta-base', num_labels=2, device='cuda'):
        self.device = device if torch.cuda.is_available() else 'cpu'
        self.tokenizer = RobertaTokenizer.from_pretrained(model_name)
        self.model = RobertaForSequenceClassification.from_pretrained(
            model_name, 
            num_labels=num_labels
        ).to(self.device)
        
    def train(self, train_data, epochs=3, batch_size=16, lr=2e-5):
    
        texts = []
        labels = []
        for item in train_data:
            texts.append(item['informal'])
            labels.append(0)  # 0 = informal
            texts.append(item['formal'])
            labels.append(1)  # 1 = formal
        
        dataset = [(text, label) for text, label in zip(texts, labels)]
        random.shuffle(dataset)
        
        optimizer = AdamW(self.model.parameters(), lr=lr)
        self.model.train()
        
        for epoch in range(epochs):
            total_loss = 0
            correct = 0
            total = 0
            
            for i in range(0, len(dataset), batch_size):
                batch = dataset[i:i+batch_size]
                batch_texts = [item[0] for item in batch]
                batch_labels = torch.tensor([item[1] for item in batch]).to(self.device)
                
                encoding = self.tokenizer(
                    batch_texts,
                    padding=True,
                    truncation=True,
                    max_length=128,
                    return_tensors='pt'
                ).to(self.device)
                
                outputs = self.model(**encoding, labels=batch_labels)
                loss = outputs.loss
            
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
                predictions = torch.argmax(outputs.logits, dim=-1)
                correct += (predictions == batch_labels).sum().item()
                total += len(batch_labels)
            
            avg_loss = total_loss / (len(dataset) / batch_size)
            accuracy = correct / total
            print(f"Epoch {epoch+1}/{epochs} - Loss: {avg_loss:.4f}, Accuracy: {accuracy:.4f}")
    
    def predict(self, texts: List[str], batch_size=16) -> List[int]:
        self.model.eval()
        predictions = []
        
        with torch.no_grad():
            for i in range(0, len(texts), batch_size):
                batch_texts = texts[i:i+batch_size]
                encoding = self.tokenizer(
                    batch_texts,
                    padding=True,
                    truncation=True,
                    max_length=128,
                    return_tensors='pt'
                ).to(self.device)
                
                outputs = self.model(**encoding)
                batch_predictions = torch.argmax(outputs.logits, dim=-1).cpu().numpy()
                predictions.extend(batch_predictions)
        
        return predictions
