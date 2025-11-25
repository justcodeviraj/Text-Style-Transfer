import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from transformers import (
    T5Tokenizer, T5ForConditionalGeneration,
    get_linear_schedule_with_warmup
)
from tqdm import tqdm
from typing import List
import warnings
warnings.filterwarnings('ignore')

class StyleTransferModel:
    
    def __init__(self, model_name='t5-small', device='cuda'):
        self.device = device if torch.cuda.is_available() else 'cpu'
        self.tokenizer = T5Tokenizer.from_pretrained(model_name)
        self.model = T5ForConditionalGeneration.from_pretrained(model_name).to(self.device)
        
    def train(self, train_dataset, val_dataset=None, epochs=3, batch_size=8, lr=5e-5):
       
        train_loader = DataLoader(
            train_dataset, 
            batch_size=batch_size, 
            shuffle=True
        )
        
        optimizer = AdamW(self.model.parameters(), lr=lr)
        total_steps = len(train_loader) * epochs
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=total_steps // 10,
            num_training_steps=total_steps
        )
        
        self.model.train()
        
        for epoch in range(epochs):
            total_loss = 0
            progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
            
            for batch in progress_bar:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                # Replace padding token id in labels with -100
                labels[labels == self.tokenizer.pad_token_id] = -100
                
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )

                loss = outputs.loss
                total_loss += loss.item()
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                
                progress_bar.set_postfix({'loss': loss.item()})
            
            avg_loss = total_loss / len(train_loader)
            print(f"Epoch {epoch+1} completed. Average Loss: {avg_loss:.4f}")
    
    def generate(self, texts: List[str], style: str = 'formal', 
                 max_length=128, num_beams=4, temperature=1.0) -> List[str]:
    
        self.model.eval()
    
        style_prefix = "formalize: " if style == 'formal' else "informalize: "
        input_texts = [style_prefix + text for text in texts]
        
        encoding = self.tokenizer(
            input_texts,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors='pt'
        ).to(self.device)
      
        with torch.no_grad():
            outputs = self.model.generate(
                input_ids=encoding['input_ids'],
                attention_mask=encoding['attention_mask'],
                max_length=max_length,
                num_beams=num_beams, # controlled generation 
                temperature=temperature,
                early_stopping=True,
                no_repeat_ngram_size=3
            )

        generated_texts = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
        return generated_texts
