from models.style_classifier import StyleClassifierModel
from typing import List, Dict

class Evaluator:
    
    def __init__(self, classifier: StyleClassifierModel):
        self.classifier = classifier
    
    def evaluate_style_transfer_accuracy(self, generated_texts: List[str], 
                                        target_style='formal') -> float:
       
        predictions = self.classifier.predict(generated_texts)
        target_label = 1 if target_style == 'formal' else 0
        accuracy = sum([1 for p in predictions if p == target_label]) / len(predictions)
        return accuracy
    
    def evaluate_content_preservation(self, source_texts: List[str], 
                                     generated_texts: List[str]) -> float:
      
        total_score = 0
        for src, gen in zip(source_texts, generated_texts):
            src_words = set(src.lower().split())
            gen_words = set(gen.lower().split())
            if len(src_words) > 0:
                overlap = len(src_words & gen_words) / len(src_words)
                total_score += overlap
        return total_score / len(source_texts)
    
    def evaluate_all(self, source_texts: List[str], generated_texts: List[str],
                    target_style='formal') -> Dict[str, float]:
        
        return {
            'style_accuracy': self.evaluate_style_transfer_accuracy(generated_texts, target_style),
            'content_preservation': self.evaluate_content_preservation(source_texts, generated_texts)
        }