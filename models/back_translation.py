from style_transfer import StyleTransferModel
from style_classifier import StyleClassifierModel
from typing import List, Tuple, Dict


class BackTranslationModel:    
    def __init__(self, forward_model: StyleTransferModel, 
                 backward_model: StyleTransferModel,
                 classifier: StyleClassifierModel):
        self.forward_model = forward_model
        self.backward_model = backward_model
        self.classifier = classifier
    
    def back_translate(self, texts: List[str], source_style='informal', 
                      target_style='formal', confidence_threshold=0.8) -> List[Tuple[str, str, float]]:
    
        results = []
        
        
        forward_texts = self.forward_model.generate(texts, style=target_style) # Forward translation
        backward_texts = self.backward_model.generate(forward_texts, style=source_style)  # Backward translation
        
        forward_labels = self.classifier.predict(forward_texts)
        target_label = 1 if target_style == 'formal' else 0
        
        
        #confidence based on Style classifier and Semantic similarity (simple word overlap)

        for orig, fwd, bwd, label in zip(texts, forward_texts, backward_texts, forward_labels):
            
            style_correct = (label == target_label)
            
            orig_words = set(orig.lower().split())
            bwd_words = set(bwd.lower().split())
            if len(orig_words) > 0:
                overlap = len(orig_words & bwd_words) / len(orig_words)
            else:
                overlap = 0

            confidence = (float(style_correct) + overlap) / 2
            
            if confidence >= confidence_threshold:
                results.append((orig, fwd, confidence))
        
        return results
    
    def augment_dataset(self, unlabeled_texts: List[str], style='informal') -> List[Dict]:
      
        target_style = 'formal' if style == 'informal' else 'informal'
        augmented_data = []
        
        results = self.back_translate(unlabeled_texts, source_style=style, 
                                     target_style=target_style)
        
        for orig, translated, confidence in results:
            if style == 'informal':
                augmented_data.append({
                    'informal': orig,
                    'formal': translated,
                    'confidence': confidence
                })
            else:
                augmented_data.append({
                    'formal': orig,
                    'informal': translated,
                    'confidence': confidence
                })

        return augmented_data