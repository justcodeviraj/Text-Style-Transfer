import torch
import numpy as np
import random
import json
from data.data_prep import GYAFCDataset, load_data
from models.style_classifier import StyleClassifierModel
from models.style_transfer import StyleTransferModel
from models.back_translation import BackTranslationModel
from eval.evaluation import Evaluator
from transformers import T5Tokenizer
import warnings
warnings.filterwarnings('ignore')

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_seed(42)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
output_path = 'data/results/outputs.txt'
augmented_data_path = 'data/augmented/style_pairs_confidence.jsonl'

tokenizer = T5Tokenizer.from_pretrained('t5-small')

dataset = load_data()
train_dataset_formal = GYAFCDataset(dataset['train'], tokenizer, style='formal')
train_dataset_informal = GYAFCDataset(dataset['train'], tokenizer, style='informal')

print('Training Classifier')
classifier = StyleClassifierModel(device=device)
classifier.train(dataset['train'], epochs=2, batch_size=16)

print('Training informal to formal style transfer')
forward_model = StyleTransferModel(device=device) 
forward_model.train(train_dataset_formal, epochs=2, batch_size=8)

print('Evaluating informal to formal style transfer')
evaluator = Evaluator(classifier)
test_samples = dataset['test']
test_informal = [item['informal'] for item in test_samples]
test_formal_ref = [item['formal'] for item in test_samples]

generated_formal = forward_model.generate(test_informal, style='formal')
metrics = evaluator.evaluate_all(test_informal, generated_formal, target_style='formal')

print(f"\nStyle Transfer Accuracy: {metrics['style_accuracy']:.3f}")
print(f"Content Preservation: {metrics['content_preservation']:.3f}")

test_results = []
for i in range(len(test_informal)):
    test_results.append({'Input (Informal)'  : test_informal[i],
                         'Generated (Formal)': generated_formal[i],
                         'Reference (Formal)': test_formal_ref[i]})

with open(output_path, 'w', encoding='utf-8') as f:
    for item in test_results:
        f.write(json.dump(item, ensure_ascii=False) + '\n')
print(f'Test Results saved at {output_path}')

print('Training formal to informal style transfer')
backward_model = StyleTransferModel(device=device)
backward_model.train(train_dataset_informal, epochs=2, batch_size=8)

bt_module = BackTranslationModel(forward_model, backward_model, classifier)

#### Generate some unlabeled informal texts for augmentation
unlabeled_informal = [item['informal'] for item in dataset['test']]
augmented_data = bt_module.augment_dataset(unlabeled_informal, style='informal')

with open(augmented_data_path, 'w', encoding='utg-8') as f:
    json.dump(augmented_data, f, indent=4, ensure_ascii=False)
print(f'Augmented Data Saved at {augmented_data_path}')
