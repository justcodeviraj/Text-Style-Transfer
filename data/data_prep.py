from torch.utils.data import Dataset

DATASET_FORMAL_TRAIN_PATH = r'data\gyafc_fr\train.tgt'
DATASET_INFORMAL_TRAIN_PATH = r'data\gyafc_fr\train.src'
DATASET_FORMAL_TEST_PATH = r'data\gyafc_fr\test.tgt'
DATASET_INFORMAL_TEST_PATH = r'data\gyafc_fr\test.src'

def load_data(
        formal_train_path: str = DATASET_FORMAL_TRAIN_PATH, 
        informal_train_path: str = DATASET_INFORMAL_TRAIN_PATH,
        formal_test_path: str = DATASET_FORMAL_TEST_PATH, 
        informal_test_path: str = DATASET_INFORMAL_TEST_PATH):
    
    dataset = {'train':[], 'test': []}

    with open(informal_train_path, "r", encoding="utf-8") as src_file, \
        open(formal_train_path, "r", encoding="utf-8") as tgt_file:

        for informal, formal in zip(src_file, tgt_file):
            informal = informal.strip()
            formal = formal.strip()

            if informal and formal:  # avoid empty lines
                dataset['train'].append({
                    "informal": informal,
                    "formal": formal
                })
    
    with open(informal_test_path, "r", encoding="utf-8") as src_file, \
        open(formal_test_path, "r", encoding="utf-8") as tgt_file:

        for informal, formal in zip(src_file, tgt_file):
            informal = informal.strip()
            formal = formal.strip()

            if informal and formal:  # avoid empty lines
                dataset['test'].append({
                    "informal": informal,
                    "formal": formal
                })
    return dataset
print(load_data()['train'][10])
class GYAFCDataset(Dataset):
    
    def __init__(self, data, tokenizer, max_length=128, style='formal'):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.style = style
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        
        # For GYAFC dataset from HuggingFace
        if self.style == 'formal':
            source_text = item['informal']
            target_text = item['formal']
            style_prefix = "formalize: "
        else:
            source_text = item['formal']
            target_text = item['informal']
            style_prefix = "informalize: "
        
        # Add style control token
        input_text = style_prefix + source_text
        
        # Tokenize
        input_encoding = self.tokenizer(
            input_text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        target_encoding = self.tokenizer(
            target_text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        return {
            'input_ids': input_encoding['input_ids'].squeeze(),
            'attention_mask': input_encoding['attention_mask'].squeeze(),
            'labels': target_encoding['input_ids'].squeeze(),
            'source_text': source_text,
            'target_text': target_text
        }



