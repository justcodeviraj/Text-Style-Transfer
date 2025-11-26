# Text Style Transfer using Back Translation and Controlled Generation

I implemented a neural text style transfer system that converts between informal and formal writing styles. The approach uses T5-based seq2seq models with controlled generation and back translation for data augmentation.

## Dataset

I used the GYAFC-FR (Grammarly's Yahoo Answers Formality Corpus - Family & Relationships) dataset, which contains parallel informal/formal sentence pairs extracted from Yahoo Answers. The dataset provides high-quality human-written formal paraphrases of informal questions and answers, making it ideal for supervised style transfer training.

```python
from datasets import load_dataset
dataset = load_dataset("RUCAIBox/Style-Transfer", "gyafc_fr")
```

The dataset is stored locally in `data/gyafc_fr/` after initial download, with train and test splits maintained separately.

## Implementation

I built the system with three main components working together. First, I trained a RoBERTa-based binary classifier that distinguishes between informal (label=0) and formal (label=1) text. This classifier serves two purposes: evaluating the style transfer quality and filtering back-translated pairs during data augmentation.

For the actual style transfer, I implemented two T5-small models - a forward model that converts informal text to formal, and a backward model that does the reverse. I used prefix-based controlled generation where each input is prepended with either `"formalize: "` or `"informalize: "` to explicitly tell the model which direction to transfer. This gives fine-grained control over the generation process.

The back translation module leverages both models to generate synthetic training data. I take unlabeled informal text, translate it to formal style, verify the transfer quality with the classifier, then translate back to informal. If the round-trip maintains both style accuracy and semantic content (confidence ≥ 0.8), I keep the pair as augmented training data.

## Training Configuration

I trained the style classifier for 2 epochs with a batch size of 16 and learning rate of 2e-5. The classifier uses standard cross-entropy loss and achieves around 90% accuracy on held-out data. For the RoBERTa model, I found that 2 epochs were sufficient to avoid overfitting while getting good style discrimination.

The T5 style transfer models required more careful tuning. I trained both the forward and backward models for 2 epochs with batch size of 8 and learning rate of 5e-5. I used gradient clipping at 1.0 to stabilize training and implemented a linear warmup schedule for 10% of the total training steps. The smaller batch size was necessary due to GPU memory constraints with T5's encoder-decoder architecture. I set the maximum sequence length to 128 tokens, which covers most conversational sentences in the dataset.

## Generation Configuration

The generation process uses several key parameters that I tuned for quality. I set `num_beams=4` for beam search, which explores multiple output sequences and picks the best one according to the model's scoring. This produces more coherent outputs than greedy decoding but remains computationally efficient compared to larger beam sizes.

I kept `temperature=1.0` as the default, which maintains the model's learned probability distribution without artificial sharpening or smoothing. For cases requiring more creative or diverse outputs, the temperature can be adjusted higher (1.2-1.5), while lower values (0.7-0.8) make generation more conservative.

To prevent repetitive text, I set `no_repeat_ngram_size=3`, which blocks the model from repeating any 3-gram that already appeared in the output. This is crucial for maintaining fluency in longer generations. I also enabled `early_stopping=True` so the model can finish generation as soon as all beam hypotheses reach the end token, avoiding unnecessary padding.

The maximum generation length is set to 128 tokens, matching the training sequence length. During inference, the model typically generates outputs that are similar in length to the inputs, which is appropriate for style transfer where content should be preserved.

## Evaluation Metrics

I implemented two metrics to evaluate the style transfer quality, both calculated in `eval/evaluation.py`.

**Style Transfer Accuracy (STA)** measures how well the model achieved the target style. I pass all generated texts through the trained classifier and count what percentage matches the intended formality level. For example, if I'm evaluating informal→formal transfer, I check how many outputs the classifier labels as formal (label=1). This gives an objective measure of style transfer success.

**Content Preservation Score (CPS)** measures whether the semantic meaning was maintained during transfer. I calculate word overlap between source and generated text - specifically, the percentage of source words that appear in the output. While simple, this metric effectively captures content retention. Higher overlap means the model kept the original meaning intact rather than generating unrelated formal text.

I report both metrics separately rather than combining them, since they capture different aspects of quality. A good style transfer system should score high on both - successfully changing style (high STA) while keeping content (high CPS).

## Back Translation Process

The back translation pipeline generates augmented training pairs from unlabeled monolingual data. I first translate source text to the target style using the forward model, then verify the transfer was successful using the classifier. Next, I translate back to the source style using the backward model. 

The confidence score combines two signals: style accuracy (0 or 1 based on classifier prediction) and semantic similarity (word overlap between original and round-trip text). I calculate it as `(style_correct + word_overlap) / 2`. Only pairs with confidence ≥ 0.8 are kept as augmented data.

This process helps in two ways: it generates more training examples from unlabeled text, and it tends to produce high-quality pairs since both models must agree on the transfer. The augmented data gets saved to `data/augmented/style_pairs_confidence.jsonl` with confidence scores attached.

## Repo Structure

```
TST/
├── data/
│   ├── data_prep.py
│   ├── gyafc_em/
│   ├── gyafc_fr/
│   └── augmented/
│       └── style_pairs_confidence.jsonl
│
├── eval/
│   ├── evaluation.py
│   └── results/
│       └── test_results.jsonl
│
├── models/
│   ├── style_classifier.py
│   ├── style_transfer.py
│   ├── back_translation.py
│   └── checkpoints/
│       ├── classifier_model.pt
│       ├── forward_model.pt
│       └── backward_model.pt
│
├── train/
│   └── train.py
│
└── requirements.txt
```


## Running the Code

The training script automatically handles model checkpoints. On first run, it trains all three models and saves them. On subsequent runs, it loads existing checkpoints and skips training.

```bash
python train/train.py
```

First run takes 2-3 hours on a single GPU (trains classifier, forward model, backward model). Later runs take 2-5 minutes since they just load checkpoints and run inference.

## Usage Example

```python
from models.style_transfer import StyleTransferModel

model = StyleTransferModel(device='cuda')
model.load('models/checkpoints/forward_model.pt')

texts = ["hey whats up?", "gonna be late"]
formal = model.generate(texts, style='formal', num_beams=4, temperature=1.0)
# Output: ["Hello, how are you?", "I will be arriving late."]
```

## Requirements

```
torch>=2.0.0
transformers>=4.30.0
datasets>=2.12.0
numpy>=1.24.0
tqdm>=4.65.0
```

Install with: `pip install -r requirements.txt`

## Results

The system achieves around 85-90% style transfer accuracy on the test set with 70-75% content preservation using T5-small. The back translation augmentation typically generates 80-90% valid pairs (confidence ≥ 0.8) from the unlabeled data, which can be used to further improve the models through continued training.
