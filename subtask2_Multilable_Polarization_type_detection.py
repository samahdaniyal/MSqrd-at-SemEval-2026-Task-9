import pandas as pd

from sklearn.metrics import recall_score, precision_score, f1_score
import numpy as np

import torch

import os
import glob

from sklearn.metrics import f1_score, hamming_loss
from sklearn.model_selection import train_test_split

from transformers import (
    AutoTokenizer,
    AutoConfig,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
    DataCollatorWithPadding,
    EarlyStoppingCallback
)
from torch.utils.data import Dataset, DataLoader
from torch import nn
import torch
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

os.environ["HF_HOME"] = os.path.expanduser("~/.cache/huggingface")
os.environ["TRANSFORMERS_CACHE"] = os.environ["HF_HOME"]

####################################
#          data processing 
####################################

# Configuration
LABEL_KEYS = ['gender/sexual','political','religious','racial/ethnic','other']

def load_split(base_path):
    print(f"Loading data from: {base_path}")

    all_files = glob.glob(os.path.join(base_path, "*.csv"))
    print(f"Found {len(all_files)} CSV files: {[os.path.basename(f) for f in all_files]}")

    df_list = []
    for f in all_files:
        lang_code = os.path.basename(f).split('.')[0]  # e.g., 'eng', 'deu'
        temp_df = pd.read_csv(f)
        temp_df['lang'] = lang_code
        df_list.append(temp_df)
        print(f"Loaded {os.path.basename(f)}: {len(temp_df)} rows")

    df = pd.concat(df_list, ignore_index=True)
    print(f"Total rows: {len(df)}")
    print(f"Columns: {df.columns.tolist()}")
    return df


# Load predefined splits
train_path = 'test_phase/subtask2/train'
val_path   = 'test_phase/subtask2/dev'

train = load_split(train_path).reset_index(drop=True)
val   = load_split(val_path).reset_index(drop=True)

print(f"\nTraining set size: {len(train)}")
print(f"Validation set size: {len(val)}")
print(f"Languages in train: {train['lang'].unique()}")
print(f"Languages in val: {val['lang'].unique()}")

train.head()

# Fix the dataset class by inheriting from torch.utils.data.Dataset
class PolarizationDataset(torch.utils.data.Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length # Store max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        encoding = self.tokenizer(text, truncation=True, padding=False, max_length=self.max_length, return_tensors='pt')

        # Ensure consistent tensor conversion for all items
        item = {key: encoding[key].squeeze() for key in encoding.keys()}
        # CHANGE THIS LINE: Use torch.float instead of torch.long for multi-label classification
        item['labels'] = torch.tensor(label, dtype=torch.float)
        return item

####################################
#          Loss function 
####################################

from torch import nn
import torch.nn.functional as F

model_name = 'cardiffnlp/twitter-xlm-roberta-base-sentiment'
# model_name = "/home/sandesh/Downloads/MSqrd/twitter-xlm-roberta-base-sentiment"

# Calculate class weights based on label distribution
label_counts = train[LABEL_KEYS].sum(axis=0)
total_samples = len(train)
pos_weights = (total_samples - label_counts) / label_counts
pos_weights = torch.tensor(pos_weights.values, dtype=torch.float)

print("\nPositive class weights:")
for label, weight in zip(LABEL_KEYS, pos_weights):
    print(f"  {label}: {weight:.2f}")

# class WeightedLossTrainer(Trainer):
#     def __init__(self, pos_weight=None, *args, **kwargs):
#         super().__init__(*args, **kwargs)
#         self.pos_weight = pos_weight

#     def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
#         labels = inputs.pop("labels")
#         outputs = model(**inputs)
#         logits = outputs.logits

#         # Use weighted BCE loss
#         loss_fct = nn.BCEWithLogitsLoss(pos_weight=self.pos_weight.to(logits.device))
#         loss = loss_fct(logits, labels)

#         return (loss, outputs) if return_outputs else loss


# class WeightedFocalLossTrainer(Trainer):
#     def __init__(self, pos_weight=None, gamma=1.5, *args, **kwargs):
#         super().__init__(*args, **kwargs)
#         self.pos_weight = pos_weight
#         self.gamma = gamma

#     def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
#         labels = inputs.pop("labels")
#         outputs = model(**inputs)
#         logits = outputs.logits

#         # BCE with logits (no reduction yet)
#         bce_loss = F.binary_cross_entropy_with_logits(
#             logits,
#             labels,
#             pos_weight=self.pos_weight.to(logits.device),
#             reduction="none"
#         )

#         # Focal loss modulation
#         probs = torch.sigmoid(logits)
#         pt = torch.where(labels == 1, probs, 1 - probs)
#         focal_factor = (1 - pt) ** self.gamma

#         loss = (focal_factor * bce_loss).mean()

#         return (loss, outputs) if return_outputs else loss


class AsymmetricFocalLoss(nn.Module):
    def __init__(
        self,
        pos_weight: torch.Tensor,
        gamma_pos: float = 1.5,
        gamma_neg: float = 1.5,
        clip: float = 0.003,
    ):
        super().__init__()

        # Register as buffer so it follows .to(device)
        self.register_buffer("pos_weight", pos_weight)

        self.gamma_pos = gamma_pos
        self.gamma_neg = gamma_neg
        self.clip = clip

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        targets = targets.float()
        probs = torch.sigmoid(logits)

        if self.clip > 0:
            probs = torch.clamp(probs, self.clip, 1.0 - self.clip)

        bce = F.binary_cross_entropy_with_logits(
            logits,
            targets,
            pos_weight=self.pos_weight,
            reduction="none"
        )

        focal_weight = (
            targets * ((1.0 - probs) ** self.gamma_pos) +
            (1.0 - targets) * ((probs) ** self.gamma_neg)
        )

        return (focal_weight * bce).mean()


class AsymmetricFocalTrainer(Trainer):
    def __init__(self, loss_fn: nn.Module, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # IMPORTANT: move loss_fn to same device as model
        self.loss_fn = loss_fn.to(self.model.device)

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits

        loss = self.loss_fn(logits, labels)
        return (loss, outputs) if return_outputs else loss


# Load the tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Create train and validation datasets for multilabel
train_dataset = PolarizationDataset(
    train["text"].tolist(),
    train[LABEL_KEYS].values.tolist(),
    tokenizer,
    max_length=128
)

val_dataset = PolarizationDataset(
    val["text"].tolist(),
    val[LABEL_KEYS].values.tolist(),
    tokenizer,
    max_length=128
)

####################################
#          model 
####################################

# Load the model
model = AutoModelForSequenceClassification.from_pretrained(
    model_name,
    num_labels=5,
    problem_type="multi_label_classification",
    ignore_mismatched_sizes=True
)


####################################
#          training args 
####################################

# Define metrics function for multi-label classification
def compute_metrics_multilabel(p):
    # Sigmoid the predictions to get probabilities
    probs = torch.sigmoid(torch.from_numpy(p.predictions))
    # Convert probabilities to predicted labels (0 or 1)
    preds = (probs > 0.5).int().numpy()
    # Compute macro F1 score
    return {
        'f1_macro': f1_score(p.label_ids, preds, average='macro'),
        'f1_micro': f1_score(p.label_ids, preds, average='micro'),
        'hamming_loss': hamming_loss(p.label_ids, preds)
        }

# Define training arguments
training_args = TrainingArguments(
    output_dir=f"./",
    num_train_epochs=6,
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=8,
    eval_strategy="epoch",
    save_strategy="epoch",
    logging_steps=50,
    weight_decay=0.01,
    warmup_ratio=0.15,
    lr_scheduler_type="cosine",
    disable_tqdm=False,
    load_best_model_at_end=True,
    metric_for_best_model="f1_macro",
    greater_is_better=True,
    seed=42,
    optim="adamw_torch_fused",
    gradient_accumulation_steps=2,
    max_grad_norm=1.0,
    save_total_limit=2,
    report_to="none"
)

####################################
#          trainer.train() 
####################################

# Initialize the WeightedLossTrainer instead of regular Trainer
# trainer = WeightedFocalLossTrainer(
#     pos_weight=pos_weights,
#     gamma=1.5,
#     model=model,
#     args=training_args,
#     train_dataset=train_dataset,
#     eval_dataset=val_dataset,
#     compute_metrics=compute_metrics_multilabel,
#     data_collator=DataCollatorWithPadding(tokenizer),
#     callbacks=[
#         EarlyStoppingCallback(early_stopping_patience=3),
#     ]
# )

loss_fn = AsymmetricFocalLoss(
    pos_weight=pos_weights,
    gamma_pos=1.5,
    gamma_neg=1.5,
    clip=0.005,
)

trainer = AsymmetricFocalTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    data_collator=DataCollatorWithPadding(tokenizer),
    compute_metrics=compute_metrics_multilabel,
    loss_fn=loss_fn,
    callbacks=[
        EarlyStoppingCallback(
            early_stopping_patience=2, 
            early_stopping_threshold=0.002),
    ]
)


# Train the model
trainer.train()

# Evaluate the model on the validation set
eval_results = trainer.evaluate()
print(f"Macro F1 score on validation set for Subtask 2: {eval_results['eval_f1_macro']}")

####################################
#       validation prediction 
####################################

# Get validation predictions
preds_output = trainer.predict(val_dataset)
probs = torch.sigmoid(torch.from_numpy(preds_output.predictions)).numpy()
labels = preds_output.label_ids

best_thresholds = []

for i, label in enumerate(LABEL_KEYS):
    best_f1 = 0
    best_t = 0.5

    for t in np.linspace(0.05, 0.9, 18):
        preds = (probs[:, i] > t).astype(int)
        f1 = f1_score(labels[:, i], preds, zero_division=0)

        if f1 > best_f1:
            best_f1 = f1
            best_t = t

    best_thresholds.append(best_t)
    print(f"{label}: best threshold = {best_t:.2f}, F1 = {best_f1:.3f}")

print("Best thresholds:", best_thresholds)

# Save model & tokenizer after training
trainer.save_model("subtask2_model")  # saves both model and tokenizer
tokenizer.save_pretrained("subtask2_model")

####################################
#       devlopment data 
####################################

# ==================== Development Data (Prediction Stage) ====================
dev_base_path = 'test_phase/subtask2/test'
all_dev_files = glob.glob(os.path.join(dev_base_path, "*.csv"))

dev_df_list = []
for f in all_dev_files:
    lang_code = os.path.basename(f).split('.')[0]
    temp_df = pd.read_csv(f)
    temp_df['lang'] = lang_code
    dev_df_list.append(temp_df)

dev_all = pd.concat(dev_df_list, ignore_index=True)
print(f"Total development rows: {len(dev_all)}")

# Create dataset for predictions (no labels)
class PredictionDataset(torch.utils.data.Dataset):
    def __init__(self, texts, tokenizer, max_length=128):
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding=False,
            max_length=self.max_length,
            return_tensors='pt'
        )
        return {key: encoding[key].squeeze() for key in encoding.keys()}
        

# Prepare prediction dataset
pred_dataset = PredictionDataset(
    dev_all['text'].tolist(),
    tokenizer,
    max_length=128
)
# Generate predictions using the Trainer
predictions = trainer.predict(pred_dataset)
probs = torch.sigmoid(torch.from_numpy(predictions.predictions))
# pred_labels = (probs > 0.5).int().numpy()
pred_labels = np.zeros_like(probs)

for i, t in enumerate(best_thresholds):
    pred_labels[:, i] = (probs[:, i] > t).int()


LABEL_KEYS = ['gender/sexual','political','religious','racial/ethnic','other']


# ==================== Save Predictions ====================
pred_df = dev_all.copy()
for i, label in enumerate(LABEL_KEYS):
    pred_df[label] = pred_labels[:, i]

output_dir = 'subtask_2'
os.makedirs(output_dir, exist_ok=True)

languages = sorted(pred_df['lang'].unique())
for lang in languages:
    lang_df = pred_df[pred_df['lang'] == lang].copy()
    output_df = lang_df[['id'] + LABEL_KEYS]
    output_file = os.path.join(output_dir, f'pred_{lang}.csv')
    output_df.to_csv(output_file, index=False)
    print(f"Saved {len(output_df)} predictions for {lang} -> {output_file}")

print(f"\n✓ All prediction files saved to '{output_dir}/' directory")

# ==================== Verify Output Format ====================
sample_lang = languages[0]
sample_file = os.path.join(output_dir, f'pred_{sample_lang}.csv')
sample_df = pd.read_csv(sample_file)
print(f"\nSample output file (pred_{sample_lang}.csv):")
print(sample_df.head())

import shutil

# Create zip file for subtask_2
shutil.make_archive("subtask_2", 'zip', root_dir=".", base_dir="subtask_2")
print("✓ Created subtask_2.zip - available in output")