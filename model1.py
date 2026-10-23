# train_trade_agent.py

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import classification_report
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertForSequenceClassification
from torch.optim import AdamW
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import torch.nn.functional as F
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import multilabel_confusion_matrix, classification_report

# Load datasets
HScode = r"C:\Users\USER\OneDrive - University of Bolton\FW_ Capstone project - case study 1_WTO\AI_Agent_WTO\trade_concerns_exploded (1).csv"
statement = r"C:\Users\USER\OneDrive - University of Bolton\FW_ Capstone project - case study 1_WTO\AI_Agent_WTO\Trade_concerns_Statements.csv"
df = pd.read_csv(HScode, encoding = 'ISO-8859-1')
df1 = pd.read_csv(statement, encoding='ISO-8859-1')

# Merge
combined_df = pd.merge(df, df1, left_on="IMS Id", right_on="ImsId")

# Combine Text
combined_df['Title'] = combined_df['Title'].fillna('').astype(str)
combined_df['Summary'] = combined_df['Summary'].fillna('').astype(str)

combined_df['Text'] = combined_df[['Title', 'Summary']].fillna('').astype(str).agg(' '.join, axis=1)

# Prepare Targets
combined_df['HS Products'] = combined_df['HS Products'].fillna("").apply(lambda x: x.split(';') if x else [])
combined_df['ICS Products'] = combined_df['ICS Products'].fillna("").apply(lambda x: x.split(';') if x else [])
# Prepare label encoding (multi-label binarization)
mlb = MultiLabelBinarizer()
y = mlb.fit_transform(combined_df['HS Products'])

# Tokenize input text using BERT tokenizer
#tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
#inputs = tokenizer(list(df['input_text']), padding=True, truncation=True, return_tensors="pt", max_length=256)

# Features
X = combined_df['Text'].values

# Split Data
X_train, X_val, y_train, y_val = train_test_split(X, y_hs, test_size=0.2, random_state=42)

# Tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Dataset class
class TradeDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels
    def __len__(self):
        return len(self.labels)
    def __getitem__(self, idx):
        item = {key: val[idx] for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx], dtype=torch.float)
        return item

# Train-test split
train_texts, val_texts, train_labels, val_labels = train_test_split(inputs, y, test_size=0.2, random_state=42)
train_dataset = TradeDataset(train_texts, train_labels)
val_dataset = TradeDataset(val_texts, val_labels)

# DataLoader
train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=8)

# Model: BERT for multi-label classification
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=y.shape[1], problem_type="multi_label_classification")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Optimizer
optimizer = AdamW(model.parameters(), lr=5e-5)

# Training loop
epochs = 3
for epoch in range(epochs):
    model.train()
    for batch in train_loader:
        optimizer.zero_grad()
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        outputs = model(input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        loss = F.binary_cross_entropy_with_logits(logits, labels)
        loss.backward()
        optimizer.step()
    print(f"Epoch {epoch+1} complete. Loss: {loss.item()}")

# Save model and tokenizer
model.save_pretrained("./trade_agent_model")
tokenizer.save_pretrained("./trade_agent_model")

print("✅ Model trained and saved successfully!")

# Evaluation
model.eval()
all_preds = []
all_labels = []

with torch.no_grad():
    for batch in val_loader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)

        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        preds = torch.sigmoid(outputs.logits).cpu().numpy()
        preds = (preds > 0.5).astype(int)

        all_preds.extend(preds)
        all_labels.extend(labels.cpu().numpy())

acc = accuracy_score(all_labels, all_preds)
precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average='micro')

print(f"Accuracy: {acc:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1 Score: {f1:.4f}")
