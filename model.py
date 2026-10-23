# ai_agent

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertForSequenceClassification
from torch.optim import AdamW
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import torch.nn.functional as F

# Device Configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load Dataset
HScode = r"C:\Users\USER\OneDrive - University of Bolton\FW_ Capstone project - case study 1_WTO\AI_Agent_WTO\trade_concerns_exploded (1).csv"
statement = r"C:\Users\USER\OneDrive - University of Bolton\FW_ Capstone project - case study 1_WTO\AI_Agent_WTO\Trade_concerns_Statements.csv"
df = pd.read_csv(HScode, encoding = 'ISO-8859-1')
df1 = pd.read_csv(statement, encoding='ISO-8859-1')

product_codes_df = df
statements_df = df1

# Merge
combined_df = pd.merge(product_codes_df, statements_df, left_on="IMS Id", right_on="ImsId")

# Combine Text
combined_df['Title'] = combined_df['Title'].fillna('').astype(str)
combined_df['Summary'] = combined_df['Summary'].fillna('').astype(str)

combined_df['Text'] = combined_df[['Title', 'Summary']].fillna('').astype(str).agg(' '.join, axis=1)

# Prepare Targets
combined_df['HS Products'] = combined_df['HS Products'].fillna("").apply(lambda x: x.split(';') if x else [])
combined_df['ICS Products'] = combined_df['ICS Products'].fillna("").apply(lambda x: x.split(';') if x else [])

mlb_hs = MultiLabelBinarizer()
y_hs = mlb_hs.fit_transform(combined_df['HS Products'])

# Features
X = combined_df['Text'].values

# Split Data
X_train, X_val, y_train, y_val = train_test_split(X, y_hs, test_size=0.2, random_state=42)

# Tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Dataset
class TradeConcernDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len=512):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        encoding = self.tokenizer(
            self.texts[idx],
            truncation=True,
            padding='max_length',
            max_length=self.max_len,
            return_tensors='pt'
        )
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.FloatTensor(self.labels[idx])
        }

train_dataset = TradeConcernDataset(X_train, y_train, tokenizer)
val_dataset = TradeConcernDataset(X_val, y_val, tokenizer)

train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=8)

# Model
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=y_hs.shape[1])
model.to(device)

optimizer = AdamW(model.parameters(), lr=2e-5)

# Training
epochs = 10
for epoch in range(epochs):
    model.train()
    total_loss = 0
    for batch in train_loader:
        optimizer.zero_grad()
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        loss = F.binary_cross_entropy_with_logits(logits, labels)
        
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(train_loader):.4f}")

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

# Save Model
torch.save(model.state_dict(), "trade_concern_bert_model.pth")
