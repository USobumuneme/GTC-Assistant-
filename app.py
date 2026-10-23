import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertForSequenceClassification
from torch.optim import AdamW
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import multilabel_confusion_matrix, classification_report, roc_auc_score, roc_curve

# Load Data
HScode = r"C:\Users\USER\OneDrive - University of Bolton\FW_ Capstone project - case study 1_WTO\AI_Agent_WTO\trade_concerns_exploded (1).csv"
statement = r"C:\Users\USER\OneDrive - University of Bolton\FW_ Capstone project - case study 1_WTO\AI_Agent_WTO\Trade_concerns_Statements.csv"
df = pd.read_csv(HScode, encoding = 'ISO-8859-1')
df1 = pd.read_csv(statement, encoding='ISO-8859-1')
df = df[['IMS Id', 'Domain', 'Title', 'Raising Members', 'HS Products',	'ICS Products',	'First date raised']]
df1 = df1[['ImsId', 'Summary']]
# Merge by ID
combined = pd.merge(df, df1, left_on="IMS Id", right_on="ImsId")

# Combine text
combined['Title'] = combined['Title'].fillna('')
combined['Statement'] = combined['Summary'].fillna('')
combined['text'] = combined['Title'] + " " + combined['Statement']
combined = combined.sample(2000, random_state=42)

# Targets - HS Products
#combined['HS Products'] = combined['HS Products'].fillna("").apply(lambda x: str(x).split(';') if x else [])
#combined['ICS Products'] = combined['ICS Products'].fillna("").apply(lambda x: x.split(';') if x else [])

mlb = MultiLabelBinarizer()
# Split HSProducts by comma, strip whitespace, and drop NaNs
combined['HS Products'] = combined['HS Products'].fillna('').apply(lambda x: [i.strip() for i in str(x).split(',') if i.strip() != ''])

# Now it's safe to transform
y = mlb.fit_transform(combined['HS Products'])

# Text data
X = combined['text'].tolist()

# Train-test split
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Tokenize
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
train_encodings = tokenizer(X_train, truncation=True, padding=True, max_length=256, return_tensors="pt")
val_encodings = tokenizer(X_val, truncation=True, padding=True, max_length=256, return_tensors="pt")

# Dataset class
class TradeDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels
    def __getitem__(self, idx):
        item = {key: val[idx] for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx], dtype=torch.float)
        return item
    def __len__(self):
        return len(self.labels)

train_dataset = TradeDataset(train_encodings, y_train)
val_dataset = TradeDataset(val_encodings, y_val)

# DataLoaders
train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=4)

# Model
model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=y.shape[1], problem_type="multi_label_classification")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Optimizer
optimizer = AdamW(model.parameters(), lr=5e-5)

# Training loop
for epoch in range(3):
    model.train()
    total_loss = 0
    for batch in train_loader:
        optimizer.zero_grad()
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        loss = F.binary_cross_entropy_with_logits(outputs.logits, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch {epoch+1}: Loss {total_loss:.4f}")

# Evaluation
model.eval()
all_preds, all_labels = [], []

with torch.no_grad():
    for batch in val_loader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].cpu().numpy()
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        preds = torch.sigmoid(outputs.logits).cpu().numpy()
        preds = (preds > 0.5).astype(int)
        all_preds.extend(preds)
        all_labels.extend(labels)

# Metrics
acc = accuracy_score(all_labels, all_preds)
p, r, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average='micro')
print(f"✅ Accuracy: {acc:.4f}, Precision: {p:.4f}, Recall: {r:.4f}, F1: {f1:.4f}")


# Confusion Matrices
conf_matrices = multilabel_confusion_matrix(all_labels, all_preds)

# Visualize confusion matrix for first 3 HS codes
for i in range(min(3, y.shape[1])):
    cm = conf_matrices[i]
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.title(f"Confusion Matrix for HS Code {mlb.classes_[i]}")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.show()

# Classification report
print("🧾 Classification Report:")
print(classification_report(all_labels, all_preds, target_names=mlb.classes_))

# ROC-AUC Score (micro-averaged)
roc_auc = roc_auc_score(all_labels, all_preds, average="micro")
print(f"📈 ROC-AUC (micro): {roc_auc:.4f}")

# OPTIONAL: ROC Curve for first 3 HS codes
for i in range(min(3, y.shape[1])):
    fpr, tpr, _ = roc_curve([row[i] for row in all_labels], [row[i] for row in all_preds])
    plt.plot(fpr, tpr, label=f'{mlb.classes_[i]} (AUC)')
plt.plot([0, 1], [0, 1], 'k--')
plt.title("ROC Curves for Selected HS Codes")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.legend()
plt.show()