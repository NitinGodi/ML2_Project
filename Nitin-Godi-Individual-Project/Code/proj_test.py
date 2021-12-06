# ============Imports===============
from tqdm.auto import tqdm
import pandas as pd
from transformers import MobileBertTokenizerFast, get_scheduler
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import RobertaForSequenceClassification, MobileBertForSequenceClassification, AdamW
from sklearn.metrics import accuracy_score, precision_score, recall_score
from nltk import RegexpTokenizer
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
# =================Hyperparameters=================
num_epochs = 5
checkpoint1 = 'distilroberta-base'
checkpoint2 = 'google/mobilebert-uncased'
max_len = 512
model1 = 'distilroberta'
model2 = 'mobilebert'
model3 = 'ensemble'
model_name = model2

# ===========Load data and preprocess ==============

with open('test.ft.txt','r+') as f:
    raw = f.read()

tokenizer1 = RegexpTokenizer(r'(__label__[12])\s([^:]+):([^\n]+)')
data_list = tokenizer1.tokenize(raw)

test_df = pd.DataFrame(data_list,columns=['label','title','text'])[:100]
test_df['label'].replace({'__label__1':0, '__label__2':1},inplace=True)
test_texts = test_df['text'].values.tolist()
test_labels = test_df['label'].values.tolist()

# =================Tokenizing and creating data loaders=====================
tokenizer = MobileBertTokenizerFast.from_pretrained(checkpoint2)
test_encodings = tokenizer(test_texts, truncation=True, padding=True, max_length=max_len)

class YelpDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

test_dataset = YelpDataset(test_encodings, test_labels)

test_dataloader = DataLoader(test_dataset, shuffle=True, batch_size=6)

# ==========================Models============================================

class Classifier(nn.Module):
    def __init__(self):
        super(Classifier, self).__init__()
        self.m1 = RobertaForSequenceClassification.from_pretrained(checkpoint1)
        self.m2 = MobileBertForSequenceClassification.from_pretrained(checkpoint2)
        self.dropout = nn.Dropout(0.3)
        self.out3 = nn.Linear(4,2)
    def forward(self, ids):
        output_1 = self.m1(ids, return_dict=False)
        output_2 = self.dropout(output_1[0])
        output_3 = self.m2(ids, return_dict=False)
        output_4 = self.dropout(output_3[0])
        output_5 = torch.cat((output_2, output_4), dim=1)
        output = self.out3(output_5)
        return output

if model_name == model1:
    model = RobertaForSequenceClassification.from_pretrained(checkpoint1)
elif model_name == model2:
    model = MobileBertForSequenceClassification.from_pretrained(checkpoint2)
elif model_name == model3:
    model = Classifier()

model.load_state_dict(torch.load('model_{}.pt'.format(model_name), map_location=device))
model.to(device)

PRED = []
Y = []
def update(pred,y):
    x = pred.detach().cpu().numpy()
    z = y.detach().cpu().numpy()
    for i in range(len(x)):
        PRED.append(x[i])
        Y.append(z[i])

num_eval_steps = len(test_dataloader)
progress_bar = tqdm(range(num_eval_steps))

model.eval()
for batch in test_dataloader:
    torch.no_grad()
    input_ids = batch['input_ids'].to(device)
    labels = batch['labels'].to(device)
    if (model_name == model1) | (model_name == model2):
        outputs = model(input_ids, labels=labels)
        logits = outputs.logits
    else:
        outputs = model(input_ids)
        logits = outputs
    predictions = torch.argmax(logits, dim=-1)
    update(predictions,labels)
    progress_bar.update(1)

acc = accuracy_score(Y, PRED)
print("\nValidation accuracy:",acc)

pre = precision_score(Y, PRED)
print("\nValidation precision:",pre)

rec = recall_score(Y, PRED)
print("\nValidation recall:",rec)