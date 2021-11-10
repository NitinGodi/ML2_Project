from nltk.tokenize import RegexpTokenizer
import pandas as pd
with open('train.txt','r+') as f:
    raw = f.read()

tokenizer = RegexpTokenizer(r'(__label__[12])\s([^:\n]+):([^\n]+)\n')

x = tokenizer.tokenize(raw)

df = pd.DataFrame(x,columns=['target','title','text'])
print(df.head())

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)