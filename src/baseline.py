"""Portable transcription of supplied original-feature notebook. RF/MLP random states are fixed at 42.
Run: python src/baseline.py
Writes freshly trained original-feature model scores to results.
"""
from pathlib import Path
import numpy as np,pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import precision_score,recall_score,f1_score
from xgboost import XGBClassifier
R=Path(__file__).resolve().parents[1]
df=pd.read_csv(R/'data/raw/wallets_features_classes_combined.csv').drop(columns=['Time step']).drop_duplicates()
for col in df.columns[2:]:df[col]=MinMaxScaler().fit_transform(df[[col]]).ravel()
addresses=df.loc[df['class']!=3,'address'];df=df.loc[df.address.isin(addresses)]
y=df['class'].map(lambda x:0 if x==2 else 1)
xtr,xte,ytr,yte=train_test_split(df,y,test_size=.30,random_state=15,shuffle=False)
xtr=xtr.drop(columns=['address','class']).values;xte=xte.drop(columns=['address','class']).values
models={'LR':LogisticRegression(max_iter=1000),'RF':RandomForestClassifier(n_estimators=50,random_state=42,n_jobs=4),'MLP':MLPClassifier(solver='adam',learning_rate_init=.001,max_iter=500,random_state=42),'XGB':XGBClassifier(objective='multi:softmax',num_class=2,random_state=42)}
rows=[]
for name,m in models.items():
 m.fit(xtr,ytr.values);pred=m.predict(xte);rows.append({'Model':name,'Precision':precision_score(yte,pred),'Recall':recall_score(yte,pred),'F1 Score':f1_score(yte,pred),'Micro-F1':f1_score(yte,pred,average='micro')});print(rows[-1],flush=True)
(R/'results').mkdir(exist_ok=True);pd.DataFrame(rows).to_csv(R/'results/original_feature_baseline.csv',index=False)
