"""Execute the experiments and compare new outputs with published values."""
from pathlib import Path
import argparse, subprocess,sys,json,importlib.util
import numpy as np,pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import precision_score,recall_score,f1_score,confusion_matrix
from xgboost import XGBClassifier
R=Path(__file__).resolve().parent;O=R/'results';O.mkdir(exist_ok=True)
p=argparse.ArgumentParser();p.add_argument('--stage',default='all',choices=['all','prepare','baseline','classification','prediction','motifs','fragility','verify']);a=p.parse_args()
def wants(s):return a.stage in ['all',s]
if wants('prepare'):
 subprocess.run([sys.executable,str(R/'src/prepare.py')],check=True)
if wants('baseline'):
 subprocess.run([sys.executable,str(R/'src/baseline.py')],check=True)
if wants('classification'):
 z=np.load(R/'data/processed/leakage_controlled_actor_64_features_arrays.npz',allow_pickle=True)
 models={'LR':LogisticRegression(max_iter=1000,random_state=42),'RF':RandomForestClassifier(n_estimators=50,random_state=42,n_jobs=4),'MLP':MLPClassifier(hidden_layer_sizes=(100,),activation='relu',solver='adam',learning_rate_init=.001,alpha=.0001,max_iter=500,tol=1e-4,random_state=42,early_stopping=True,batch_size=1024),'XGB':XGBClassifier(objective='multi:softmax',num_class=2,random_state=42,n_jobs=8,tree_method='hist')};rows=[]
 for n,m in models.items():
  m.fit(z['Xtr'],z['ytr']);pred=m.predict(z['Xte']);tn,fp,fn,tp=confusion_matrix(z['yte'],pred).ravel()
  rows.append({'Model':n,'Precision':precision_score(z['yte'],pred),'Recall':recall_score(z['yte'],pred),'F1 Score':f1_score(z['yte'],pred),'Micro-F1':f1_score(z['yte'],pred,average='micro'),'TN':tn,'FP':fp,'FN':fn,'TP':tp});print(rows[-1],flush=True)
  (O/(n+'_parameters.json')).write_text(json.dumps(m.get_params(),default=str,indent=2))
 pd.DataFrame(rows).to_csv(O/'table_6.csv',index=False)
if wants('prediction'):
 z=np.load(R/'data/processed/leakage_controlled_actor_64_features_with_address_ids.npz',allow_pickle=True)
 m=XGBClassifier(objective='multi:softmax',num_class=2,random_state=42,n_jobs=8,tree_method='hist');m.fit(z['Xtr'],z['ytr']);prob=m.predict_proba(z['X_unknown'])[:,1]
 order=np.argsort(-prob,kind='mergesort');rank=np.empty(len(prob),dtype=int);rank[order]=np.arange(1,len(prob)+1)
 pd.DataFrame({'address':z['unknown_addresses'],'original_class':3,'predicted_licit_probability':1-prob,'predicted_illicit_probability':prob,'predicted_label':np.where(prob>=.5,1,2),'illicit_probability_rank':rank,'illicit_probability_percentile':100*(len(prob)-rank+1)/len(prob)}).to_csv(O/'unknown_probabilities.csv',index=False,float_format='%.10f')
 bins=[0,.1,.25,.5,.75,.9,1.0000001];counts=np.histogram(prob,bins)[0]
 pd.DataFrame({'lower':bins[:-1],'upper':[.1,.25,.5,.75,.9,1.0],'count':counts,'percentage':100*counts/len(prob)}).to_csv(O/'table_7.csv',index=False)
if wants('motifs') or wants('fragility'):
 spec=importlib.util.spec_from_file_location('frag',R/'src/fragility.py');mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
if wants('motifs'):
 edges=pd.read_csv(R/'data/raw/AddrAddr_edgelist.csv',dtype=str);labels=pd.read_csv(R/'data/raw/wallets_classes.csv');ms=mod.enumerate_motifs(edges,labels);rows=[]
 for k,m in enumerate(ms,1):m.to_csv(O/f'motif_{k}.csv',index=False);rows.append({'motif_type':k,'motifs':len(m)})
 pd.DataFrame(rows).to_csv(O/'table_8.csv',index=False)
if wants('fragility'):
 prob=O/'unknown_probabilities.csv'
 if not prob.exists():raise SystemExit('Run --stage prediction first.')
 subprocess.run([sys.executable,str(R/'src/fragility.py'),'--edges',str(R/'data/raw/AddrAddr_edgelist.csv'),'--labels',str(R/'data/raw/wallets_classes.csv'),'--probabilities',str(prob),'--out-dir',str(O/'fragility')],check=True)
if wants('verify'):
 subprocess.run([sys.executable,str(R/'tests/verify_results.py')],check=True)
