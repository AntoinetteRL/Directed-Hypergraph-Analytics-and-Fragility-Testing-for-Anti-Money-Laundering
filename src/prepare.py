"""Rebuild numerical inputs from the distributed canonical enhanced address table."""
from pathlib import Path
import numpy as np,pandas as pd,json
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
R=Path(__file__).resolve().parents[1];O=R/'results';O.mkdir(exist_ok=True)
ref=np.load(R/'data/processed/leakage_controlled_actor_64_features_arrays.npz',allow_pickle=True);cols=ref['feature_cols'].tolist()
x=pd.read_csv(R/'data/features/wallets_features_with_25_graph_hypergraph_features.csv.gz',usecols=['address']+cols,low_memory=False).drop_duplicates('address',keep='first')
y=pd.read_csv(R/'data/raw/wallets_classes.csv',dtype={'address':'string','class':'int8'});x['address']=x.address.astype('string');x=x.merge(y,on='address',how='left',validate='one_to_one');x['class']=x['class'].fillna(3).astype('int8');x[cols]=x[cols].replace([np.inf,-np.inf],np.nan).fillna(0).astype('float64')
k=x[x['class'].isin([1,2])];u=x[x['class']==3];sc=MinMaxScaler();X=sc.fit_transform(k[cols]).astype('float32');Y=(k['class']==1).to_numpy('int8');ids=k.address.astype(str).to_numpy(object)
xtr,xte,ytr,yte,atr,ate=train_test_split(X,Y,ids,test_size=.3,shuffle=False)
checks={n:bool(np.allclose(v,ref[n],rtol=0,atol=5e-7)) for n,v in [('Xtr',xtr),('Xte',xte),('ytr',ytr),('yte',yte)]};checks['address_disjoint']=not bool(set(atr)&set(ate));(O/'preprocessing_checks.json').write_text(json.dumps(checks,indent=2));print(checks)
np.savez_compressed(O/'prepared_inputs.npz',Xtr=xtr,Xte=xte,ytr=ytr,yte=yte,train_addresses=atr,test_addresses=ate,X_unknown=sc.transform(u[cols]).astype('float32'),unknown_addresses=u.address.astype(str).to_numpy(object),feature_cols=np.array(cols,dtype=object),scaler_data_min=sc.data_min_,scaler_data_max=sc.data_max_,scaler_scale=sc.scale_,scaler_min=sc.min_)
