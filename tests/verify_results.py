from pathlib import Path
import json
import numpy as np,pandas as pd
R=Path(__file__).resolve().parents[1];O=R/'results';checks={}
for table,actual in [('table_5','original_feature_baseline.csv'),('table_6','table_6.csv')]:
 if not (O/actual).exists():checks[table]={'status':'not_run'};continue
 ref=pd.read_csv(R/'expected'/(table+'.csv')).set_index('Model');cur=pd.read_csv(O/actual).set_index('Model');cols=['Precision','Recall','F1 Score','Micro-F1'];checks[table]={n:bool(np.all(np.abs(ref.loc[n,cols].astype(float)-cur.loc[n,cols].astype(float))<=.000501)) for n in ref.index}
if (O/'table_8.csv').exists():checks['table_8']=bool(pd.read_csv(O/'table_8.csv').equals(pd.read_csv(R/'expected/table_8.csv')))
if (O/'table_7.csv').exists():checks['table_7_counts']=bool(np.array_equal(pd.read_csv(O/'table_7.csv')['count'],[486067,31319,18761,9350,4810,7281]))
if (O/'fragility/fragility_summary.csv').exists():
 ref=pd.read_csv(R/'expected/table_9.csv');cur=pd.read_csv(O/'fragility/fragility_summary.csv');cols=['motifs','connected_n','disconnected_n','mean_remaining_paths','mean_path_loss'];checks['table_9']=bool(np.allclose(ref[cols],cur[cols],rtol=0,atol=5.1e-7))
(O/'verification.json').write_text(json.dumps(checks,indent=2));print(json.dumps(checks,indent=2))
if any(v is False or (isinstance(v,dict) and any(x is False for x in v.values())) for v in checks.values()):raise SystemExit('One or more fresh outputs differ from published reference values; inspect results/verification.json.')
