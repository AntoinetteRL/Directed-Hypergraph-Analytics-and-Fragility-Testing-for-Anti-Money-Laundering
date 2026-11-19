"""Compute the nine retained structural features from raw address relations.
The distributed canonical feature table is used by the main reproduction workflow.
"""
from pathlib import Path
import importlib.util
import numpy as np
import pandas as pd
import networkx as nx
R=Path(__file__).resolve().parents[1];raw=R/'data/raw';out=R/'results';out.mkdir(exist_ok=True)
addresses=pd.read_csv(raw/'wallets_features.csv',usecols=['address'])['address'].drop_duplicates().sort_values().tolist()
edges=pd.read_csv(raw/'AddrAddr_edgelist.csv').drop_duplicates();edges=edges[edges.input_address!=edges.output_address]
g=nx.DiGraph();g.add_nodes_from(addresses);g.add_edges_from(edges.itertuples(index=False,name=None))
pr=nx.pagerank(g,alpha=.85,tol=1e-12,max_iter=1000)
wc={v:len(c) for c in nx.weakly_connected_components(g) for v in c}
valid=[v for v in addresses if g.out_degree(v)>0];valid.sort(key=lambda v:g.out_degree(v))
sources=[valid[i] for i in np.linspace(0,len(valid)-1,64,dtype=int)]
bc=nx.betweenness_centrality_subset(g,sources,list(g.nodes),normalized=True);scale=len(g)/len(sources)
spec=importlib.util.spec_from_file_location('b',R/'src/hypergraph.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
hg=m.construct_hypergraph_features(raw/'AddrTx_edgelist.csv',raw/'TxAddr_edgelist.csv')
f=pd.DataFrame({'address':addresses,'addr_pagerank':[pr[v] for v in addresses],'addr_betweenness':[bc[v]*scale for v in addresses],'addr_weak_component_size':[wc[v] for v in addresses]})
f=f.merge(hg,on='address',how='left').fillna(0);f.to_csv(out/'reconstructed_structural_features.csv.gz',index=False)
print('Saved structural features. Compare with the canonical feature table before substituting inputs.')
