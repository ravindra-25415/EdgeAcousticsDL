"""Repeatable baselines, ablation, cross-machine and held-out anomaly protocols."""
import json
from pathlib import Path
import numpy as np
from .config import Config
from .dataset import load_manifest,save_manifest,leave_machine_id_out,leave_fault_family_out
from .engine import train
from .classical import run as run_classical

SEEDS=[42,123,2024,2025,3407]
DL_COMPARATORS=['cnn','lstm','gru','attention','tinycnn','mobilenetv2','efficientnet_lite']
CLASSICAL=['svm','random_forest','knn','decision_tree','xgboost','lightgbm']
ABLATIONS={'full':{},'without_tcn':{'use_tcn':False},'without_attention':{'use_attention':False},
 'standard_convolutions':{'depthwise':False},'anomaly_only':{'mode':'anomaly'},'machine_only':{'mode':'machine'}}

def run_experiment(manifest,root,experiment='primary',seeds=SEEDS,cfg=Config(),cache=None):
    out=Path(root);out.mkdir(parents=True,exist_ok=True)
    records=load_manifest(manifest);tasks=[]
    if experiment=='primary':tasks=[('primary','tiny',{},records)]
    elif experiment=='deep_baselines':tasks=[(name,name,{},records) for name in DL_COMPARATORS]
    elif experiment=='ablation':tasks=[(name,'tiny',kwargs,records) for name,kwargs in ABLATIONS.items()]
    elif experiment=='lomo':tasks=[('holdout_'+id,'tiny',{},leave_machine_id_out(records,id)) for id in ('00','02','04','06')]
    elif experiment=='lofo':tasks=[('excluded_'+name,'tiny',{},[r for r in leave_fault_family_out(records,name) if r['split']!='test' or r['machine']==name]) for name in ('fan','pump','slider','valve')]
    elif experiment=='classical':
        return [run_classical(manifest,out/name,name,cfg) for name in CLASSICAL]
    else:raise ValueError(f'Unknown experiment {experiment}')
    summary=[]
    for label,model,kwargs,rows in tasks:
        folder=out/label;folder.mkdir(exist_ok=True)
        split=folder/'split.csv';save_manifest(rows,split)
        for seed in seeds:
            c=Config(**{**cfg.to_dict(),'seed':seed})
            result=train(split,folder/f'seed_{seed}',c,model,kwargs,cache)
            summary.append({'experiment':label,'seed':seed,**result})
            (out/'summary.json').write_text(json.dumps(summary,indent=2))
    return summary
