"""Recording-level classical comparators, tuning on validation only."""
from pathlib import Path
import json
import numpy as np
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import ParameterGrid
from sklearn.metrics import f1_score
from .audio import Normalizer,load_features
from .dataset import load_manifest
from .metrics import summarize,optimize_threshold
from .config import Config

def features(rows,norm,cfg):
    x=[];ya=[];ym=[]
    for r in rows:
        a=norm.transform(load_features(r['path'],cfg))
        # Per-band mean and std; one feature vector per recording.
        x.append(np.r_[a.mean((0,2)),a.std((0,2))]);ya.append(r['anomaly']);ym.append(r['machine_label'])
    return np.array(x),np.array(ya),np.array(ym)

def candidates(name):
    if name=='svm':return SVC(probability=True,random_state=42),{'C':[1,10],'gamma':['scale',.01]}
    if name=='random_forest':return RandomForestClassifier(random_state=42,n_jobs=-1),{'n_estimators':[100,300],'max_depth':[None,20,40]}
    if name=='knn':return KNeighborsClassifier(),{'n_neighbors':[3,5,7]}
    if name=='decision_tree':return DecisionTreeClassifier(random_state=42),{'max_depth':[10,20,40]}
    if name=='xgboost':
        from xgboost import XGBClassifier
        return XGBClassifier(n_jobs=4,eval_metric='logloss',random_state=42),{'n_estimators':[100,300],'max_depth':[4,6],'learning_rate':[.05,.1]}
    if name=='lightgbm':
        from lightgbm import LGBMClassifier
        return LGBMClassifier(verbosity=-1,random_state=42),{'n_estimators':[100,300],'max_depth':[4,6],'learning_rate':[.05,.1]}
    raise ValueError(name)

def run(manifest,output,name='random_forest',cfg=Config()):
    rows=load_manifest(manifest);groups={s:[r for r in rows if r['split']==s] for s in ('train','val','test')}
    out=Path(output);out.mkdir(parents=True,exist_ok=True)
    norm=Normalizer().fit(groups['train'],cfg)
    data={s:features(v,norm,cfg) for s,v in groups.items()}
    train_x,train_ya,train_ym=data['train'];val_x,val_ya,val_ym=data['val'];test_x,test_ya,test_ym=data['test']
    base,grid=candidates(name);fitted={};chosen={}
    from sklearn.base import clone
    for task,train_y,val_y in [('anomaly',train_ya,val_ya),('machine',train_ym,val_ym)]:
        best=-1
        for params in ParameterGrid(grid):
            model=clone(base).set_params(**params).fit(train_x,train_y)
            scores=model.predict_proba(val_x)
            if task=='anomaly':
                p=scores[:,list(model.classes_).index(1)]
                t=optimize_threshold(val_y,p)
                score=f1_score(val_y,p>=t,zero_division=0)
            else:score=f1_score(val_y,model.predict(val_x),average='macro',zero_division=0)
            if score>best:best=score;fitted[task]=model;chosen[task]=params
    an=fitted['anomaly'];mc=fitted['machine'];valp=an.predict_proba(val_x)[:,list(an.classes_).index(1)]
    threshold=optimize_threshold(val_ya,valp)
    result=summarize(test_ya,an.predict_proba(test_x)[:,list(an.classes_).index(1)],test_ym,mc.predict_proba(test_x),threshold)
    result['best_hyperparameters']=chosen;result['name']=name
    (out/'metrics.json').write_text(json.dumps(result,indent=2))
    import joblib
    joblib.dump({'anomaly':an,'machine':mc,'normalizer':norm,'threshold':threshold},out/'classical.joblib')
    return result
