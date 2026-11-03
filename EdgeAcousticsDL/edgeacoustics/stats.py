"""Per-run inference and paired comparison with Holm correction."""
import json
from pathlib import Path
import numpy as np
from scipy import stats

def report(summary_path,output,metric='joint_f1'):
    rows=json.loads(Path(summary_path).read_text());group={}
    for r in rows:
        if metric in r:group.setdefault(r['experiment'],{})[r['seed']]=float(r[metric])
    result={'metric':metric,'groups':{},'paired_comparisons':[]}
    for name,values in group.items():
        a=np.asarray(list(values.values()))
        ci=stats.t.interval(.95,len(a)-1,loc=a.mean(),scale=stats.sem(a)) if len(a)>1 else (None,None)
        result['groups'][name]=dict(n=len(a),mean=float(a.mean()),sd=float(a.std(ddof=1)) if len(a)>1 else None,ci95=list(ci))
    names=list(group);tests=[]
    for i in range(len(names)):
        for j in range(i+1,len(names)):
            a,b=group[names[i]],group[names[j]];common=sorted(set(a)&set(b))
            if len(common)<2:continue
            diff=np.array([a[s]-b[s] for s in common])
            if np.allclose(diff,0):p=1.
            else:p=float(stats.wilcoxon(diff).pvalue)
            effect=float(diff.mean()/diff.std(ddof=1)) if diff.std(ddof=1)>0 else None
            tests.append(dict(a=names[i],b=names[j],n=len(common),p_raw=p,paired_cohens_d=effect))
    indices=np.argsort([t['p_raw'] for t in tests]);m=len(indices);adjusted=np.zeros(m)
    running=0.
    for rank,idx in enumerate(indices):
        running=max(running,min(1.,(m-rank)*tests[idx]['p_raw']));adjusted[idx]=running
    for idx,t in enumerate(tests):t['p_holm']=float(adjusted[idx])
    result['paired_comparisons']=tests
    Path(output).write_text(json.dumps(result,indent=2));return result
