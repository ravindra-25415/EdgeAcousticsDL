"""MIMII file manifest and recording-level leakage-safe split protocols."""
from pathlib import Path
import re,csv,json,random
import numpy as np
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
from .audio import load_features
from .config import Config
from . import CLASSES

def scan_mimii(root):
    """Find WAV files under varied MIMII archive layouts (6 dB only)."""
    root=Path(root)
    rows=[]
    for file in sorted(root.rglob('*.wav')):
        parts=[p.lower() for p in file.parts]
        joined='/'.join(parts)
        machine=next((m for m in CLASSES if any(re.search(r'(^|[^a-z])'+m+r'([^a-z]|$)',p) for p in parts)),None)
        # MIMII uses 'slider' (slide rail). Also accept 'slide_rail'.
        if machine is None and ('slide_rail' in joined or 'slide rail' in joined):machine='slider'
        if machine is None:continue
        condition=next((a for a in ('normal','abnormal') if a in parts),None)
        if condition is None:continue
        ids=re.findall(r'(?:id|machine)[_ -]?(00|02|04|06)(?:\D|$)',joined)
        if not ids:continue
        # Avoid accidentally including 0 or -6dB archives when SNR is encoded.
        snrs=re.findall(r'(?:^|[/_ -])(?:snr[_ -]?)?(-?6|0)\s?db(?:[/_. -]|$)',joined)
        if snrs and any(x!='6' for x in snrs):continue
        rows.append({'path':str(file.resolve()),'machine':machine,'machine_id':ids[-1],
                     'anomaly':int(condition=='abnormal'),'machine_label':CLASSES.index(machine),
                     'recording_id':str(file.resolve())})
    if not rows:raise RuntimeError(f'No MIMII files found in {root}. Expected machine/id_00/normal/*.wav etc.')
    return rows

def save_manifest(rows,path):
    with open(path,'w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=['path','machine','machine_id','anomaly','machine_label','recording_id','split'])
        writer.writeheader();writer.writerows(rows)

def load_manifest(path):
    with open(path,newline='') as f:return [{**r,'anomaly':int(r['anomaly']),'machine_label':int(r['machine_label'])} for r in csv.DictReader(f)]

def stratified_split(rows,seed=42):
    """70/15/15 within each machine × ID × condition stratum, seeded and deterministic."""
    from collections import defaultdict
    groups=defaultdict(list)
    for i,r in enumerate(rows):groups[(r['machine'],r['machine_id'],r['anomaly'])].append(i)
    rng=np.random.default_rng(seed);mapping={}
    for name,ix in sorted(groups.items()):
        if len(ix)<3:raise ValueError(f'Stratum {name} has only {len(ix)} files; need at least 3')
        ix=np.asarray(ix)[rng.permutation(len(ix))]
        n=len(ix);ntrain=int(np.floor(.70*n));nval=int(np.floor(.15*n))
        ntrain=max(1,ntrain);nval=max(1,nval)
        if ntrain+nval>=n:ntrain=n-nval-1
        for split,subset in [('train',ix[:ntrain]),('val',ix[ntrain:ntrain+nval]),('test',ix[ntrain+nval:])]:
            for i in subset:mapping[int(i)]=split
    out=[{**r,'split':mapping[i]} for i,r in enumerate(rows)]
    verify_splits(out)
    return out

def leave_machine_id_out(rows,machine_id,seed=42):
    test=[{**r,'split':'test'} for r in rows if r['machine_id']==machine_id]
    pool=[r for r in rows if r['machine_id']!=machine_id]
    strata=[f"{r['machine']}|{r['anomaly']}" for r in pool]
    tr,va=train_test_split(np.arange(len(pool)),test_size=.176470588,random_state=seed,stratify=strata)
    result=test+[{**r,'split':'train' if i in set(tr) else 'val'} for i,r in enumerate(pool)]
    verify_splits(result)
    return result

def leave_fault_family_out(rows,machine,seed=42):
    # No fault-type labels in MIMII. Exclude all anomaly examples of one machine type from train and val.
    base=stratified_split(rows,seed)
    return [r for r in base if r['split']=='test' or not(r['machine']==machine and r['anomaly']==1)]

def verify_splits(rows):
    seen={}
    for r in rows:
        key=r['recording_id'];value=r['split']
        if key in seen and seen[key]!=value:raise AssertionError(f'Recording leakage: {key}')
        seen[key]=value

class WindowDataset(Dataset):
    def __init__(self,records,normalizer,cfg=Config(),augment=False,channel=0,cache_dir=None):
        self.records=list(records);self.norm=normalizer;self.cfg=cfg;self.augment=augment;self.channel=channel
        self.cache_dir=Path(cache_dir) if cache_dir else None
        if self.cache_dir:self.cache_dir.mkdir(parents=True,exist_ok=True)
        self.index=[]
        for rid,r in enumerate(self.records):
            # Full 10-s MIMII -> 19 windows; verify at retrieval for nonstandard input.
            self.index.extend((rid,i) for i in range(19))
        self._memo={}
    def __len__(self):return len(self.index)
    def _record(self,rid):
        if rid not in self._memo:
            r=self.records[rid]
            if self.cache_dir:
                import hashlib
                key=hashlib.sha256((r['path']+str(self.cfg.to_dict())).encode()).hexdigest()[:24]
                file=self.cache_dir/f'{key}.npy'
                if file.exists():x=np.load(file)
                else:
                    x=load_features(r['path'],self.cfg,self.channel)
                    np.save(file,x)
            else:x=load_features(r['path'],self.cfg,self.channel)
            if x.shape[0]!=19:raise ValueError(f'Expected 19 windows from 10s recording, got {x.shape[0]}: {r["path"]}')
            # bounded cache for worker memory safety
            if len(self._memo)>=16:self._memo.pop(next(iter(self._memo)))
            self._memo[rid]=self.norm.transform(x)
        return self._memo[rid]
    def __getitem__(self,idx):
        rid,win=self.index[idx];r=self.records[rid]
        x=self._record(rid)[win].copy()
        if self.augment and np.random.random()<.25:
            x+=np.random.normal(0,.02,x.shape).astype(np.float32)
        return x[None],np.float32(r['anomaly']),np.int64(r['machine_label']),rid
