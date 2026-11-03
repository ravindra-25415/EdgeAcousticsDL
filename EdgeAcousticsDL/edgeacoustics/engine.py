"""Train, validate, record-level aggregate, checkpoint and evaluate."""
import json,random,time
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from .config import Config
from .dataset import WindowDataset,load_manifest
from .audio import Normalizer
from .model import build_model
from .metrics import optimize_threshold,summarize,plot_metrics

def seed_all(seed):
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
    if torch.cuda.is_available():torch.cuda.manual_seed_all(seed)

def choose_device(name='auto'):return 'cuda' if name=='auto' and torch.cuda.is_available() else ('cpu' if name=='auto' else name)

def loaders(manifest,norm,cfg,cache=None):
    bysplit={k:[r for r in manifest if r['split']==k] for k in ('train','val','test')}
    return {k:DataLoader(WindowDataset(rows,norm,cfg,augment=False,cache_dir=cache),batch_size=cfg.batch_size,
      shuffle=k=='train',num_workers=cfg.num_workers) for k,rows in bysplit.items() if rows}

def forward_loss(model,batch,device,cfg):
    x,ya,ym,_=batch;x=x.to(device);ya=ya.to(device);ym=ym.to(device)
    a,m,_=model(x)
    la=nn.functional.binary_cross_entropy_with_logits(a,ya)
    lm=nn.functional.cross_entropy(m,ym)
    mode=getattr(model,'mode','both')
    if mode=='anomaly':return la
    if mode=='machine':return lm
    return cfg.anomaly_weight*la+cfg.machine_weight*lm

@torch.no_grad()
def predict(model,loader,device):
    model.eval();acc={}
    for x,ya,ym,rids in loader:
        a,m,_=model(x.to(device))
        pa=torch.sigmoid(a).cpu().numpy();pm=torch.softmax(m,1).cpu().numpy()
        for i,rid in enumerate(rids.tolist()):
            item=acc.setdefault(rid,{'a':[],'m':[],'ya':int(ya[i]),'ym':int(ym[i])})
            item['a'].append(float(pa[i]));item['m'].append(pm[i])
    keys=sorted(acc)
    return dict(y_anomaly=np.array([acc[k]['ya'] for k in keys]),
        p_anomaly=np.array([np.mean(acc[k]['a']) for k in keys]),
        y_machine=np.array([acc[k]['ym'] for k in keys]),
        p_machine=np.array([np.mean(acc[k]['m'],axis=0) for k in keys]),record_ids=keys)

def checkpoint_load(path,device='cpu'):
    data=torch.load(path,map_location=device,weights_only=False)
    cfg=Config(**data['config']);model=build_model(data['model_name'],cfg,**data.get('model_kwargs',{}))
    model.load_state_dict(data['state']);model.to(device).eval()
    return model,cfg,data

def train(manifest_path,output,cfg=Config(),model_name='tiny',model_kwargs=None,cache=None,normalizer_path=None):
    seed_all(cfg.seed);device=choose_device(cfg.device);out=Path(output);out.mkdir(parents=True,exist_ok=True)
    rows=load_manifest(manifest_path);trainrows=[r for r in rows if r['split']=='train']
    valrows=[r for r in rows if r['split']=='val']
    if not trainrows or not valrows:raise ValueError('Training and validation records are required')
    norm=Normalizer.load(normalizer_path) if normalizer_path else Normalizer().fit(trainrows,cfg)
    norm.save(out/'normalization.npz')
    cfg.save(out/'config.json')
    loader=loaders(rows,norm,cfg,cache)
    kwargs=model_kwargs or {};model=build_model(model_name,cfg,**kwargs).to(device)
    opt=torch.optim.Adam(model.parameters(),lr=cfg.lr,weight_decay=cfg.weight_decay,betas=(.9,.999),eps=1e-8)
    scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau(opt,factor=.5,patience=cfg.lr_patience,min_lr=cfg.min_lr)
    history=[];best=float('inf');stale=0
    for epoch in range(cfg.epochs):
        model.train();losses=[]
        for batch in loader['train']:
            opt.zero_grad(set_to_none=True);loss=forward_loss(model,batch,device,cfg);loss.backward();opt.step();losses.append(float(loss.item()))
        model.eval();validation=[]
        with torch.no_grad():
            for batch in loader['val']:validation.append(float(forward_loss(model,batch,device,cfg).item()))
        value=float(np.mean(validation));scheduler.step(value)
        hist={'epoch':epoch+1,'train_loss':float(np.mean(losses)),'val_loss':value,'lr':opt.param_groups[0]['lr']}
        history.append(hist);print(json.dumps(hist),flush=True)
        if value<best-1e-7:
            best=value;stale=0
            torch.save(dict(state=model.state_dict(),config=cfg.to_dict(),model_name=model_name,model_kwargs=kwargs,epoch=epoch+1),out/'best.pt')
        else:stale+=1
        if stale>=cfg.patience:break
    (out/'history.json').write_text(json.dumps(history,indent=2))
    model,_,_=checkpoint_load(out/'best.pt',device)
    val=predict(model,loader['val'],device)
    threshold=optimize_threshold(val['y_anomaly'],val['p_anomaly'])
    (out/'threshold.json').write_text(json.dumps({'threshold':threshold,'source':'validation recording F1'},indent=2))
    if 'test' in loader:
        result=evaluate_model(model,loader['test'],device,threshold,out/'test')
    else:result={}
    return result

def evaluate_model(model,loader,device,threshold,outdir):
    out=Path(outdir);out.mkdir(parents=True,exist_ok=True)
    pred=predict(model,loader,device)
    metrics=summarize(pred['y_anomaly'],pred['p_anomaly'],pred['y_machine'],pred['p_machine'],threshold)
    (out/'metrics.json').write_text(json.dumps(metrics,indent=2))
    np.savez_compressed(out/'predictions.npz',**pred)
    plot_metrics(pred['y_anomaly'],pred['p_anomaly'],pred['y_machine'],pred['p_machine'],out)
    return metrics

def evaluate(checkpoint,manifest_path,outdir,split='test',cache=None):
    model,cfg,_=checkpoint_load(checkpoint,choose_device('auto'));device=next(model.parameters()).device
    root=Path(checkpoint).parent;norm=Normalizer.load(root/'normalization.npz')
    records=[r for r in load_manifest(manifest_path) if r['split']==split]
    dl=DataLoader(WindowDataset(records,norm,cfg,cache_dir=cache),batch_size=cfg.batch_size,shuffle=False)
    th=json.loads((root/'threshold.json').read_text())['threshold']
    return evaluate_model(model,dl,device,th,outdir)
