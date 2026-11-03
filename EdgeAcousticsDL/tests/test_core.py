import numpy as np
import torch
from edgeacoustics.config import Config
from edgeacoustics.audio import logmel,audio_windows,Normalizer
from edgeacoustics.model import TinySoundMTLNet
from edgeacoustics.metrics import summarize,optimize_threshold
from edgeacoustics.dataset import stratified_split,verify_splits,leave_machine_id_out,leave_fault_family_out

def test_preprocessing_shape_and_segments():
    c=Config();x=np.random.default_rng(2).standard_normal(160000).astype('float32')
    windows=list(audio_windows(x,c));assert len(windows)==19
    f=logmel(windows[0],c);assert f.shape==(128,32);assert np.isfinite(f).all()

def test_model_and_gradients():
    cfg=Config(channels=(8,12,16,24),hidden=16,attention_dim=8)
    model=TinySoundMTLNet(cfg)
    a,m,w=model(torch.randn(2,1,128,32))
    assert a.shape==(2,) and m.shape==(2,4) and w.shape[0]==2
    assert torch.allclose(w.sum(1),torch.ones(2),atol=1e-5)
    (a.mean()+m.mean()).backward()
    assert any(p.grad is not None for p in model.parameters())

def test_split_and_fault_protocol():
    rows=[]
    for machine in ('fan','pump','slider','valve'):
        for mid in ('00','02','04','06'):
            for label in [0,1]:
                for i in range(12):
                    rows.append(dict(path=f'{machine}/{mid}/{label}/{i}.wav',recording_id=f'{machine}/{mid}/{label}/{i}',
                         machine=machine,machine_id=mid,anomaly=label,machine_label=0))
    split=stratified_split(rows);verify_splits(split)
    assert set(r['split'] for r in split)=={'train','val','test'}
    left=leave_machine_id_out(rows,'00');assert all(r['split']=='test' for r in left if r['machine_id']=='00')
    fault=leave_fault_family_out(rows,'pump');assert all(r['split']=='test' for r in fault if r['machine']=='pump' and r['anomaly'])

def test_metrics():
    y=[0,0,1,1];p=[.1,.2,.9,.8];ym=[0,1,2,3];pm=np.eye(4)
    th=optimize_threshold(y,p);assert 0<=th<=1
    score=summarize(y,p,ym,pm,th);assert score['machine_accuracy']==1
