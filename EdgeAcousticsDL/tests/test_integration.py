import json
import numpy as np
from pathlib import Path
import torch
from edgeacoustics.engine import checkpoint_load
from edgeacoustics.streaming import replay
from edgeacoustics.benchmark import benchmark
from edgeacoustics.compress import magnitude_prune

def test_pruning():
    import torch.nn as nn
    net=nn.Sequential(nn.Linear(10,12),nn.ReLU(),nn.Linear(12,1))
    zeros,total=magnitude_prune(net,.3)
    assert 0<zeros<total

def test_checkpoint_and_streaming(tmp_path):
    # Builds self-contained checkpoint without relying on an external MIMII download.
    from edgeacoustics.config import Config
    from edgeacoustics.model import TinySoundMTLNet
    from edgeacoustics.audio import Normalizer
    import soundfile as sf
    cfg=Config(channels=(8,12,16,24),hidden=16,attention_dim=8)
    model=TinySoundMTLNet(cfg)
    torch.save(dict(state=model.state_dict(),config=cfg.to_dict(),model_name='tiny',model_kwargs={}),tmp_path/'best.pt')
    Normalizer(np.zeros(128),np.ones(128)).save(tmp_path/'normalization.npz')
    (tmp_path/'threshold.json').write_text(json.dumps({'threshold':.5}))
    audio=tmp_path/'test.wav';sf.write(audio,np.zeros(160000,dtype='float32'),16000)
    log=tmp_path/'replay.jsonl';replay(tmp_path/'best.pt',audio,log)
    rows=[json.loads(s) for s in log.read_text().splitlines()]
    assert len(rows)==19 and isinstance(rows[0]['alert'],bool)
    result=benchmark(tmp_path/'best.pt',tmp_path/'bench.json',repeats=2,threads=1)
    assert result['inference_ms_mean']>=0
