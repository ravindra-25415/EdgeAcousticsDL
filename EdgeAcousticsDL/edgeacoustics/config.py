from dataclasses import dataclass, asdict
import json
from pathlib import Path

@dataclass
class Config:
    sr:int=16000
    n_fft:int=1024
    hop:int=512
    n_mels:int=128
    eps:float=1e-6
    segment_samples:int=16000
    segment_hop:int=8000
    n_frames:int=32
    batch_size:int=32
    epochs:int=100
    lr:float=.001
    min_lr:float=.00001
    patience:int=10
    lr_patience:int=5
    dropout:float=.3
    weight_decay:float=.0001
    anomaly_weight:float=.5
    machine_weight:float=.5
    num_workers:int=0
    seed:int=42
    device:str='auto'
    channels:tuple=(32,64,96,128)
    hidden:int=318
    attention_dim:int=64
    aggregation:int=5
    def to_dict(self):return asdict(self)
    @classmethod
    def load(cls,path=None):
        if path is None:return cls()
        data=json.loads(Path(path).read_text())
        return cls(**{k:v for k,v in data.items() if k in cls.__dataclass_fields__})
    def save(self,path):Path(path).write_text(json.dumps(self.to_dict(),indent=2))
