"""Replay / ring-buffer monitoring with five-window rolling aggregation."""
import json,time
from collections import deque
from pathlib import Path
import numpy as np
import torch
from .audio import load_audio,logmel,Normalizer
from .engine import checkpoint_load
from . import CLASSES

class Monitor:
    def __init__(self,checkpoint,threshold=None,history=5):
        self.model,self.cfg,_=checkpoint_load(checkpoint,'cpu')
        self.norm=Normalizer.load(Path(checkpoint).parent/'normalization.npz')
        default=json.loads((Path(checkpoint).parent/'threshold.json').read_text())['threshold']
        self.threshold=default if threshold is None else threshold
        self.history=deque(maxlen=history)
    @torch.no_grad()
    def push(self,window,timestamp=None):
        start=time.perf_counter()
        feature=logmel(window,self.cfg)
        x=self.norm.transform(feature[None])[:,None]
        y,m,_=self.model(torch.from_numpy(x))
        a=torch.sigmoid(y).item();probs=torch.softmax(m,1)[0].numpy()
        self.history.append((a,probs))
        avg=float(np.mean([z[0] for z in self.history]))
        cls=np.mean([z[1] for z in self.history],axis=0)
        return dict(timestamp=timestamp,anomaly_probability=avg,machine_type=CLASSES[cls.argmax()],
          machine_probabilities=cls.tolist(),status='anomalous' if avg>=self.threshold else 'normal',
          alert=bool(avg>=self.threshold and len(self.history)==self.history.maxlen),
          warmup=bool(len(self.history)<self.history.maxlen),processing_latency_ms=(time.perf_counter()-start)*1000)

def replay(checkpoint,audio_path,output,threshold=None,history=5):
    from .audio import audio_windows
    monitor=Monitor(checkpoint,threshold,history)
    audio=load_audio(audio_path,monitor.cfg.sr)
    path=Path(output);path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('w') as f:
        for idx,window in enumerate(audio_windows(audio,monitor.cfg)):
            result=monitor.push(window,idx*monitor.cfg.segment_hop/monitor.cfg.sr)
            f.write(json.dumps(result)+'\n')
    return str(path)

def live(checkpoint,output,history=5):
    try:import sounddevice as sd
    except ImportError as exc:raise RuntimeError('Install sounddevice for optional live microphone capture') from exc
    monitor=Monitor(checkpoint,history=history)
    sr=monitor.cfg.sr;step=monitor.cfg.segment_hop;size=monitor.cfg.segment_samples
    ring=deque(maxlen=size)
    with open(output,'a') as f:
        with sd.InputStream(samplerate=sr,channels=1,blocksize=step,dtype='float32') as stream:
            while True:
                data,_=stream.read(step)
                ring.extend(data[:,0].tolist())
                if len(ring)==size:
                    item=monitor.push(np.asarray(ring,dtype=np.float32),time.time())
                    f.write(json.dumps(item)+'\n');f.flush();print(json.dumps(item),flush=True)
