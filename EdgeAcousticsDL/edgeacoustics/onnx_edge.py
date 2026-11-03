"""Deploy exported ONNX/QDQ model on CPU with the exact training normalization."""
import json,time
from pathlib import Path
import numpy as np
from .audio import load_audio,logmel,audio_windows,Normalizer
from . import CLASSES

class ONNXMonitor:
    def __init__(self,model_path,normalization_path,threshold=.5,history=5,threads=4):
        try:import onnxruntime as ort
        except ImportError as e:raise RuntimeError('Install onnxruntime on this device') from e
        from .config import Config
        self.cfg=Config()
        # Config can optionally be supplied alongside model in its export folder.
        cfg_path=Path(model_path).parent/'config.json'
        if cfg_path.exists():self.cfg=Config.load(cfg_path)
        self.norm=Normalizer.load(normalization_path)
        opts=ort.SessionOptions();opts.intra_op_num_threads=threads
        self.session=ort.InferenceSession(str(model_path),sess_options=opts,providers=['CPUExecutionProvider'])
        self.input=self.session.get_inputs()[0].name
        self.threshold=threshold;self.history=history;self.scores=[]
    def push(self,window,timestamp=None):
        start=time.perf_counter()
        x=self.norm.transform(logmel(window,self.cfg)[None])[:,None]
        outputs=self.session.run(None,{self.input:x.astype(np.float32)})
        logit=float(np.asarray(outputs[0]).reshape(-1)[0]);logits=np.asarray(outputs[1]).reshape(-1)
        a=1/(1+np.exp(-np.clip(logit,-80,80)))
        exp=np.exp(logits-logits.max());p=exp/exp.sum()
        self.scores.append((a,p));self.scores=self.scores[-self.history:]
        mean_a=float(np.mean([z[0] for z in self.scores]));mean_p=np.mean([z[1] for z in self.scores],axis=0)
        return {'timestamp':timestamp,'machine_type':CLASSES[int(mean_p.argmax())],
            'machine_probabilities':mean_p.tolist(),'anomaly_probability':mean_a,
            'status':'anomalous' if mean_a>=self.threshold else 'normal',
            'warmup':len(self.scores)<self.history,
            'alert':len(self.scores)==self.history and bool(mean_a>=self.threshold),
            'processing_latency_ms':(time.perf_counter()-start)*1000}

def replay_onnx(model,normalization,wav,output,threshold=.5,threads=4):
    monitor=ONNXMonitor(model,normalization,threshold=threshold,threads=threads)
    audio=load_audio(wav,monitor.cfg.sr)
    out=Path(output);out.parent.mkdir(parents=True,exist_ok=True)
    with out.open('w') as f:
        for i,window in enumerate(audio_windows(audio,monitor.cfg)):
            row=monitor.push(window,i*monitor.cfg.segment_hop/monitor.cfg.sr)
            f.write(json.dumps(row)+'\n')
    return str(out)
