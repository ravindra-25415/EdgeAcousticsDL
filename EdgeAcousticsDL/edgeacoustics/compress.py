"""Magnitude pruning + ONNX export; static INT8 requires onnxruntime quantization."""
import json
from pathlib import Path
import numpy as np
import torch
from torch import nn
from .engine import checkpoint_load
from .audio import Normalizer,load_features
from .dataset import load_manifest

def magnitude_prune(model,fraction=.3):
    """Global magnitude mask applied to eligible conv/linear weights; zeros retained in dense ops."""
    if not 0<=fraction<1:raise ValueError('sparsity must be within [0,1)')
    layers=[m for m in model.modules() if isinstance(m,(nn.Linear,nn.Conv1d,nn.Conv2d))]
    weights=[m.weight.detach().abs().flatten() for m in layers]
    flattened=torch.cat(weights)
    if fraction>0:
        threshold=torch.quantile(flattened,fraction)
        with torch.no_grad():
            for m in layers:m.weight.mul_((m.weight.abs()>threshold).to(m.weight.dtype))
    return sum(int((w.weight==0).sum()) for w in layers),sum(w.weight.numel() for w in layers)

def export(checkpoint,manifest,output,sparsity=.30,calibration_limit=128,quantize=True):
    out=Path(output);out.mkdir(parents=True,exist_ok=True)
    model,cfg,meta=checkpoint_load(checkpoint,'cpu')
    pruned,total=magnitude_prune(model,sparsity)
    torch.save({**meta,'state':model.state_dict()},out/'pruned.pt')
    (out/'pruning.json').write_text(json.dumps({'nonzero_weights':total-pruned,'total_weights':total,
        'sparsity':pruned/total,'note':'Unstructured sparsity does not guarantee dense-runtime speedup'},indent=2))
    model.eval()
    cfg.save(out/'config.json')
    path=out/'model.onnx'
    try:
        torch.onnx.export(model,torch.zeros(1,1,128,32),str(path),input_names=['spectrogram'],
          output_names=['anomaly_logits','machine_logits','attention_weights'],
          dynamic_axes={'spectrogram':{0:'batch'},'anomaly_logits':{0:'batch'},'machine_logits':{0:'batch'},'attention_weights':{0:'batch'}},opset_version=17)
    except Exception as exc:
        raise RuntimeError('ONNX export failed; install onnx, onnxscript and the matching torch exporter requirements') from exc
    if quantize:
        try:
            from onnxruntime.quantization import quantize_static,CalibrationDataReader,QuantType,QuantFormat
        except ImportError as exc:raise RuntimeError('Install onnxruntime to enable calibrated INT8 export') from exc
        rows=[r for r in load_manifest(manifest) if r['split']=='train']
        if not rows:raise ValueError('Calibration requires training records')
        norm=Normalizer.load(Path(checkpoint).parent/'normalization.npz')
        class Reader(CalibrationDataReader):
            def __init__(self):
                self.items=[]
                for r in rows:
                    x=norm.transform(load_features(r['path'],cfg))
                    for v in x:
                        self.items.append({'spectrogram':v[None,None].astype(np.float32)})
                        if len(self.items)>=calibration_limit:break
                    if len(self.items)>=calibration_limit:break
                self._it=iter(self.items)
            def get_next(self):return next(self._it,None)
            def rewind(self):self._it=iter(self.items)
        quantize_static(str(path),str(out/'model_int8.onnx'),Reader(),quant_format=QuantFormat.QDQ,
              per_channel=True,weight_type=QuantType.QInt8,activation_type=QuantType.QUInt8)
        # ONNX QDQ may contain float I/O; not claim full integer I/O.
    return {'onnx':str(path),'int8':str(out/'model_int8.onnx') if quantize else None,'pruned_weights':pruned}
