"""CPU benchmark with separate model-only and preprocessing-plus-model timing."""
import time,json,os,platform
from pathlib import Path
import numpy as np
import torch
import psutil
from .engine import checkpoint_load
from .audio import logmel,Normalizer

def benchmark(checkpoint,output,repeats=100,threads=4):
    torch.set_num_threads(threads)
    model,cfg,_=checkpoint_load(checkpoint,'cpu')
    norm=Normalizer.load(Path(checkpoint).parent/'normalization.npz')
    wave=np.zeros(cfg.segment_samples,dtype=np.float32)
    x=torch.from_numpy(norm.transform(logmel(wave,cfg)[None])[:,None])
    with torch.no_grad():
        for _ in range(10):model(x)
        latency=[];total=[]
        process=psutil.Process(os.getpid());rss0=process.memory_info().rss
        for _ in range(repeats):
            t=time.perf_counter()
            with torch.no_grad():model(x)
            t2=time.perf_counter()
            ft=norm.transform(logmel(wave,cfg)[None])[:,None]
            with torch.no_grad():model(torch.from_numpy(ft))
            end=time.perf_counter()
            latency.append((t2-t)*1000);total.append((end-t2)*1000)
        rss1=process.memory_info().rss
    out=dict(device=platform.platform(),processor=platform.processor(),runtime=f'torch {torch.__version__}',threads=threads,
      inference_ms_mean=float(np.mean(latency)),inference_ms_p95=float(np.percentile(latency,95)),
      preprocess_plus_inference_ms_mean=float(np.mean(total)),windows_per_second=float(1000/np.mean(total)),
      process_rss_mb=float(rss1/1024**2),rss_delta_mb=float((rss1-rss0)/1024**2),
      note='Synthetic zero waveform benchmark, CPU model processing only; excludes audio acquisition and rolling-window wait; no power estimate')
    Path(output).write_text(json.dumps(out,indent=2));return out
