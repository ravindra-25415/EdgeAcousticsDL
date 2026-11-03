"""Generate a tiny realistic-format dataset (NOT real scientific data)."""
import argparse
from pathlib import Path
import numpy as np
import soundfile as sf
p=argparse.ArgumentParser();p.add_argument('--output',default='demo_mimii');p.add_argument('--per-group',type=int,default=6)
a=p.parse_args();rng=np.random.default_rng(42);root=Path(a.output)
for m,frequency in [('fan',100),('pump',160),('slider',220),('valve',300)]:
 for mid in ['00','02','04','06']:
  for status in ['normal','abnormal']:
   folder=root/m/f'id_{mid}'/status;folder.mkdir(parents=True,exist_ok=True)
   for j in range(a.per_group):
    t=np.arange(160000,dtype=np.float32)/16000
    signal=(0.08*np.sin(2*np.pi*(frequency+int(mid))*t)+rng.normal(0,.03,len(t))).astype(np.float32)
    if status=='abnormal':signal+=0.05*np.sin(2*np.pi*(frequency*2.3)*t).astype(np.float32)
    sf.write(folder/f'{j:04d}.wav',signal,16000)
print(f'Synthetic smoke-test audio saved to {root}. NOT MIMII.')
