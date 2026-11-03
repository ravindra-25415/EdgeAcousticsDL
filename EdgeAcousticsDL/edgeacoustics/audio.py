"""Deterministic audio features shared by offline and replay inference."""
import numpy as np
import soundfile as sf
from scipy.signal import resample_poly, stft
from scipy.fft import rfft
from functools import lru_cache
from math import gcd
from .config import Config

@lru_cache(maxsize=12)
def mel_filters(sr=16000,n_fft=1024,n_mels=128):
    # Slaney-style triangular filters with HTK mel mapping. Persist config with checkpoints.
    def hz_to_mel(f):return 2595*np.log10(1+np.asarray(f)/700)
    def mel_to_hz(m):return 700*(10**(np.asarray(m)/2595)-1)
    hz=mel_to_hz(np.linspace(hz_to_mel(0),hz_to_mel(sr/2),n_mels+2))
    freq=np.linspace(0,sr/2,n_fft//2+1)
    banks=np.zeros((n_mels,len(freq)),dtype=np.float32)
    for i in range(n_mels):
        left,mid,right=hz[i:i+3]
        banks[i]=np.maximum(0,np.minimum((freq-left)/(mid-left+1e-12),(right-freq)/(right-mid+1e-12)))
        banks[i]*=2/(right-left+1e-12)
    return banks

def load_audio(path,sr=16000,channel=0):
    samples,original=sf.read(str(path),dtype='float32',always_2d=True)
    if channel>=samples.shape[1]:raise ValueError(f'channel {channel} unavailable in {path}: {samples.shape[1]} channels')
    audio=samples[:,channel]
    if original!=sr:
        factor=gcd(int(original),int(sr))
        audio=resample_poly(audio,sr//factor,original//factor).astype(np.float32)
    return audio

def audio_windows(audio,cfg=Config()):
    """Always 19 windows from a 10-s recording; short waveforms are padded once."""
    audio=np.asarray(audio,dtype=np.float32)
    if len(audio)<cfg.segment_samples:
        audio=np.pad(audio,(0,cfg.segment_samples-len(audio)))
    count=1+(len(audio)-cfg.segment_samples)//cfg.segment_hop
    for i in range(count):yield audio[i*cfg.segment_hop:i*cfg.segment_hop+cfg.segment_samples]

def logmel(window,cfg=Config()):
    """Centered Hamming STFT, 1024 FFT, 512 hop. Output [128,32]."""
    x=np.asarray(window,dtype=np.float32)
    if len(x)!=cfg.segment_samples:raise ValueError(f'Expected {cfg.segment_samples} samples, found {len(x)}')
    # librosa-compatible centered frame extraction: pad 512 samples each side.
    pad=cfg.n_fft//2
    x=np.pad(x,(pad,pad),mode='constant')
    frames=np.lib.stride_tricks.sliding_window_view(x,cfg.n_fft)[::cfg.hop]
    power=np.abs(rfft(frames*np.hamming(cfg.n_fft).astype(np.float32)[None,:],axis=1))**2
    mel=mel_filters(cfg.sr,cfg.n_fft,cfg.n_mels)@power.T
    feat=np.log(mel+cfg.eps).astype(np.float32)
    # 16k/512 = 31.25 hops -> 32 frames from centered transform.
    if feat.shape[1]<cfg.n_frames:
        feat=np.pad(feat,((0,0),(0,cfg.n_frames-feat.shape[1])))
    return feat[:,:cfg.n_frames].copy()

def load_features(path,cfg=Config(),channel=0):
    return np.stack([logmel(w,cfg) for w in audio_windows(load_audio(path,cfg.sr,channel),cfg)])

class Normalizer:
    def __init__(self,mean=None,std=None):
        self.mean=None if mean is None else np.asarray(mean,dtype=np.float32)
        self.std=None if std is None else np.asarray(std,dtype=np.float32)
    def fit(self,records,cfg=Config(),channel=0):
        # Streaming moments, calculated exclusively over training windows.
        count=0; total=np.zeros(cfg.n_mels,dtype=np.float64); total2=total.copy()
        for rec in records:
            x=load_features(rec['path'],cfg,channel)
            total+=x.sum(axis=(0,2)); total2+=(x.astype(np.float64)**2).sum(axis=(0,2));count+=x.shape[0]*x.shape[2]
        if not count:raise ValueError('Cannot fit normalization on empty training set')
        mean=total/count
        std=np.sqrt(np.maximum(total2/count-mean**2,1e-8))
        self.mean=mean.astype(np.float32); self.std=std.astype(np.float32)
        return self
    def transform(self,x):
        if self.mean is None:raise ValueError('Normalizer not fitted')
        return ((x-self.mean[None,:,None])/self.std[None,:,None]).astype(np.float32)
    def save(self,path):np.savez_compressed(path,mean=self.mean,std=self.std)
    @classmethod
    def load(cls,path):
        z=np.load(path)
        return cls(z['mean'],z['std'])
