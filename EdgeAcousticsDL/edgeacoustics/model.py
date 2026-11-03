"""Depthwise CNN + causal residual TCN + additive attention; two task logits."""
import torch
from torch import nn
from torch.nn import functional as F
from .config import Config

class DSBlock(nn.Module):
    def __init__(self,inp,out,stride=(2,1)):
        super().__init__()
        self.seq=nn.Sequential(nn.Conv2d(inp,inp,3,stride=stride,padding=1,groups=inp,bias=False),nn.BatchNorm2d(inp),nn.ReLU(inplace=True),
             nn.Conv2d(inp,out,1,bias=False),nn.BatchNorm2d(out),nn.ReLU(inplace=True))
    def forward(self,x):return self.seq(x)

class CausalConv(nn.Module):
    def __init__(self,ch,k,dilation):
        super().__init__();self.padding=(k-1)*dilation
        self.conv=nn.Conv1d(ch,ch,k,dilation=dilation)
    def forward(self,x):return self.conv(F.pad(x,(self.padding,0)))

class TCNBlock(nn.Module):
    def __init__(self,ch,dilation):
        super().__init__();self.layers=nn.Sequential(CausalConv(ch,3,dilation),nn.ReLU(),CausalConv(ch,3,dilation),nn.ReLU())
    def forward(self,x):return x+self.layers(x)

class TinySoundMTLNet(nn.Module):
    def __init__(self,cfg=Config(),use_tcn=True,use_attention=True,depthwise=True,mode='both'):
        super().__init__();self.cfg=cfg;self.use_attention=use_attention;self.mode=mode
        c0,c1,c2,c3=cfg.channels
        self.stem=nn.Sequential(nn.Conv2d(1,c0,3,padding=1,bias=False),nn.BatchNorm2d(c0),nn.ReLU(inplace=True))
        if depthwise:
            self.encoder=nn.Sequential(DSBlock(c0,c1),DSBlock(c1,c2),DSBlock(c2,c3))
        else:
            def standard(a,b):return nn.Sequential(nn.Conv2d(a,b,3,stride=(2,1),padding=1,bias=False),nn.BatchNorm2d(b),nn.ReLU())
            self.encoder=nn.Sequential(standard(c0,c1),standard(c1,c2),standard(c2,c3))
        self.proj=nn.Conv1d(c3,cfg.hidden,1)
        self.tcn=nn.Sequential(*(TCNBlock(cfg.hidden,d) for d in (1,2,4))) if use_tcn else nn.Identity()
        self.att_linear=nn.Linear(cfg.hidden,cfg.attention_dim)
        self.att_score=nn.Linear(cfg.attention_dim,1,bias=False)
        self.shared=nn.Sequential(nn.Linear(cfg.hidden,cfg.hidden),nn.ReLU(),nn.Dropout(cfg.dropout))
        self.anomaly_head=nn.Linear(cfg.hidden,1)
        self.machine_head=nn.Linear(cfg.hidden,4)
    def forward(self,x):
        x=self.encoder(self.stem(x)) # [B,C,F,T]
        x=x.mean(dim=2) # frequency only, preserve T
        x=self.tcn(self.proj(x)).transpose(1,2) # [B,T,C]
        if self.use_attention:
            att=torch.softmax(self.att_score(torch.tanh(self.att_linear(x))).squeeze(-1),dim=1)
            pooled=torch.sum(x*att.unsqueeze(-1),dim=1)
        else:
            pooled=x.mean(dim=1)
            att=torch.ones_like(x[:,:,0])/x.size(1)
        shared=self.shared(pooled)
        return self.anomaly_head(shared).squeeze(-1),self.machine_head(shared),att

class CNNBaseline(nn.Module):
    def __init__(self,variant='cnn',cfg=Config()):
        super().__init__();self.variant=variant
        self.conv=nn.Sequential(nn.Conv2d(1,32,3,stride=(2,1),padding=1),nn.ReLU(),
           nn.Conv2d(32,64,3,stride=(2,1),padding=1),nn.ReLU(),
           nn.Conv2d(64,96,3,stride=(2,1),padding=1),nn.ReLU())
        if variant in ('lstm','gru'):
            rnn=nn.LSTM if variant=='lstm' else nn.GRU
            self.temporal=rnn(96,96,batch_first=True)
        elif variant=='attention':self.temporal=nn.MultiheadAttention(96,4,batch_first=True)
        self.anomaly_head=nn.Linear(96,1);self.machine_head=nn.Linear(96,4)
    def forward(self,x):
        x=self.conv(x).mean(2).transpose(1,2)
        if self.variant in ('lstm','gru'):x=self.temporal(x)[0]
        elif self.variant=='attention':x=self.temporal(x,x,x,need_weights=False)[0]
        h=x.mean(1)
        return self.anomaly_head(h).squeeze(-1),self.machine_head(h),torch.ones(x.shape[:2],device=x.device)/x.shape[1]

def build_model(name='tiny',cfg=Config(),**kwargs):
    if name=='tiny':return TinySoundMTLNet(cfg,**kwargs)
    if name in ('cnn','tinycnn','lstm','gru','attention'):
        return CNNBaseline('cnn' if name=='tinycnn' else name,cfg)
    if name in ('mobilenetv2','efficientnet_lite'):
        try:
            import torchvision.models as models
            if name=='mobilenetv2':encoder=models.mobilenet_v2(weights=None).features;dim=1280
            else:encoder=models.efficientnet_b0(weights=None).features;dim=1280 # declared surrogate, not EfficientNet-Lite
        except Exception as e:raise RuntimeError('Install compatible torchvision to run mobile comparators') from e
        class VisionAdapter(nn.Module):
            def __init__(self):
                super().__init__();self.encoder=encoder;self.ah=nn.Linear(dim,1);self.mh=nn.Linear(dim,4)
            def forward(self,x):
                y=self.encoder(x.repeat(1,3,1,1)).mean((2,3));return self.ah(y).squeeze(-1),self.mh(y),torch.ones(x.size(0),1,device=x.device)
        return VisionAdapter()
    raise ValueError(f'Unknown model: {name}')
