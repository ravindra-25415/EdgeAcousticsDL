"""python -m edgeacoustics.cli <command> --help"""
import argparse,json
from pathlib import Path
from .config import Config

def main(argv=None):
    p=argparse.ArgumentParser(description='EdgeAcousticsDL research pipeline')
    sub=p.add_subparsers(dest='command',required=True)
    def cmd(name,**kwargs):return sub.add_parser(name,**kwargs)
    s=cmd('prepare',help='Index WAV recordings and create leakage-safe split')
    s.add_argument('--data',required=True);s.add_argument('--output',default='data/manifest.csv');s.add_argument('--seed',type=int,default=42)
    s=cmd('train',help='Train and evaluate a model')
    s.add_argument('--manifest',required=True);s.add_argument('--output',required=True);s.add_argument('--config');s.add_argument('--model',default='tiny');s.add_argument('--cache')
    s=cmd('evaluate');s.add_argument('--checkpoint',required=True);s.add_argument('--manifest',required=True);s.add_argument('--output',required=True);s.add_argument('--split',default='test')
    s=cmd('experiment');s.add_argument('--manifest',required=True);s.add_argument('--output',required=True)
    s.add_argument('--kind',choices=['primary','deep_baselines','classical','ablation','lomo','lofo'],default='primary')
    s.add_argument('--config');s.add_argument('--seeds',nargs='+',type=int,default=[42,123,2024,2025,3407]);s.add_argument('--cache')
    s=cmd('compress');s.add_argument('--checkpoint',required=True);s.add_argument('--manifest',required=True);s.add_argument('--output',required=True)
    s.add_argument('--sparsity',type=float,default=.3);s.add_argument('--no-int8',action='store_true')
    s=cmd('replay');s.add_argument('--checkpoint',required=True);s.add_argument('--audio',required=True);s.add_argument('--output',required=True)
    s=cmd('replay-onnx');s.add_argument('--model',required=True);s.add_argument('--normalization',required=True)
    s.add_argument('--audio',required=True);s.add_argument('--output',required=True);s.add_argument('--threshold',type=float,default=.5)
    s=cmd('live');s.add_argument('--checkpoint',required=True);s.add_argument('--output',required=True)
    s=cmd('benchmark');s.add_argument('--checkpoint',required=True);s.add_argument('--output',required=True);s.add_argument('--repeats',type=int,default=100)
    s=cmd('stats');s.add_argument('--summary',required=True);s.add_argument('--output',required=True);s.add_argument('--metric',default='joint_f1')
    args=p.parse_args(argv)
    if args.command=='prepare':
        from .dataset import scan_mimii,stratified_split,save_manifest
        rows=stratified_split(scan_mimii(args.data),args.seed);Path(args.output).parent.mkdir(parents=True,exist_ok=True)
        save_manifest(rows,args.output);result={'rows':len(rows),'splits':{k:sum(r['split']==k for r in rows) for k in ['train','val','test']}}
    elif args.command=='train':
        from .engine import train
        result=train(args.manifest,args.output,Config.load(args.config),args.model,cache=args.cache)
    elif args.command=='evaluate':
        from .engine import evaluate
        result=evaluate(args.checkpoint,args.manifest,args.output,args.split)
    elif args.command=='experiment':
        from .experiments import run_experiment
        result=run_experiment(args.manifest,args.output,args.kind,args.seeds,Config.load(args.config),args.cache)
    elif args.command=='compress':
        from .compress import export
        result=export(args.checkpoint,args.manifest,args.output,args.sparsity,quantize=not args.no_int8)
    elif args.command=='replay':
        from .streaming import replay
        result={'log':replay(args.checkpoint,args.audio,args.output)}
    elif args.command=='replay-onnx':
        from .onnx_edge import replay_onnx
        result={'log':replay_onnx(args.model,args.normalization,args.audio,args.output,args.threshold)}
    elif args.command=='live':
        from .streaming import live
        result=live(args.checkpoint,args.output)
    elif args.command=='benchmark':
        from .benchmark import benchmark
        result=benchmark(args.checkpoint,args.output,args.repeats)
    elif args.command=='stats':
        from .stats import report
        result=report(args.summary,args.output,args.metric)
    print(json.dumps(result,indent=2))
if __name__=='__main__':main()
