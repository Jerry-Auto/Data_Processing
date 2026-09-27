"""Complete CondOT experiment: all parameterizations, solvers, statistics and samples."""
import argparse,csv,subprocess,sys,time
from pathlib import Path
import os
os.environ.setdefault('MPLCONFIGDIR','/tmp/matplotlib')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch
from torch.nn import functional as F
from benchmark.benchmark import load_checkpoint,run_inference,synchronize
from model import CENTERS,conditional_gaussian_metrics
from schedules import CondOTPath
from reports import save_table_image

KINDS=('velocity','data','score'); METHODS=('euler','midpoint','heun','rk4','ab2','condot_2m','condot_2m_full_endpoint','condot_lambda_exact','condot_lambda_dpmpp')
DISPLAY={m:m.replace('_',' ') for m in METHODS}
def paths(base):
 p=Path(base); return {k:p.with_name(f'{p.stem}_{k}{p.suffix or ".pt"}') for k in KINDS}

def plot_summary(rows, output):
 """Match the VP report: numerical-cost Pareto view plus target-quality view."""
 fig,axes=plt.subplots(1,2,figsize=(13,5)); colors=plt.get_cmap('tab10').colors; markers={'velocity':'o','data':'s','score':'^'}
 for i,method in enumerate(METHODS):
  subset=[r for r in rows if r['method']==method]
  axes[0].scatter([r['latency_ms'] for r in subset],[r['reference_mse'] for r in subset],color=colors[i],label=DISPLAY[method],s=45)
 for r in rows: axes[0].scatter(r['latency_ms'],r['reference_mse'],marker=markers[r['prediction']],color=colors[METHODS.index(r['method'])],s=45)
 axes[0].set(xlabel='Mean inference latency per batch (ms)',ylabel='MSE vs own RK4 reference',yscale='log',title='Numerical error / latency trade-off'); axes[0].grid(alpha=.25); axes[0].legend(fontsize=7,ncol=2)
 for i,kind in enumerate(KINDS):
  subset=[r for r in rows if r['prediction']==kind]; xs=torch.arange(len(METHODS)).numpy()+(i-1)*.24
  axes[1].bar(xs,[r['w2'] for r in subset],width=.24,label=kind)
 axes[1].set_xticks(range(len(METHODS)),[DISPLAY[m].replace(' ','\n') for m in METHODS],fontsize=7)
 axes[1].set(ylabel='Gaussian Wasserstein-2 distance to true conditional target',title='Sample-distribution quality',yscale='log'); axes[1].grid(axis='y',alpha=.25); axes[1].legend()
 fig.tight_layout(); fig.savefig(output,dpi=180); plt.close(fig)
def main():
 p=argparse.ArgumentParser(); p.add_argument('--device',default='cuda' if torch.cuda.is_available() else 'cpu'); p.add_argument('--checkpoint-base',default='checkpoints/toy_condot.pt'); p.add_argument('--output-dir',default='outputs/condot_all'); p.add_argument('--batch-size',type=int,default=1024); p.add_argument('--steps',type=int,default=30); p.add_argument('--reference-steps',type=int,default=512); p.add_argument('--repeats',type=int,default=100); p.add_argument('--schedule',default='uniform',choices=['uniform','linear','shifted','karras','log_snr']); p.add_argument('--endpoint-eps',type=float,default=None); a=p.parse_args()
 d=torch.device('cuda' if a.device=='gpu' else a.device); root=Path(__file__).resolve().parents[1]; out=(root/a.output_dir).resolve(); out.mkdir(parents=True,exist_ok=True); ckpts=paths(a.checkpoint_base)
 if not all(x.exists() for x in ckpts.values()): raise FileNotFoundError('CondOT checkpoints missing; run train_all.py or train.py --prediction-type all first.')
 torch.manual_seed(7); labels=(torch.arange(a.batch_size,device=d)%4)[torch.randperm(a.batch_size,device=d)]; cond=F.one_hot(labels,4).float(); x=torch.randn(a.batch_size,2,device=d); times=CondOTPath.times(a.steps,a.schedule,device=d); ref_times=CondOTPath.times(a.reference_steps,a.schedule,device=d); rows=[]
 for kind,path in ckpts.items():
  model,saved=load_checkpoint(str(path),d); model.eval(); endpoint_eps=model.training_endpoint_eps if a.endpoint_eps is None else a.endpoint_eps; ref,*_=run_inference(model,saved,'rk4',x,cond,ref_times,endpoint_eps)
  for method in METHODS:
   run_inference(model,saved,method,x,cond,times,endpoint_eps); synchronize(d); elapsed=[]
   for _ in range(a.repeats):
    start=time.perf_counter(); sample,nfe,_,_=run_inference(model,saved,method,x,cond,times,endpoint_eps); synchronize(d); elapsed.append((time.perf_counter()-start)*1e3)
   q=conditional_gaussian_metrics(sample,labels); rows.append(dict(prediction=kind,method=method,intervals=a.steps,nfe=nfe,latency_ms=sum(elapsed)/len(elapsed),reference_mse=(sample-ref).square().mean().item(),**q)); print(kind,method,rows[-1]['reference_mse'],q['w2'])
 fields=list(rows[0]);
 with (out/'results.csv').open('w',newline='') as f: w=csv.DictWriter(f,fieldnames=fields); w.writeheader(); w.writerows(rows)
 (out/'results.md').write_text('# CondOT results\n\n'+ '\n'.join(str(r) for r in rows)+'\n'); save_table_image(rows,out/'table.png','CondOT: solver cost and quality',METHODS,DISPLAY); plot_summary(rows,out/'summary.png')
 visual_cmd=[sys.executable,str(root/'visualization'/'visualize_solvers.py'),'--device',a.device,'--velocity-checkpoint',str(ckpts['velocity']),'--data-checkpoint',str(ckpts['data']),'--score-checkpoint',str(ckpts['score']),'--samples',str(a.batch_size),'--steps',str(a.steps),'--reference-steps',str(a.reference_steps),'--schedule',a.schedule,'--output',str(out/'samples.png')]
 if a.endpoint_eps is not None: visual_cmd += ['--endpoint-eps',str(a.endpoint_eps)]
 subprocess.run(visual_cmd,check=True)
 print('Complete:',out)
if __name__=='__main__': main()
