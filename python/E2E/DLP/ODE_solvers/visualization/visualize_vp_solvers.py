"""Scatter-plot VP-SDE samples for four parameterizations and all local solvers."""
import argparse, os
from pathlib import Path
os.environ.setdefault('MPLCONFIGDIR','/tmp/matplotlib')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch
from torch.nn import functional as F
from model import CENTERS, conditional_gaussian_metrics
from benchmark.benchmark_vp import load, run
from schedules import LinearVPSDE

METHODS=('euler','midpoint','heun','rk4','ab2','dpmsolver_2m','dpmsolverpp_2m')
LABELS={'euler':'Euler','midpoint':'Midpoint','heun':'Heun','rk4':'RK4','ab2':'AB2','dpmsolver_2m':'DPM-Solver 2M','dpmsolverpp_2m':'DPM-Solver++ 2M'}
COLORS=('#e15759','#4e79a7','#59a14f','#b07aa1')

def inputs(n,device):
 labels=(torch.arange(n,device=device)%4)[torch.randperm(n,device=device)]; centers=torch.tensor(CENTERS,device=device)
 return labels,F.one_hot(labels,4).float(),centers[labels]+.25*torch.randn(n,2,device=device),torch.randn(n,2,device=device)
def draw(ax,x,labels,limit):
 x=x.detach().float().cpu(); labels=labels.cpu(); finite=torch.isfinite(x).all(-1)
 for k,c in enumerate(COLORS):
  y=x[(labels==k)&finite]; ax.scatter(y[:,0],y[:,1],s=3,alpha=.35,color=c,linewidths=0)
 ax.set(xlim=(-limit,limit),ylim=(-limit,limit),xticks=(),yticks=()); ax.set_aspect('equal')
@torch.no_grad()
def main():
 p=argparse.ArgumentParser(); p.add_argument('--noise-checkpoint',default='checkpoints/toy_vpsde_noise.pt'); p.add_argument('--x-start-checkpoint',default='checkpoints/toy_vpsde_x_start.pt'); p.add_argument('--pf-velocity-checkpoint',default='checkpoints/toy_vpsde_pf_velocity.pt'); p.add_argument('--score-checkpoint',default='checkpoints/toy_vpsde_score.pt'); p.add_argument('--device',default='cuda' if torch.cuda.is_available() else 'cpu'); p.add_argument('--samples',type=int,default=2048); p.add_argument('--steps',type=int,default=30); p.add_argument('--reference-steps',type=int,default=512); p.add_argument('--schedule',choices=['uniform','logsnr'],default='logsnr'); p.add_argument('--endpoint-eps',type=float,default=1e-2); p.add_argument('--limit',type=float,default=4.); p.add_argument('--output',default='outputs/vp_solver_samples.png'); a=p.parse_args()
 device=torch.device('cuda' if a.device=='gpu' else a.device); torch.manual_seed(7); labels,cond,target,xinit=inputs(a.samples,device); sde=LinearVPSDE(); times=sde.times(a.steps,device,eps=a.endpoint_eps,schedule=a.schedule); ref_times=sde.times(a.reference_steps,device,eps=a.endpoint_eps,schedule=a.schedule)
 checkpoints=(('noise',a.noise_checkpoint),('x_start',a.x_start_checkpoint),('pf_velocity',a.pf_velocity_checkpoint),('score',a.score_checkpoint)); fig,axes=plt.subplots(4,8,figsize=(17,8),squeeze=False)
 for row,(name,path) in enumerate(checkpoints):
  model,kind=load(path,device); model.eval(); reference,_=run(model,kind,'rk4',xinit,cond,ref_times,sde)
  draw(axes[row,0],target,labels,a.limit); axes[row,0].set_title('Target',fontsize=8); axes[row,0].set_ylabel(name,fontsize=10)
  for col,method in enumerate(METHODS,1):
   sample,nfe=run(model,kind,method,xinit,cond,times,sde); quality=conditional_gaussian_metrics(sample,labels); mse=(sample-reference).square().mean().item(); draw(axes[row,col],sample,labels,a.limit); axes[row,col].set_title(f'{LABELS[method]}\nNFE={nfe}, ref={mse:.1e}\nW2={quality["w2"]:.2f}, std={quality["component_std"]:.2f}',fontsize=7)
 fig.suptitle(f'VP-SDE: {a.steps} intervals, {a.schedule}; target std=0.25',fontsize=13); fig.tight_layout(rect=(0,0,1,.94)); out=Path(a.output); out.parent.mkdir(parents=True,exist_ok=True); fig.savefig(out,dpi=180,bbox_inches='tight'); print(f'figure saved: {out.resolve()}')
if __name__=='__main__': main()
