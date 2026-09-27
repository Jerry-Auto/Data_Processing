"""Benchmark VP-SDE solvers from one of the four train_vp.py checkpoints."""
import argparse,time,torch
from pathlib import Path
from model import sample_data, ToyEncoderDecoderFlow, conditional_gaussian_metrics
from train.train_vp import FORMAT, LEGACY_FORMAT
from schedules import VPSDEPath
from solvers import VPSDESolver
def load(path,device):
 d=torch.load(path,map_location=device,weights_only=True); assert d['format'] in {FORMAT,LEGACY_FORMAT}; m=ToyEncoderDecoderFlow(d['condition_dim'],d['hidden_dim'],d['depth'],d['num_heads']).to(device); m.load_state_dict(d['model_state_dict']); m.training_endpoint_eps=d.get('training_endpoint_eps',1e-3); return m,d['prediction_type']
@torch.no_grad()
def run(model,kind,method,x,cond,times,sde):
 model.reset_inference_counters(); ctx=model.encode(cond); eps=model.training_endpoint_eps; solver=VPSDESolver(lambda z,t:model.decode(z,t.clamp(eps,1.-eps),ctx),sde,method,kind); out=solver.sample(x,times)
 return out,model.decoder_calls
def main():
 p=argparse.ArgumentParser(); p.add_argument('--checkpoint',required=True); p.add_argument('--device',default='cuda' if torch.cuda.is_available() else 'cpu'); p.add_argument('--steps',type=int,default=30); p.add_argument('--repeats',type=int,default=50); p.add_argument('--batch-size',type=int,default=1024); p.add_argument('--schedule',choices=['uniform','logsnr'],default='logsnr'); p.add_argument('--endpoint-eps',type=float,default=None); a=p.parse_args(); device=torch.device('cuda' if a.device=='gpu' else a.device); torch.manual_seed(7); model,kind=load(a.checkpoint,device); model.eval(); eps=model.training_endpoint_eps if a.endpoint_eps is None else a.endpoint_eps; sde=VPSDEPath(); _,cond=sample_data(a.batch_size,device); labels=cond.argmax(dim=1); x=torch.randn(a.batch_size,2,device=device); times=sde.times(a.steps,device,eps=eps,schedule=a.schedule); ref,_=run(model,kind,'rk4',x,cond,sde.times(512,device,eps=eps,schedule=a.schedule),sde)
 print(f'VP-SDE benchmark: prediction={kind}, intervals={a.steps}, schedule={a.schedule}'); print('method             NFE    ms       MSE vs RK4(512)     Gaussian W2   mean/cov err')
 for method in ('euler','midpoint','heun','rk4','ab2','dpmsolver_2m','dpmsolverpp_2m'):
  run(model,kind,method,x,cond,times,sde); 
  if device.type=='cuda': torch.cuda.synchronize()
  start=time.perf_counter()
  for _ in range(a.repeats): out,nfe=run(model,kind,method,x,cond,times,sde)
  if device.type=='cuda': torch.cuda.synchronize()
  q=conditional_gaussian_metrics(out,labels); print(f'{method:18s} {nfe:3d} {(time.perf_counter()-start)*1e3/a.repeats:8.3f} {(out-ref).square().mean().item():.6e} {q["w2"]:13.6f} {q["mean_error"]:.3f}/{q["covariance_error"]:.3f}')
if __name__=='__main__': main()
