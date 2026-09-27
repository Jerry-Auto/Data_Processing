"""Train VP-SDE parameterizations with PF-velocity-equivalent weighted losses."""
import argparse, torch
from pathlib import Path
from model import sample_data, ToyEncoderDecoderFlow
from schedules import VPSDEPath

FORMAT='toy-vpsde-dit-v2-pf-velocity'
LEGACY_FORMAT='toy-vpsde-dit-v1'
def path_for(base, kind):
    p=Path(base); return str(p.with_name(f'{p.stem}_{kind}{p.suffix or ".pt"}'))
def main():
 p=argparse.ArgumentParser(); p.add_argument('--device',default='cuda' if torch.cuda.is_available() else 'cpu'); p.add_argument('--train-steps',type=int,default=3200); p.add_argument('--batch-size',type=int,default=1024); p.add_argument('--checkpoint',default='checkpoints/toy_vpsde.pt'); p.add_argument('--training-endpoint-eps',type=float,default=1e-2); p.add_argument('--weight-clip',type=float,default=100.); p.add_argument('--no-normalize-loss-weight',action='store_true'); a=p.parse_args()
 if not 0.<a.training_endpoint_eps<.5: raise ValueError('training-endpoint-eps must lie in (0, .5)')
 device=torch.device('cuda' if a.device=='gpu' else a.device); sde=VPSDEPath()
 for kind in ('noise','x_start','pf_velocity','score'):
  torch.manual_seed(7); model=ToyEncoderDecoderFlow().to(device); opt=torch.optim.Adam(model.parameters(),lr=2e-3); print(f'\n=== training prediction_type={kind} ===')
  for step in range(a.train_steps):
   data,cond=sample_data(a.batch_size,device); noise=torch.randn_like(data); t=a.training_endpoint_eps+(1.-2.*a.training_endpoint_eps)*torch.rand(a.batch_size,1,device=device); alpha,sigma,beta=sde.alpha(t),sde.sigma(t),sde.beta(t)
   xt=alpha*data+sigma*noise; pf_velocity=-.5*beta*xt+.5*beta*noise/sigma
   target={'noise':noise,'x_start':data,'pf_velocity':pf_velocity,'score':-noise/sigma}[kind]
   weight={'noise':(beta/(2.*sigma)).square(), 'x_start':(beta*alpha/(2.*sigma.square())).square(), 'pf_velocity':torch.ones_like(t), 'score':(beta/2.).square()}[kind]
   weight=weight.clamp_max(a.weight_clip)
   if not a.no_normalize_loss_weight: weight=weight/(weight.mean().detach()+1e-8)
   loss=((model(xt,t,cond)-target).square()*weight).mean(); opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
   if (step+1)%max(1,a.train_steps//4)==0: print(f'train {step+1:5d}/{a.train_steps}: loss={loss.item():.4f}')
  out=Path(path_for(a.checkpoint,kind)); out.parent.mkdir(parents=True,exist_ok=True); torch.save({'format':FORMAT,'model_state_dict':model.state_dict(),'hidden_dim':model.hidden_dim,'condition_dim':model.condition_dim,'depth':model.depth,'num_heads':model.num_heads,'prediction_type':kind,'train_steps':a.train_steps,'loss_space':'pf_velocity_equivalent','training_endpoint_eps':a.training_endpoint_eps,'weight_clip':a.weight_clip,'normalize_loss_weight':not a.no_normalize_loss_weight},out); print(f'checkpoint saved: {out.resolve()}')
if __name__=='__main__': main()
