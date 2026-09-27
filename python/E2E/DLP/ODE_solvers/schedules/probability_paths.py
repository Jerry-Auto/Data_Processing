"""Probability paths and raw-network parameterization conversions.

Both paths expose a time grid and turn one of several raw neural-network
outputs into the velocity required by a numerical ODE solver.  This keeps the
mathematical model of the path next to its algebraic parameter conversions.
"""
from __future__ import annotations
import math
from typing import Callable, Literal
import torch
from torch import Tensor


class CondOTPath:
    """Straight conditional optimal-transport path, from noise at 0 to data at 1."""
    @staticmethod
    def uniform(num_steps: int, device=None, dtype=torch.float32) -> Tensor:
        return torch.linspace(0., 1., num_steps + 1, device=device, dtype=dtype)
    linear = uniform
    @staticmethod
    def shifted(num_steps: int, seq_len: int = 30, base_seq_len: int = 256,
                base_shift: float = .5, max_shift: float = 1.15,
                max_seq_len: int = 4096, device=None, dtype=torch.float32) -> Tensor:
        slope = (max_shift-base_shift)/(max_seq_len-base_seq_len)
        mu = slope*seq_len + base_shift - slope*base_seq_len
        return CondOTPath.uniform(num_steps, device, dtype).pow(1./math.exp(mu))
    @staticmethod
    def karras(num_steps: int, sigma_min: float = .002, sigma_max: float = 80., rho: float = 7., device=None, dtype=torch.float32) -> Tensor:
        ramp=torch.linspace(0.,1.,num_steps+1,device=device,dtype=dtype); lo,hi=sigma_min**(1/rho),sigma_max**(1/rho)
        sigma=(hi+ramp*(lo-hi)).pow(rho); grid=1.-(sigma-sigma_min)/(sigma_max-sigma_min); grid[0],grid[-1]=0.,1.; return grid
    @staticmethod
    def log_snr(num_steps: int, snr_min: float = -10., snr_max: float = 10., device=None, dtype=torch.float32) -> Tensor:
        grid=torch.sigmoid(torch.linspace(snr_min,snr_max,num_steps+1,device=device,dtype=dtype)/2.); grid[0],grid[-1]=0.,1.; return grid
    @staticmethod
    def times(num_steps: int, schedule: str = 'uniform', **kwargs) -> Tensor:
        try: return getattr(CondOTPath, schedule)(num_steps, **kwargs)
        except AttributeError as exc: raise ValueError('unknown CondOT schedule') from exc
    @staticmethod
    def make(name: str, num_steps: int, **kwargs) -> Tensor:
        """Compatibility order used by earlier CondOT scripts."""
        return CondOTPath.times(num_steps, name, **kwargs)


class CondOTParameterization:
    """Convert CondOT velocity/data/score prediction into ODE velocity."""
    valid_types = {'velocity', 'data', 'score'}
    def __init__(self, predict: Callable[[Tensor, Tensor], Tensor], prediction_type: Literal['velocity','data','score']='velocity', endpoint_eps: float=1e-3):
        if prediction_type not in self.valid_types: raise ValueError(f'prediction_type must be one of {self.valid_types}')
        self.predict,self.prediction_type,self.endpoint_eps=predict,prediction_type,endpoint_eps
    def __call__(self, x: Tensor, t: Tensor) -> Tensor:
        raw=self.predict(x,t)
        if self.prediction_type=='velocity': return raw
        if self.prediction_type=='data': return (raw-x)/(1.-t).clamp_min(self.endpoint_eps)
        return x/t.clamp_min(self.endpoint_eps)+(1.-t)*raw/t.clamp_min(self.endpoint_eps)
    def data_from_velocity(self, x: Tensor, velocity: Tensor, t: Tensor) -> Tensor:
        return x+(1.-t)*velocity


class VPSDEPath:
    def __init__(self, beta_min=.1, beta_max=20.): self.beta_min, self.beta_max = beta_min, beta_max
    def beta(self, t): return self.beta_min + (self.beta_max-self.beta_min)*t
    def log_alpha(self, t): return -.5*self.beta_min*t - .25*(self.beta_max-self.beta_min)*t.square()
    def alpha(self, t): return self.log_alpha(t).exp()
    def sigma(self, t): return torch.sqrt((-torch.expm1(2*self.log_alpha(t))).clamp_min(1e-12))
    def lam(self, t): return self.log_alpha(t)-torch.log(self.sigma(t))
    def inverse_lambda(self, lam):
        temp = 2*(self.beta_max-self.beta_min)*torch.logaddexp(-2*lam, torch.zeros_like(lam))
        return temp/(torch.sqrt(self.beta_min**2+temp)+self.beta_min)/(self.beta_max-self.beta_min)
    def times(self, steps, device, eps=1e-3, schedule='uniform', start=1.):
        if schedule == 'uniform': return torch.linspace(start, eps, steps+1, device=device)
        if schedule == 'logsnr': return self.inverse_lambda(torch.linspace(self.lam(torch.tensor(start,device=device)), self.lam(torch.tensor(eps,device=device)), steps+1, device=device))
        raise ValueError("schedule must be uniform or logsnr")


class VPSDEParameterization:
    """Convert raw VP outputs to epsilon, x0, or probability-flow velocity.

    ``pf_velocity`` is the actual ODE drift dx/dt. ``v`` is retained solely
    for loading old checkpoints; it denotes the DDPM latent-v parameterization.
    """
    def __init__(self, predict, schedule, prediction_type):
        if prediction_type not in {'noise','x_start','pf_velocity','score','v'}: raise ValueError('unknown prediction type')
        self.predict, self.schedule, self.prediction_type = predict, schedule, prediction_type
    def _coeff(self, x,t):
        shape=(1,)* (x.ndim-1); return self.schedule.alpha(t).reshape(-1,*shape) if t.ndim else self.schedule.alpha(t), self.schedule.sigma(t).reshape(-1,*shape) if t.ndim else self.schedule.sigma(t)
    def epsilon(self,x,t):
        raw=self.predict(x,t); a,s=self._coeff(x,t)
        if self.prediction_type=='noise': return raw
        if self.prediction_type=='x_start': return (x-a*raw)/s
        if self.prediction_type=='pf_velocity':
            beta=self.schedule.beta(t)
            while beta.ndim < x.ndim: beta=beta.unsqueeze(-1)
            return s*(2.*raw/beta+x)
        if self.prediction_type=='v': return s*x+a*raw
        return -s*raw
    def x_start(self,x,t):
        raw=self.predict(x,t); a,s=self._coeff(x,t)
        if self.prediction_type=='x_start': return raw
        if self.prediction_type=='noise': return (x-s*raw)/a
        if self.prediction_type=='v': return a*x-s*raw
        return (x+s.square()*raw)/a
    def velocity(self,x,t):
        if self.prediction_type=='pf_velocity': return self.predict(x,t)
        eps=self.epsilon(x,t); beta=self.schedule.beta(t)
        while beta.ndim < x.ndim: beta=beta.unsqueeze(-1)
        return -.5*beta*x + .5*beta*eps/self._coeff(x,t)[1]


# Legacy aliases keep prior commands and notebooks import-compatible while the
# public names above make the two probability paths visually symmetric.
TimeSchedule = CondOTPath
LinearVPSDE = VPSDEPath
VPParameterization = VPSDEParameterization
