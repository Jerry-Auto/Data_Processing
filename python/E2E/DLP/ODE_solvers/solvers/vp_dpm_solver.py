"""Compact hand-written VP DPM-Solver / DPM-Solver++ two-step multistep samplers.

Derived from the original repository's `multistep_dpm_solver_second_update`,
but kept deliberately small for the controlled parameterization experiment.
"""
import torch


class VPDPM2MSolver:
    def __init__(self, model, schedule, algorithm='dpmsolver++'):
        if algorithm not in {'dpmsolver','dpmsolver++'}: raise ValueError('invalid algorithm')
        self.model,self.schedule,self.algorithm,self.nfe=model,schedule,algorithm,0
    def evaluate(self,x,t):
        self.nfe+=1
        return self.model.x_start(x,t) if self.algorithm=='dpmsolver++' else self.model.epsilon(x,t)
    @torch.no_grad()
    def sample(self,x,times):
        self.nfe=0; old_model=old_t=None
        for i,(s,t) in enumerate(zip(times[:-1],times[1:])):
            model_s=self.evaluate(x,s); ls,lt=self.schedule.lam(s),self.schedule.lam(t); h=lt-ls
            a_s,a_t=self.schedule.alpha(s),self.schedule.alpha(t); sig_s,sig_t=self.schedule.sigma(s),self.schedule.sigma(t)
            if i==0:
                if self.algorithm=='dpmsolver++': x=sig_t/sig_s*x-a_t*torch.expm1(-h)*model_s
                else: x=a_t/a_s*x-sig_t*torch.expm1(h)*model_s
            else:
                h0=ls-self.schedule.lam(old_t); d1=(model_s-old_model)/(h0/h)
                if self.algorithm=='dpmsolver++': x=sig_t/sig_s*x-a_t*torch.expm1(-h)*model_s-.5*a_t*torch.expm1(-h)*d1
                else: x=a_t/a_s*x-sig_t*torch.expm1(h)*model_s-.5*sig_t*torch.expm1(h)*d1
            old_model,old_t=model_s,s
        return self.model.x_start(x,times[-1])
