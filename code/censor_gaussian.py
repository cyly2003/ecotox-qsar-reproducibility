"""Gaussian point/censor likelihood on standardized negative-log concentration.

Qualifiers define supervision only. They must never be prediction features.
One sigma per original task head; C1 and C2 share this exact implementation.
"""
import math
import torch
from torch import nn
from torch.nn import functional as F

def concentration_bounds(value, qualifier, conversion=1.0):
    """Convert a positive concentration to bounds after a positive unit conversion."""
    if not math.isfinite(value) or value<=0 or not math.isfinite(conversion) or conversion<=0:
        raise ValueError('Positive finite concentration and conversion are required')
    q=str(qualifier).strip().replace('≤','<=').replace('≥','>=')
    product=value*conversion
    if not math.isfinite(product) or product<=0:raise ValueError('Converted concentration overflow or underflow')
    y=-math.log10(product)
    if q in ('','=', '=='):return y,y
    if q in ('>','>='):return -math.inf,y
    if q in ('<','<='):return y,math.inf
    raise ValueError('Unknown qualifier; refusing exact-value fallback')

def standardized_bounds(lower,upper,mean,std):
    if not math.isfinite(std) or std<=0 or not math.isfinite(mean):raise ValueError('Scaler mean/std invalid')
    if math.isnan(lower) or math.isnan(upper) or lower>upper:raise ValueError('Bounds invalid')
    return (lower-mean)/std,(upper-mean)/std

def log_difference_exp(a,b):
    """Log(exp(a)-exp(b)), requiring a>b. expm1 retains close-tail precision."""
    if bool((b>=a).any()):raise FloatingPointError('CDF difference vanished; check interval precision')
    return a+torch.log(-torch.expm1(b-a))

def gaussian_nll(mu,sigma,lower,upper,is_exact):
    """Elementwise NLL; float64 tail evaluation avoids float32 CDF cancellation."""
    mu,sigma,lower,upper=[x.to(torch.float64) for x in (mu,sigma,lower,upper)]
    if not (mu.shape==sigma.shape==lower.shape==upper.shape==is_exact.shape):raise ValueError('Shape mismatch')
    if bool((torch.isnan(lower) | torch.isnan(upper)).any()):raise ValueError('NaN bounds forbidden')
    if not bool(torch.isfinite(mu).all() & torch.isfinite(sigma).all() & (sigma>0).all()):raise ValueError('Invalid Gaussian parameters')
    exact=is_exact.bool();censored=~exact
    if bool((exact & (~torch.isfinite(lower) | (lower!=upper))).any()):raise ValueError('Exact point requires equal finite bounds')
    if bool((censored & (lower>=upper)).any()):raise ValueError('Censored bounds must be ordered')
    if bool((censored & ~torch.isfinite(lower) & ~torch.isfinite(upper)).any()):raise ValueError('Unbounded observation has no information')
    result=torch.empty_like(mu)
    result[exact]=.5*((lower[exact]-mu[exact])/sigma[exact]).square()+torch.log(sigma[exact])+.5*math.log(2*math.pi)
    left=censored & torch.isneginf(lower)
    right=censored & torch.isposinf(upper)
    interval=censored & ~left & ~right
    result[left]=-torch.special.log_ndtr((upper[left]-mu[left])/sigma[left])
    result[right]=-torch.special.log_ndtr((mu[right]-lower[right])/sigma[right])
    if bool(interval.any()):
        lo=(lower[interval]-mu[interval])/sigma[interval]
        hi=(upper[interval]-mu[interval])/sigma[interval]
        vals=torch.empty_like(lo)
        positive=lo>0
        # For the right tail use survival probabilities: Phi(-lo)-Phi(-hi).
        vals[positive]=-log_difference_exp(torch.special.log_ndtr(-lo[positive]),torch.special.log_ndtr(-hi[positive]))
        vals[~positive]=-log_difference_exp(torch.special.log_ndtr(hi[~positive]),torch.special.log_ndtr(lo[~positive]))
        result[interval]=vals
    return result

class HeadSigma(nn.Module):
    def __init__(self,heads,sigma_floor=1e-4):
        super().__init__();self.heads=tuple(heads);self.lookup={h:i for i,h in enumerate(heads)}
        self.sigma_floor=float(sigma_floor)
        self.raw=nn.Parameter(torch.full((len(heads),),math.log(math.expm1(1.0-sigma_floor))))
    def forward(self,head_names):
        ids=torch.tensor([self.lookup[h] for h in head_names],device=self.raw.device)
        return F.softplus(self.raw[ids])+self.sigma_floor

def task_balanced_nll(mu,sigma,lower,upper,is_exact,heads,task_weights=None):
    losses=gaussian_nll(mu,sigma,lower,upper,is_exact)
    weights=task_weights or {}
    values=[];normalizer=0.
    for h in sorted(set(heads)):
        mask=torch.tensor([v==h for v in heads],device=mu.device)
        w=float(weights.get(h,1.))
        if not math.isfinite(w) or w<0:raise ValueError('Invalid task weight')
        values.append(w*losses[mask].mean());normalizer+=w
    if not values or normalizer<=0:raise ValueError('No weighted observations')
    return torch.stack(values).sum()/normalizer
