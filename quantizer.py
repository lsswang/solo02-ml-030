"""Symmetric fixed-range fake quantizer for the original scalar-scale client."""
import numpy as np
QMIN,QMAX=-127,127

def forward(x, log_scale):
    x=np.asarray(x,float)
    exponent=np.rint(np.asarray(log_scale,float)).astype(int)
    scale=np.exp2(exponent)
    codes=np.rint(np.clip(x/scale,QMIN,QMAX)).astype(np.int64)
    return codes*scale,{'x':x,'log_scale':np.asarray(log_scale,float),'scale':scale,'codes':codes}

def vjp(cache, cotangent):
    cotangent=np.asarray(cotangent,float)
    return {'x':cotangent.copy(),'log_scale':np.zeros_like(cache['log_scale'])}
