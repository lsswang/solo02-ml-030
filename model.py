"""Floating fake-quant execution. Parameters are ordinary dense matrices."""
import numpy as np
from quantizer import forward

def run(graph, values):
    state={}
    def scale(sid):return graph['scales'][sid]
    for node in graph['nodes']:
        kind=node['op'];nid=node['id']
        if kind=='input':state[nid]=forward(values[node['name']],scale(node['scale']))[0]
        elif kind=='dense':
            parent=state[node['input']]
            weight=forward(node['weight'],scale(node['weight_scale']))[0]
            # Ordinary floating bias: old single-layer fixture has exactly representable zero bias.
            y=parent@weight+np.asarray(node['bias'],float)
            state[nid]=forward(y,scale(node['scale']))[0]
        elif kind=='relu':state[nid]=np.maximum(state[node['input']],0)
        elif kind=='add':state[nid]=forward(state[node['left']]+state[node['right']],scale(node['scale']))[0]
        else:raise ValueError(kind)
    return {key:state[key] for key in graph['outputs']}
