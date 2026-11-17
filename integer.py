"""Original dense-only exporter/runtime; wider protocol is not implemented."""
import copy
import numpy as np

def export(graph):
    result=copy.deepcopy(graph)
    for node in result['nodes']:
        if node['op']=='dense':
            exponent=np.rint(np.asarray(graph['scales'][node['weight_scale']],float)).astype(int)
            node['codes']=np.rint(np.clip(np.asarray(node.pop('weight'))/np.exp2(exponent),-127,127)).astype(int).tolist()
    return result

def run(program, input_codes):
    state={}
    for node in program['nodes']:
        kind=node['op'];nid=node['id']
        if kind=='input':state[nid]=np.asarray(input_codes[node['name']],dtype=np.int64)
        elif kind=='dense':
            sx=np.rint(program['scales'][next(n['scale'] for n in program['nodes'] if n['id']==node['input'])])
            sw=np.rint(np.asarray(program['scales'][node['weight_scale']]))
            so=np.rint(program['scales'][node['scale']])
            accumulator=state[node['input']]@np.asarray(node['codes'],dtype=np.int64)
            state[nid]=np.clip(np.trunc(accumulator*np.exp2(sx+sw-so)),-127,127).astype(np.int64)
        else:raise ValueError('old integer runtime only accepts input/dense')
    return {key:state[key] for key in program['outputs']}
