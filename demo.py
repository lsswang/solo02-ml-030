import json
from pathlib import Path
import numpy as np
import model,integer

def convert(v):
    if isinstance(v,np.ndarray):return v.tolist()
    if isinstance(v,dict):return {k:convert(a) for k,a in v.items()}
    return v

if __name__=='__main__':
    cases=json.loads((Path(__file__).parent/'fixtures/cases.json').read_text())
    for name,case in cases.items():
        try:
            print(json.dumps({'case':name,'fake':convert(model.run(case['graph'],case['float_inputs'])),'integer':convert(integer.run(integer.export(case['graph']),case['input_codes']))}))
        except ValueError as error:print(json.dumps({'case':name,'current_error':str(error)}))
