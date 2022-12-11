from json import JSONEncoder
from mopilot import StatTensor
import torch
import json


class MySelfJSONEncoder(JSONEncoder):
    def default(self, obj):
        if isinstance(obj, torch.Tensor) and len(torch.Tensor(obj))>0:
            return obj.detach().cpu().numpy().tolist()

        if isinstance(obj, torch.Tensor):
            return obj.cpu().detach().item()

        if isinstance(obj, StatTensor):
           return obj.to_json()

        # if isinstance(obj, np.integer):
        #    return int(obj)
        # if isinstance(obj, np.floating):
        #             # 👇️ alternatively use str()
        #    return float(obj)
        # if isinstance(obj, np.ndarray):
        #             return obj.tolist()

        return json.JSONEncoder.default(self,obj)