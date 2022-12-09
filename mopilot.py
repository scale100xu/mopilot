from typing import Union, Tuple, Any, Callable, Iterator, Set, Optional, overload, TypeVar, Mapping, Dict, List
from torch import  nn
from collections import OrderedDict, namedtuple
from torch.utils.hooks import  RemovableHandle
import torch
from functools import reduce
import json

_grad_t = Union[Tuple[torch.Tensor, ...], torch.Tensor]

class StatTensor(object):
    def __init__(self,data:torch.Tensor, bins:int= 10):
        self.data = data.detach().cpu()
        self.max = None
        self.min = None
        self.hist = None
        self.std = None
        self.mean = None
        self.bins = bins
        self.quantile_config = torch.tensor([0.25, 0.5, 0.75])
        self.quantile = None
        self.calc()

    def calc(self):
        self.count = reduce((lambda x, y: x * y), self.data.shape)
        self.data = self.data.reshape((self.count,))
        self.min = self.data.min().item()
        self.max = self.data.max().item()
        self.std = self.data.std().item()
        self.mean = self.data.mean().item()
        range = (self.min, self.max)
        self.histogram = torch.histogram(self.data, bins=self.bins, range=range, weight=None, density=False, out=None)
        self.quantile = torch.quantile(self.data, self.quantile_config, dim=0)

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        # class NpEncoder(json.JSONEncoder):
        #     def default(self, obj):
        #         if isinstance(obj, np.integer):
        #             return int(obj)
        #         if isinstance(obj, np.floating):
        #             # 👇️ alternatively use str()
        #             return float(obj)
        #         if isinstance(obj, np.ndarray):
        #             return obj.tolist()
        #         return json.JSONEncoder.default(self, obj)

        return json.dumps(self.to_json())

    def to_json(self):
        obj = {
            "count": self.count,
            "min": self.min,
            "max": self.max,
            "std": self.std,
            "mean": self.mean,
            "histogram": {
                "hist": self.histogram.hist.numpy().tolist(),
                "bin_edges": self.histogram.bin_edges.numpy().tolist()
            },
            "quantile": self.quantile.numpy().tolist(),
        }
        return obj

class Mopilot:
    ALL:str = "ALL"
    def __init__(self, model, name=""):
        self.model = model
        self.name = name
        self.model_path_key: Dict[str, nn.Module] = OrderedDict()
        # backward hook removable handle
        self.model_path_register_backward_hook: Dict[str, RemovableHandle] = OrderedDict()
        # forward input hook removable handle
        self.model_path_register_forward_pre_hook: Dict[str, RemovableHandle] = OrderedDict()
        # forward hook removable handle
        self.model_path_register_forward_hook: Dict[str, RemovableHandle] = OrderedDict()
        self.model_input_output_shape_dict: Dict[str, Dict[str, Any]] = OrderedDict()
        # init model_path
        self.init_model_path_key()

    """
    
    """
    def iterate_get_submodules_name(self, parentCount: int, pname: str, model: nn.Module):
        for name, module in model.named_children():
            n_name = f"{pname}{name}."
            prefix = ""
            for i in range(parentCount):
                prefix += "-"
            # print(f"{prefix}:{name},{n_name}")
            self.iterate_get_submodules_name(parentCount + 1, n_name, module)
        pname = f"{pname}{model.__class__.__name__}"
        self.model_path_key[pname] = model

    """
    """
    def init_model_path_key(self):
        self.iterate_get_submodules_name(0, "", self.model)

    def add_register_backward_hook(self, module_path_key:str=ALL,  hook:Callable[['Module', _grad_t, _grad_t], Union[None, torch.Tensor]]=None):
        is_finded = False
        for i, path_key in enumerate(self.model_path_key):
            if path_key == module_path_key or module_path_key==self.ALL:
                module = self.model_path_key[path_key]
                removableHandle = module.register_backward_hook(hook)
                self.model_path_register_backward_hook[path_key] = removableHandle
                is_finded = True
                if module_path_key!=self.ALL:
                   break

        assert True == is_finded, "Not Found module path key"

    def remove_register_backward_hook(self,module_path_key:str=ALL):
        if module_path_key==self.ALL:
            for i, path_key in enumerate(self.model_path_register_backward_hook):
                self.model_path_register_backward_hook[path_key].remove()
                del self.model_path_register_backward_hook[path_key]
        else:
            for i, path_key in enumerate(self.model_path_register_backward_hook):
                if module_path_key == path_key:
                    self.model_path_register_backward_hook[path_key].remove()
                    del self.model_path_register_backward_hook[path_key]


    def add_register_forward_pre_hook(self, module_path_key: str=ALL,
                                    hook: Callable[..., None]=None):
        is_finded = False
        for i, path_key in enumerate(self.model_path_key):
            if path_key == module_path_key or module_path_key==self.ALL:
                module = self.model_path_key[path_key]
                removableHandle = module.register_forward_pre_hook(hook)
                self.model_path_register_forward_pre_hook[path_key] = removableHandle
                is_finded = True
                if module_path_key != self.ALL:
                    break

        assert True == is_finded, "Not Found module path key"


    def remove_register_forward_pre_hook(self,module_path_key:str=ALL):
        if module_path_key==self.ALL:
            for i, path_key in enumerate(self.model_path_register_backward_hook):
                self.model_path_register_forward_pre_hook[path_key].remove()
                del self.model_path_register_forward_pre_hook[path_key]
        else:
            for i, path_key in enumerate(self.model_path_register_backward_hook):
                if module_path_key == path_key:
                    self.model_path_register_forward_pre_hook[path_key].remove()
                    del self.model_path_register_forward_pre_hook[path_key]

    def add_register_forward_hook(self, module_path_key: str=ALL,hook: Callable[..., None]=None):
        is_finded = False
        for i, path_key in enumerate(self.model_path_key):
            if path_key == module_path_key or module_path_key==self.ALL:
                module = self.model_path_key[path_key]
                removableHandle = module.register_forward_hook(hook)
                self.model_path_register_forward_pre_hook[path_key] = removableHandle
                is_finded = True
                if module_path_key!=self.ALL:
                   break

        assert True == is_finded, "Not Found module path key"


    def remove_register_forward_hook(self,module_path_key:str=ALL):
        if module_path_key==self.ALL:
            for i, path_key in enumerate(self.model_path_register_backward_hook):
                self.model_path_register_forward_hook[path_key].remove()
                del self.model_path_register_forward_hook[path_key]
        else:
            for i, path_key in enumerate(self.model_path_register_backward_hook):
                if module_path_key == path_key:
                    self.model_path_register_forward_hook[path_key].remove()
                    del self.model_path_register_forward_hook[path_key]

    def get_model_summary_parameters(self):
        return sum(p.numel() for p in self.model.parameters())

    def hook_model_input_output_shape(self, module, inputs, outputs):
        for i, path_key in enumerate(self.model_path_key):
            if self.model_path_key[path_key]==module:
                self.save_input_output_shape_to_dict(path_key, inputs, outputs)
                break

    def save_input_output_shape_to_dict(self, path_key:str, inputs, outputs):
        input_shape = []
        output_shape = []
        if isinstance(inputs,dict):
            for key in inputs:
                input_shape = input_shape + [tuple(inputs[key].shape)]
        else:
            for input in inputs:
                input_shape = input_shape + [tuple(input.shape)]

        if isinstance(outputs,dict):
            for key in outputs:
                output_shape = output_shape + [tuple(outputs[key].shape)]
        else:
            for output in outputs:
                output_shape = output_shape + [tuple(output.shape)]
        self.model_input_output_shape_dict[path_key] = {"input_shape": input_shape, "output_shape": output_shape}


    """
    model module input and output shape
    final output as:
    OrderedDict([('time_embed.0.Linear', {'input_shape': [(1, 320)], 'output_shape': [(1280,)]})])
    """
    def get_model_input_output_shape(self,**inputs):
        self.add_register_forward_hook(self.ALL, self.hook_model_input_output_shape)
        output = self.model(**inputs)
        self.save_input_output_shape_to_dict(self.name, inputs, output)
        self.remove_register_forward_hook(self.ALL)
        return self.model_input_output_shape_dict

    def print_model_path_key(self):
        for i, path_key in enumerate(self.model_path_key):
            print(f"{path_key}")

    def find_module_key(self,module:nn.Module):
        for i, path_key in enumerate(self.model_path_key):
            if self.model_path_key[path_key]==module:
                return path_key

        assert False==True,"Not found module key"

    def stat_module_weight(self, module_path_key:str=None):
        assert None != module_path_key, "module path key is None"

        result = {}
        for i, path_key in enumerate(self.model_path_key):
            if module_path_key==path_key:
                module = self.model_path_key[module_path_key]
                for name, param in module.named_parameters():
                    result[name] = StatTensor(param)
        return result