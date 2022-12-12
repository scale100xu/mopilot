import torch.nn

from . import SamplerMopilot
from typing import Any, Dict
from collections import OrderedDict
from torch.nn import functional as F
from asyncer import asyncify
from fastapi import Query
from starlette.responses import JSONResponse


class ScaleModule(SamplerMopilot):
    def __init__(self, model, name):
        super(ScaleModule, self).__init__(model,name)
        self.scale_module_dict: Dict[str, Any] = OrderedDict()


    def add_http_support(self):
        @self.app.get(
            path="/add_scale_module",
            tags=["AddData"],
            summary="add module scale",
            description="add module scale",
        )
        async def add_http_scale_module(
                key:str = Query(
                    default=...,
                    description="module path key",
                ),
                out_dim: int = Query(
                    default=0,
                    description="module out_features value in torch.nn.Linear;module out_channels value in torch.nn.Conv2d",
                )
        ):
            return await asyncify(self.add_scale_module)(key, out_dim)


    def add_scale_module(self, module_path_key:str, out_dim:int=0)->JSONResponse:
        if module_path_key not in self.model_path_key:
            assert False==True, 'Module not found!'
        module = self.model_path_key[module_path_key]
        if isinstance(module, torch.nn.Linear):
            # scale param , is only out_features
            output_dim = None
            for name, param in module.named_parameters():
                if name == "weight":
                   output_dim = param.shape[0]
                   param = F.interpolate(param.T.unsqueeze(0), size=[out_dim], mode="nearest")
                   param = param.reshape(*param.shape[1:]).T
                   module.weight = torch.nn.Parameter(param)
                   module.out_features = out_dim
                elif name == "bias":
                   param = F.interpolate(param.unsqueeze(0).unsqueeze(0), size=[out_dim], mode="nearest")
                   param = param.reshape(*param.shape[2:])
                   module.bias = torch.nn.Parameter(param)


        elif isinstance(module, torch.nn.Conv2d):
            # scale param , is only out_channels
            output_dim = None
            for name, param in module.named_parameters():
                if name == "weight":
                    origin_shape = param.shape
                    output_dim = origin_shape[0]
                    param = param.permute(1, 0, 2, 3)
                    param = F.interpolate(param.unsqueeze(0), size=[out_dim, origin_shape[2], origin_shape[3]], mode="nearest")
                    param = param.reshape(*param.shape[1:]).permute(1, 0, 2, 3)
                    module.weight = torch.nn.Parameter(param)
                    module.out_channels = out_dim
                elif name == "bias":
                    param = F.interpolate(param.unsqueeze(0).unsqueeze(0), size=[out_dim], mode="nearest")
                    param = param.reshape(*param.shape[2:])
                    module.bias = torch.nn.Parameter(param)
        else:
            assert False == True, 'Module not scale!'
        self.scale_module_dict[module_path_key] = {"input_dim": out_dim, "output_dim": output_dim}
        module.register_forward_hook(self.hook_moduel_forward_output)

        return {"status":"success"}


    def hook_moduel_forward_output(self, module, inputs, outputs):
        module_path_key = self.find_module_key(module)

        re = []
        output_dim = self.scale_module_dict[module_path_key]["output_dim"]
        if isinstance(module, torch.nn.Linear):
            # inverse scale output
            if isinstance(outputs, tuple):
                for i, output in enumerate(outputs):
                    if None == output:
                        re = re + [output]
                        continue
                    else:
                        if len(outputs.shape) == 2:
                            output = outputs.unsqueeze(0)
                            _output = F.interpolate(output, size=[output_dim], mode="nearest")
                            _output = _output.reshape(*_output.shape[1:])
                            re = re + [_output]
                        elif len(outputs.shape) < 2:
                            assert False == True, 'torch.nn.Linear Not Support <2'
                        elif len(outputs.shape) == 3:
                            _output = F.interpolate(outputs, size=[output_dim], mode="nearest")
                            re = re + [_output]
                        elif len(outputs.shape) == 4:
                            shape = outputs.shape
                            _output = outputs.reshape(shape[0] * shape[1], shape[2], shape[3])
                            _output = F.interpolate(_output, size=[output_dim], mode="nearest")
                            _output =  _output.reshape(shape[0], shape[1], shape[2], output_dim)
                            re = re + [_output]
                        else:
                            assert False == True, 'torch.nn.Linear Not Support > 4'
            else:
                if len(outputs.shape) == 2:
                    output = outputs.unsqueeze(0)
                    _output = F.interpolate(output, size=[output_dim], mode="nearest")
                    _output = _output.reshape(*_output.shape[1:])
                    return _output
                elif len(outputs.shape) < 2:
                    assert False == True, 'torch.nn.Linear Not Support <2'
                elif len(outputs.shape) == 3:
                    _output = F.interpolate(outputs, size=[output_dim], mode="nearest")
                    return _output
                elif len(outputs.shape) == 4:
                    shape = outputs.shape
                    _output = outputs.reshape(shape[0]*shape[1], shape[2], shape[3])
                    _output = F.interpolate(_output, size=[output_dim], mode="nearest")
                    return _output.reshape(shape[0], shape[1], shape[2], output_dim)
                else:
                    assert False == True, 'torch.nn.Linear Not Support > 4'
        elif isinstance(module, torch.nn.Conv2d):
            if isinstance(outputs, tuple):
                for i, output in enumerate(outputs):
                    if None == output:
                        re = re + [output]
                        continue
                    else:
                        if len(outputs.shape) <= 2:
                            assert False == True, 'torch.nn.Conv2d Not Support <=2'
                        elif len(outputs.shape) == 3:
                            _output = F.interpolate(outputs.unsqueeze(0), size=[output_dim, outputs.shape[2]],
                                                    mode="nearest")
                            _output = _output.reshape(_output.shape[1], output_dim, outputs.shape[2])
                            re = re + [_output]
                        elif len(outputs.shape) == 4:
                            shape = outputs.shape
                            _output = F.interpolate(outputs, size=[output_dim, shape[2], shape[3]], mode="nearest")
                            _output = _output.reshape(1*shape[0], output_dim, shape[2], shape[3])
                            re = re + [_output]
                        else:
                            assert False == True, 'torch.nn.Conv2d Not Support > 4'
            else:
                if len(outputs.shape) <= 2:
                    assert False == True, 'torch.nn.Conv2d Not Support <=2'
                elif len(outputs.shape) == 3:
                    _output = F.interpolate(outputs.unsqueeze(0), size=[output_dim,outputs.shape[2]], mode="nearest")
                    return _output.reshape(_output.shape[1], output_dim, outputs.shape[2])
                elif len(outputs.shape) == 4:
                    shape = outputs.shape
                    _output = F.interpolate(outputs.unsqueeze(0), size=[output_dim, shape[2], shape[3]], mode="nearest")
                    return _output.reshape(1*shape[0], output_dim, shape[2], shape[3])
                else:
                    assert False == True, 'torch.nn.Conv2d Not Support > 4'
        else:
            assert False == True, 'Module not scale!'

        re = tuple(re)

        return re

