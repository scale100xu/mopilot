import torch.nn

from . import Mopilot,StatTensor
from collections import OrderedDict
from typing import Any, Dict
import mopilot
import json
import uvicorn
from asyncer import asyncify
from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from starlette.responses import JSONResponse,Response
from mopilot.custom_json_encoder import MySelfJSONEncoder

class SamplerMopilot(Mopilot):
    def __init__(self, model, name):
        super(SamplerMopilot, self).__init__(model,name)
        self.sampler_hook_grad_call_counts: Dict[str,int] = OrderedDict()
        self.sampler_hook_grad_dict: Dict[str,Any] = OrderedDict()

        self.sampler_hook_forward_dict: Dict[str,Any] = OrderedDict()
        self.sampler_hook_forward_call_count: Dict[str,int] = OrderedDict()


    """
    description:
        hook grad input/output function
    param:
        module: module instance
        grad_input: gradent input
        grad_output: gradent output
    return: void
    """
    def sampler_hook_grad(self, module, grad_input, grad_output):
        module_path_key = self.find_module_key(module)
        self.sampler_hook_grad_dict[module_path_key] = {
            "grad_input":grad_input,
            "grad_output":grad_output,
        }

    def get_sampler_hook_grad_data(self, module_path_key:str):
        if module_path_key not in self.sampler_hook_grad_dict:
            assert False == True, "module not hook grad"
        return self.sampler_hook_grad_dict[module_path_key]



    """
    description:
        hook forward input/output function
    param:
        module: module instance
        inputs: inputs
        outputs: outputs
    return: void
    """
    def sampler_hook_forward(self, module, inputs, outputs):
        module_path_key = self.find_module_key(module)
        self.sampler_hook_forward_dict[module_path_key] = {
            "inputs": inputs,
            "outputs": outputs,
        }

    def get_sampler_hook_forward_data(self, module_path_key: str):
        if module_path_key not in self.sampler_hook_forward_dict:
            assert False == True, "module not hook forward"
        return self.sampler_hook_forward_dict[module_path_key]

    def stat_tensor(self, data):
        result = None
        if isinstance(data, tuple):
            result = []
            for d in data:
                if None == d:
                    continue
                if isinstance(d, torch.Tensor):
                    stat = StatTensor(d)
                    result = result + [stat.to_json()]
                else:
                    print(f"stat not support type")
        if isinstance(data, torch.Tensor):
            result = StatTensor(d).to_json()

        return result

    def stat_sampler_module_grad_input(self, module_path_key:str):
        if module_path_key not in self.sampler_hook_grad_dict:
            assert False == True, "module not hook grad"
        grad_input_output = self.sampler_hook_grad_dict[module_path_key]
        return self.stat_tensor(grad_input_output["grad_input"])


    def stat_sampler_module_grad_output(self, module_path_key:str):
        if module_path_key not in self.sampler_hook_grad_dict:
            assert False == True, "module not hook grad"
        grad_input_output = self.sampler_hook_grad_dict[module_path_key]
        return self.stat_tensor(grad_input_output["grad_output"])


    def stat_sampler_module_forward_input(self, module_path_key: str):
        if module_path_key not in self.sampler_hook_grad_dict:
            assert False == True, "module not hook forward"
        input_output = self.sampler_hook_forward_dict[module_path_key]
        return self.stat_tensor(input_output["inputs"])


    def stat_sampler_module_forward_output(self, module_path_key: str):
        if module_path_key not in self.sampler_hook_grad_dict:
            assert False == True, "module not hook forward"
        input_output = self.sampler_hook_forward_dict[module_path_key]
        return self.stat_tensor(input_output["outputs"])

    def http_request(self, threads:int=None)->None:
        tags_metadata = [
            {
                "name": "Model Copilot for pytorch",
                "description": "Model Copilot for pytorch",
                "externalDocs": {
                    "description": "GitHub Source",
                    "url": "https://github.com/scale100xu/mopilot",
                },
            },
        ]
        app = FastAPI(
            title="Mopilot",
            description="Model Copilot for pytorch",
            version= mopilot.VERSION,
            contact={
                "name": "fanghui xu",
                "url": "https://github.com/scale100xu",
                "email": "sunforgetive@gmail.com",
            },
            license_info={
                "name": "MIT License",
                "url": "https://github.com/scale100xu/mopilot/blob/main/LICENSE.txt",
            },
            openapi_tags=tags_metadata,
        )
        app.add_middleware(
            CORSMiddleware,
            allow_credentials=True,
            allow_origins=["*"],
            allow_methods=["*"],
            allow_headers=["*"],
        )

        self.app = app
        def custom_json_encoder(data)->str:
            return json.dumps(data,cls=MySelfJSONEncoder)

        def model_register_module_hook_grad(path:str) -> Response:
            self.add_register_backward_hook(path, self.sampler_hook_grad)
            return Response(content=custom_json_encoder({"status":"success"}), media_type = "application/json")

        def model_register_module_forward_grad(path: str) -> JSONResponse:
            self.add_register_forward_hook(path, self.sampler_hook_forward)
            return Response(content=custom_json_encoder({"status":"success"}), media_type = "application/json")

        def http_get_sampler_hook_grad_data(path: str) -> Response:
            data = self.get_sampler_hook_grad_data(path)
            return Response(content=custom_json_encoder(data), media_type = "application/json")

        def http_get_sampler_hook_forward_data(path: str) -> Response:
            data = self.get_sampler_hook_forward_data(path)
            return Response(content=custom_json_encoder(data), media_type = "application/json")

        def http_stat_sampler_hook_grad_input(path: str) -> Response:
            data = self.stat_sampler_module_grad_input(path)
            return Response(content=custom_json_encoder(data), media_type = "application/json")


        def http_stat_sampler_hook_grad_output(path: str) -> Response:
            # print(f"http grad model: {id(self.model)}")
            data = self.stat_sampler_module_grad_output(path)
            return Response(custom_json_encoder(data), media_type = "application/json")

        def http_stat_sampler_hook_forward_input(path: str) -> Response:
            data = self.stat_sampler_module_forward_input(path)
            return Response(custom_json_encoder(data), media_type = "application/json")

        def http_stat_sampler_hook_forward_output(path: str) -> Response:
            data = self.stat_sampler_module_forward_output(path)
            return Response(custom_json_encoder(data), media_type = "application/json")

        @app.on_event("startup")
        def startup():
            if threads is not None:
                from anyio import CapacityLimiter
                from anyio.lowlevel import RunVar

                RunVar("_default_thread_limiter").set(CapacityLimiter(threads))

        @app.get("/")
        async def root():
            return {"message": "Hello World"}

        @app.get(
            path="/register_module_hook_grad",
            tags=["Register"],
            summary="Register Hook Grad",
            description="Register Hook Grad",
        )
        async def get_register_module_hook_grad(
                key: str =  Query(
                    default=...,
                    description="module path key",
                )
        ):
            print(key)
            return await asyncify(model_register_module_hook_grad)(key)


        @app.get(
            path="/register_module_forward_grad",
            tags=["Register"],
            summary="Register Hook for forward",
            description="Register Hook for forward",
        )
        async def get_register_module_forward_grad(
                key:str = Query(
                    default=...,
                    description="module path key",
                )
        ):
            return await asyncify(model_register_module_forward_grad)(key)

        @app.get(
            path="/get_module_hook_grad_data",
            tags=["GetData"],
            summary="get Hook for grad input and output data",
            description="get Hook for grad input and output data",
        )
        async def get_module_hook_grad_data(
                key:str = Query(
                    default=...,
                    description="module path key",
                )
        ):
            return await asyncify(http_get_sampler_hook_grad_data)(key)


        @app.get(
            path="/get_module_hook_forward_data",
            tags=["GetData"],
            summary="get Hook for forward input and output data",
            description="get Hook for forward input and output data",
        )
        async def get_module_hook_forward_data(
                key:str = Query(
                    default=...,
                    description="module path key",
                )
        ):
            return await asyncify(http_get_sampler_hook_forward_data)(key)




        @app.get(
            path="/stat_module_hook_forward_input",
            tags=["StatData"],
            summary="stat Hook for forward input  data",
            description="stat Hook for forward input data",
        )
        async def stat_module_hook_forward_input(
                key:str = Query(
                    default=...,
                    description="module path key",
                )
        ):
            return await asyncify(http_stat_sampler_hook_forward_input)(key)



        @app.get(
            path="/stat_module_hook_forward_output",
            tags=["StatData"],
            summary="stat Hook for forward input and output data",
            description="stat Hook for forward input and output data",
        )
        async def stat_module_hook_forward_output(
                key:str = Query(
                    default=...,
                    description="module path key",
                )
        ):
            return await asyncify(http_stat_sampler_hook_forward_output)(key)


        @app.get(
            path="/stat_module_hook_grad_input",
            tags=["StatData"],
            summary="stat Hook for forward input data",
            description="stat Hook for forward input data",
        )
        async def stat_module_hook_grad_input(
                key:str = Query(
                    default=...,
                    description="module path key",
                )
        ):
            return await asyncify(http_stat_sampler_hook_grad_input)(key)


        @app.get(
            path="/stat_module_hook_grad_output",
            tags=["StatData"],
            summary="stat Hook for grad  output data",
            description="stat Hook for grad output data",
        )
        async def stat_module_hook_grad_output(
                key:str = Query(
                    default=...,
                    description="module path key",
                )
        ):
            return await asyncify(http_stat_sampler_hook_grad_output)(key)


    def run_http_server(self, host:str="0.0.0.0", port:int=8765, log_level: str="debug")->None:
        uvicorn.run(self.app, host=host, port=port, log_level=log_level)



