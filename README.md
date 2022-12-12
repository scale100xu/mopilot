# mopilot

mopilot is a PyTorch helper for models, which is a wrapper of the hook functions in torch.nn.module. It enhances the hook functions and makes them easier and more elegant to use. All functions are based on the module path key in the model. 
The module path key is a string combination just like the dictionary key in a pt file, and can be regarded as the coordinates of a certain module. Its composition is {Model Name}. {The index of the module if there is none, then ignore}. {Module class name}. It can have multiple levels, and each level represents a node in the model.

Below is a list of functions that can be achieved with this module:
1. Print module information.
2. Print information about inputs, outputs, gradient inputs, and gradient outputs of the module.
3. Inject modules to modify/replace the inputs and outputs of the module.
4. Inject modules to modify/replace the gradient inputs of the module (equivalent to replacing the gradient algorithm).
5. More complex injection can be achieved (such as implementing teacher/student models).
6. Further, you can implement chaos tests on the model.

And you can do even more interesting things!!!

# install
```shell
pip install -i https://test.pypi.org/simple/ mopilot
```

# examples
You can check the files in the test directory, including: 
1. print_module_key_test.py which prints the keys of modules and simulates the input to print the input and output dimensions of the modules; 
2. hook_print_input_output_test.py which prints the inputs and outputs of the modules; 
3. hook_print_grad_input_output_test.py which prints the inputs and outputs of gradients after the module backward; 
4. hook_forward_input_scale_test.py which scales the inputs of each module; 
5. stat_module_weight_test.py which counts and prints the weights of the module with the key seq.0.Conv2d; 
6. stat_tensor_test.py which tests the StatTensor class; 
7. http_sampler_mopilot_test.py which tests the sampling class of the model.



## print_module_key_test.py code

```python
import torch

from mopilot import Mopilot
import json
from test.test_model import TestModel

m = TestModel()

p = Mopilot(m, name="printModuleKey")
# print module path key
print(f"{json.dumps(list(p.model_path_key.keys()), indent=4)}")
"""
output:
[
    "seq.0.Conv2d",
    "seq.1.Conv2d",
    "seq.2.Conv2d",
    "seq.3.Conv2d",
    "seq.ModuleList",
    "TestModel"
]
"""

x = torch.randn(1, 20, 5, 5)
model_input_output_shape = p.get_model_input_output_shape(x=x)

print(f"{json.dumps(model_input_output_shape, indent=4)}")

"""
output:
{
    "seq.0.Conv2d": {
        "input_shape": [
            [
                1,
                20,
                5,
                5
            ]
        ],
        "output_shape": [
            [
                20,
                5,
                5
            ]
        ]
    },
    "seq.1.Conv2d": {
        "input_shape": [
            [
                1,
                20,
                5,
                5
            ]
        ],
        "output_shape": [
            [
                40,
                5,
                5
            ]
        ]
    },
    "seq.2.Conv2d": {
        "input_shape": [
            [
                1,
                40,
                5,
                5
            ]
        ],
        "output_shape": [
            [
                40,
                5,
                5
            ]
        ]
    },
    "seq.3.Conv2d": {
        "input_shape": [
            [
                1,
                40,
                5,
                5
            ]
        ],
        "output_shape": [
            [
                20,
                5,
                5
            ]
        ]
    },
    "TestModel": {
        "input_shape": [],
        "output_shape": [
            [
                20,
                5,
                5
            ]
        ]
    },
    "printModuleKey": {
        "input_shape": [
            [
                1,
                20,
                5,
                5
            ]
        ],
        "output_shape": [
            [
                20,
                5,
                5
            ]
        ]
    }
}
"""
```

## print_module_key_test.py code

```python
import torch,time

from test.test_model import  TestModel
from mopilot.sampler_mopilot import SamplerMopilot
import threading

class HttpThread(threading.Thread):
    def __init__(self, sampler:SamplerMopilot):
        threading.Thread.__init__(self)
        self.sampler = sampler
        self.threadLock = threading.Lock()


    def run(self) -> None:
        self.threadLock.acquire()
        self.sampler.http_request()
        self.sampler.run_http_server()
        self.threadLock.release()



if __name__ == "__main__":

    m = TestModel()
    m.train(True)

    http_sampler_mopilot = SamplerMopilot(m, "HttpSamplerMopilot")
    def todo_train():
        # todo your train code
        # path = "seq.0.Conv2d"
        # http_sampler_mopilot = SamplerMopilot(m, "HttpSamplerMopilot")
        # http_sampler_mopilot.add_register_backward_hook(path, http_sampler_mopilot.sampler_hook_grad)
        # http_sampler_mopilot.add_register_backward_hook(path, http_sampler_mopilot.sampler_hook_forward)
        for i in range(1000):
            print(f"run: {i}")
            x = torch.randn(1, 20, 5, 5)
            y = m(x)
            loss = torch.nn.functional.mse_loss(x.float(), y.float(), reduction="mean")
            loss.backward()
            # grad_data = http_sampler_mopilot.get_sampler_hook_grad_data(path)
            # forward_data = http_sampler_mopilot.get_sampler_hook_forward_data(path)
            time.sleep(1)


    # run program
    http = HttpThread(http_sampler_mopilot)
    http.daemon = True
    http.start()
    todo_train()



"""
# add grad sample
curl -XGET "http://0.0.0.0:8765/register_module_hook_grad?key=seq.0.Conv2d"

# add forward sample
curl -XGET "http://0.0.0.0:8765/register_module_forward_grad?key=seq.0.Conv2d"

# get sample grad data
curl -XGET "http://0.0.0.0:8765/get_module_hook_grad_data?key=seq.0.Conv2d"

# get sample forward data
curl -XGET "http://0.0.0.0:8765/get_module_hook_forward_data?key=seq.0.Conv2d"

# stat sampler forward input
curl -XGET "http://0.0.0.0:8765/stat_module_hook_forward_input?key=seq.0.Conv2d"

# stat sampler forward output
curl -XGET "http://0.0.0.0:8765/stat_module_hook_forward_output?key=seq.0.Conv2d"

# stat sampler grad input
curl -XGET "http://0.0.0.0:8765/stat_module_hook_grad_input?key=seq.0.Conv2d"

# stat sampler grad output
curl -XGET "http://0.0.0.0:8765/stat_module_hook_grad_output?key=seq.0.Conv2d"
"""

```