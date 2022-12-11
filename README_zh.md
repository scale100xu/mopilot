## mopilot

mopilot是pytorch的model辅助器,它是对torch.nn.module中hook函数的封装，它增强了hook函数并使其使用更简单且更优雅，所有的函数都基于model中的module路径key操作，
module路径key 是一个字符串组合与pt文件的字典key一样，可以认为是某个module的坐标，它的组成为{模型名称}.{module所在的索取值，如果没有表示忽视}.{module类名称} 可以有多级，每一级代表model中的一个节点。

下面依赖此模块可以实现的功能列表
1. 打印 module 信息
2. 打印module的输入，输出，梯度的输入，梯度的输出信息
3. 注入module更改/替换module的输入和输出
4. 注入module更改/替换module的梯度输入(相当于替换 梯度 算法)
5. 可以更加复杂的注入(比如实现 teacher/student 模型)
6. 进一步你可以实现chaos测试模型

也可以做一些更有意思的事情！！！

## 安装方法
```shell
pip install -i https://test.pypi.org/simple/ mopilot
```

## 例子
你可以看带 test目录 的文件
1. print_module_key_test.py 打印module的key及模拟输入后打印module的输入和输出维度。
2. hook_print_input_output_test.py 打印执行module的输入和输出
3. hook_print_grad_input_output_test.py 打印module backward后梯度的输入和输出
4. hook_forward_input_scale_test.py 对各个module的输入进行缩放
5. stat_module_weight_test.py 对module的key为seq.0.Conv2d的模块权重进行统计且打印结果
6. stat_tensor_test.py 测试StatTensor类
7. http_sampler_mopilot_test.py 测试model的采样器类

## print_module_key_test.py 代码如下

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

## print_module_key_test.py 代码如下
```python
import torch,time

from test.test_model import  TestModel
from sampler_mopilot import SamplerMopilot
import threading

class HttpThread(threading.Thread):
    def __init__(self, sampler:SamplerMopilot):
        threading.Thread.__init__(self)
        self.sampler = sampler
        self.threadLock = threading.Lock()


    def run(self) -> None:
        self.threadLock.acquire()
        self.sampler.http_sampler_mopilot()
        print(f"test")
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