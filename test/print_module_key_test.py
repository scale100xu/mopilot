import torch

from mopilot import Mopilot
import json
from test.test_model import  TestModel

m = TestModel()

p = Mopilot(m,name="printModuleKey")
# print module path key
print(f"{json.dumps(list(p.model_path_key.keys()),indent=4)}")
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

x = torch.randn(1,20,5,5)
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


