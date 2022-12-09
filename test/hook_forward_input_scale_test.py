import torch
from mopilot import Mopilot
from test_model import TestModel

m = TestModel()

x = torch.randn(1,20,5,5)
mopilot = Mopilot(m, name="HookForwardInputScale")

class HookForwardInputScale:
    def __init__(self, mopilot, scale):
        super(HookForwardInputScale, self).__init__()
        self.mopilot = mopilot
        self.scale = scale
        # todo save module inputs

    def register_forward_pre_hook(self,module, inputs):
        module_key = self.mopilot.find_module_key(module)
        print(f"module Name: {module.__class__.__name__}, key:{module_key}")
        print(f"before hook inputs: {inputs}")
        re = []
        for i, input in enumerate(inputs):
            re = re + [input * self.scale]
        print(f"after hook inputs: {re}")
        re = tuple(re)
        return  re

print(f"{mopilot.model_path_key.keys()}")
hook_class = HookForwardInputScale(mopilot,2.0)
mopilot.add_register_forward_pre_hook(Mopilot.ALL, hook_class.register_forward_pre_hook)

y = m(x)
