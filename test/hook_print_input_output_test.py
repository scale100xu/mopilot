import torch
from mopilot import Mopilot
from test.test_model import TestModel

m = TestModel()
x = torch.randn(1,20,5,5)
mopilot = Mopilot(m, name="AddModuleHookUsingPrintInputOutput")

class AddModuleHookUsingPrintInputOutput:
    def __init__(self, mopilot):
        super(AddModuleHookUsingPrintInputOutput, self).__init__()
        self.mopilot = mopilot
        # todo save module input output

    def register_forward_hook(self,module, inputs, outputs):
        print(f"module Name: {module.__class__.__name__}")
        print(f"inputs: {inputs}")
        print(f"outputs: {outputs}")


hook_class = AddModuleHookUsingPrintInputOutput(mopilot)
mopilot.add_register_forward_hook(Mopilot.ALL, hook_class.register_forward_hook)

y = m(x)