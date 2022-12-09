import torch
from mopilot import Mopilot
from test.test_model import TestModel

m = TestModel()
m.train(True)

x = torch.randn(1,20,5,5)
mopilot = Mopilot(m, name="HookUsingGradInputOutput")

class HookUsingGradInputOutput:
    def __init__(self, mopilot):
        super(HookUsingGradInputOutput, self).__init__()
        self.mopilot = mopilot
        # todo save module grad input output

    def register_backward_hook(self,module, grad_inputs, grad_outputs):
        print(f"module Name: {module.__class__.__name__}")
        print(f"grad inputs: {grad_inputs}")
        print(f"grad outputs: {grad_outputs}")


hook_class = HookUsingGradInputOutput(mopilot)
mopilot.add_register_backward_hook(Mopilot.ALL, hook_class.register_backward_hook)

# optimizerG = torch.optim.Adam(m.parameters(), lr=1e-05, betas=(0.001, 0.999))

y = m(x)

loss = torch.nn.functional.mse_loss(x.float(), y.float(), reduction="mean")

loss.backward()