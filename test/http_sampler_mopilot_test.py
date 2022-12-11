import torch,time

from test_model import  TestModel
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
