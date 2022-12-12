import torch,time

from test_model import  TestModel2
from mopilot.scale_module import ScaleModule
import threading

class HttpThread(threading.Thread):
    def __init__(self, sampler:ScaleModule):
        threading.Thread.__init__(self)
        self.sampler = sampler
        self.threadLock = threading.Lock()


    def run(self) -> None:
        self.threadLock.acquire()
        self.sampler.http_request()
        self.sampler.add_http_support()
        self.sampler.run_http_server()
        self.threadLock.release()



if __name__ == "__main__":

    m = TestModel2()
    m.train(True)

    http_sampler_mopilot = ScaleModule(m, "ScaleModuleOut")
    def todo_train():
        # todo your train code
        for i in range(1000):
            print(f"run: {i}")
            x = torch.randn(1, 20, 5, 5)
            y = m(x)
            loss = torch.nn.functional.mse_loss(x.float(), y.float(), reduction="mean")
            loss.backward()
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

# scale out features using torch.nn.Linear
curl -XGET "http://0.0.0.0:8765/add_scale_module?key=seq.1.Linear&out_dim=2"

# scale out features using torch.nn.Linear
curl -XGET "http://0.0.0.0:8765/add_scale_module?key=seq.0.Conv2d&out_dim=18"
"""
