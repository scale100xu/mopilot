# mopilot

mopilot is model copilot using pytorch,
it is seal torch.nn.module hook functions.
it strange module hook functions and simple hook module that module struct path key(module struct path key is {module_name}.{index}.{module_class})

1. print module info 
2. print module input/output or grad_input grad_output
3. inject module input/output(may be replace module function)
4. inject module grad_input(replace grad algorithm)
5. may be complex inject module(as teacher/student model)
6. you can chaos module in model

# install
```shell
pip install -i https://test.pypi.org/simple/ mopilot
```

# examples
you can view test/*.py file