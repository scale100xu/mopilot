from torch import  nn

class TestModel(nn.Module):
    def __init__(self):
        super(TestModel,self).__init__()
        self.seq = nn.ModuleList([nn.Conv2d(20,20,1), nn.Conv2d(20,40,1), nn.Conv2d(40,40,1),nn.Conv2d(40,20,1)])

    def forward(self, x):
        y = x
        for m in self.seq:
            y = m(y)
        return y

class TestModel2(nn.Module):
    def __init__(self):
        super(TestModel2,self).__init__()
        self.seq = nn.ModuleList([nn.Conv2d(20,20,1),nn.Linear(5,5),nn.Conv2d(20,40,1), nn.Conv2d(40,40,1),nn.Conv2d(40,20,1)])

    def forward(self, x):
        y = x
        for m in self.seq:
            y = m(y)
        return y