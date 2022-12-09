import json

from test.test_model import TestModel
from mopilot import Mopilot
m = TestModel()
mopilot = Mopilot(m,name="stat_module_weight_test")
print(f"{mopilot.model_path_key.keys()}")

stat_result = mopilot.stat_module_weight('seq.0.Conv2d')

from json import JSONEncoder
class MySelfJSONEncoder(JSONEncoder):
    def default(self, obj):
        return obj.to_json()

print(f"{json.dumps(stat_result,indent=4, cls=MySelfJSONEncoder)}")


"""
output:
{
    "weight": {
        "count": 400,
        "min": -0.2212895154953003,
        "max": 0.22298917174339294,
        "std": 0.1309584528207779,
        "mean": 0.006253551226109266,
        "histogram": {
            "hist": [
                36.0,
                44.0,
                38.0,
                38.0,
                34.0,
                47.0,
                40.0,
                32.0,
                44.0,
                47.0
            ],
            "bin_edges": [
                -0.2212895154953003,
                -0.17686164379119873,
                -0.13243377208709717,
                -0.0880059152841568,
                -0.04357804358005524,
                0.0008498281240463257,
                0.04527769982814789,
                0.08970557153224945,
                0.13413342833518982,
                0.17856130003929138,
                0.22298917174339294
            ]
        },
        "quantile": [
            -0.11142657697200775,
            0.012381263077259064,
            0.12332037836313248
        ]
    },
    "bias": {
        "count": 20,
        "min": -0.2064807415008545,
        "max": 0.21812549233436584,
        "std": 0.13603518903255463,
        "mean": -0.05464944243431091,
        "histogram": {
            "hist": [
                7.0,
                1.0,
                3.0,
                0.0,
                2.0,
                0.0,
                5.0,
                0.0,
                0.0,
                2.0
            ],
            "bin_edges": [
                -0.2064807415008545,
                -0.1640201210975647,
                -0.1215594932436943,
                -0.07909886538982391,
                -0.03663824498653412,
                0.005822375416755676,
                0.04828299582004547,
                0.09074361622333527,
                0.13320425152778625,
                0.17566487193107605,
                0.21812549233436584
            ]
        },
        "quantile": [
            -0.17783448100090027,
            -0.10030193626880646,
            0.061886437237262726
        ]
    }
}
"""