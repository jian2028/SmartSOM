"""Sample-visit weighted update diagnostics from existing forward outputs."""

from collections import defaultdict
from math import isfinite


class UpdateDiagnostics:
    def __init__(self):
        self.sums = defaultdict(lambda: defaultdict(float))
        self.counts = defaultdict(lambda: defaultdict(int))
        self.nonfinite = defaultdict(lambda: defaultdict(int))

    def add(self, rows, metrics, *, mask=None):
        for index, row in enumerate(rows):
            if mask is not None and not mask[index]:
                continue
            request = row["request"]
            for key in (request.group, request.group + "." + request.stage):
                for metric, values in metrics.items():
                    if isfinite(values[index]):
                        self.sums[key][metric] += values[index]
                    else:
                        self.sums[key][metric] += 0.0
                        self.nonfinite[key][metric] += 1
                    self.counts[key][metric] += 1

    def summary(self, group):
        return {
            key: {
                metric: {
                    "mean": None
                    if self.nonfinite[key][metric]
                    else total / self.counts[key][metric],
                    "sample_visits": self.counts[key][metric],
                    **(
                        {"nonfinite_visits": self.nonfinite[key][metric]}
                        if self.nonfinite[key][metric]
                        else {}
                    ),
                }
                for metric, total in values.items()
            }
            for key, values in self.sums.items()
            if key == group or key.startswith(group + ".")
        }
