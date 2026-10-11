"""Causal tail estimation with a measured production floor and full endpoint."""
import math

import numpy as np

from ..latency_v2.worker import CausalDelayEstimator


class FlooredDelayEstimator(CausalDelayEstimator):
    def ticks(self):
        latest = float(np.percentile(self.completed, 95)) if self.completed else self.initial
        return math.ceil(max(self.initial, latest) / self.tick_seconds)
