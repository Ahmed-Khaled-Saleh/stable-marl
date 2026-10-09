"""
Stand-in for `wandb`, which MAMBA's learner and losses log to unconditionally (added for
stable-marl): `wandb.log` keeps the latest value of each metric in `wandb.latest`.
"""


class _Logger:
    def __init__(self):
        self.latest = {}

    def init(self, **kwargs):
        pass

    def define_metric(self, *args, **kwargs):
        pass

    def log(self, metrics: dict, **kwargs):
        for key, value in metrics.items():
            try:
                self.latest[key] = float(value)
            except (TypeError, ValueError):
                pass


wandb = _Logger()
