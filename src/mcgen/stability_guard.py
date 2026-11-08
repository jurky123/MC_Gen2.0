"""Stop persistent finite-valued optimization divergence before more updates."""
from dataclasses import asdict, dataclass
import math
import statistics


@dataclass(frozen=True)
class StabilityPolicy:
    warmup: int = 300
    calibration: int = 400
    window: int = 100
    gradient_clip: float = 1.
    gradient_ratio: float = 5.
    loss_ratio: float = 1.15
    clipped_fraction: float = .8
    patience: int = 3


class StabilityGuard:
    def __init__(self, policy=None):
        self.policy = policy or StabilityPolicy()
        p = self.policy
        if (p.warmup < 0 or min(p.calibration, p.window, p.patience) < 1
                or p.gradient_clip <= 0 or p.gradient_ratio <= 1 or p.loss_ratio <= 1
                or not 0 < p.clipped_fraction <= 1):
            raise ValueError('Invalid stability policy')
        self.last_step = 0
        self.calibration = []
        self.baseline_gradient = None
        self.window = []
        self.best_window_loss = None
        self.bad_windows = 0

    def observe(self, step, loss, gradient_norm):
        if step <= self.last_step:
            raise ValueError('Stability observations must advance monotonically')
        self.last_step = step
        if not math.isfinite(loss) or not math.isfinite(gradient_norm):
            return {'step': step, 'stop': True, 'reason': 'nonfinite_loss_or_gradient'}
        p = self.policy
        if step <= p.warmup:
            return None
        if self.baseline_gradient is None:
            self.calibration.append(float(gradient_norm))
            if len(self.calibration) >= p.calibration:
                self.baseline_gradient = max(statistics.median(self.calibration), 1e-12)
                self.calibration.clear()
            return None
        self.window.append((float(loss), float(gradient_norm)))
        if len(self.window) < p.window:
            return None
        loss_median = statistics.median(x[0] for x in self.window)
        gradient_median = statistics.median(x[1] for x in self.window)
        clipped = sum(x[1] > p.gradient_clip for x in self.window)/len(self.window)
        self.window.clear()
        self.best_window_loss = loss_median if self.best_window_loss is None else min(self.best_window_loss, loss_median)
        threshold = max(1.5*p.gradient_clip, p.gradient_ratio*self.baseline_gradient)
        unhealthy = (gradient_median > threshold and clipped >= p.clipped_fraction
                     and loss_median > p.loss_ratio*self.best_window_loss)
        self.bad_windows = self.bad_windows+1 if unhealthy else 0
        return {'step': step, 'loss_median': loss_median, 'gradient_median': gradient_median,
            'baseline_gradient_median': self.baseline_gradient, 'best_window_loss': self.best_window_loss,
            'clipped_fraction': clipped, 'unhealthy': unhealthy, 'bad_windows': self.bad_windows,
            'stop': self.bad_windows >= p.patience,
            'reason': 'persistent_loss_and_gradient_growth' if self.bad_windows >= p.patience else None}

    def state_dict(self):
        return {'policy': asdict(self.policy), 'last_step': self.last_step,
            'calibration': list(self.calibration), 'baseline_gradient': self.baseline_gradient,
            'window': [list(x) for x in self.window], 'best_window_loss': self.best_window_loss,
            'bad_windows': self.bad_windows}

    @classmethod
    def from_state(cls, state):
        guard = cls(StabilityPolicy(**state['policy']))
        for name in ('last_step', 'calibration', 'baseline_gradient', 'window', 'best_window_loss', 'bad_windows'):
            setattr(guard, name, state[name])
        return guard
