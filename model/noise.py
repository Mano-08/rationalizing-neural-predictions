import math
from dataclasses import dataclass, field
from typing import Optional

import torch


SCHEDULES = ['constant', 'exponential', 'cosine', 'linear', 'closed_loop']
SIGNALS = ['jsd', 'entropy']


def js_div_per_example(P, Q, eps = 1e-12):
    # Jensen-Shannon divergence per row, normalized to [0, 1] (divided by ln 2)
    P, Q = P.clamp_min(eps), Q.clamp_min(eps)
    M = (P + Q)/2
    kl = lambda X: (X * (X.log() - M.log())).sum(-1)
    return (kl(P) + kl(Q))/(2 * math.log(2))


def normalized_attention_entropy(token_att, selectable = None, eps = 1e-12):
    # Entropy of the generator's attention per row, normalized to [0, 1] by
    # the log of the number of positions the attention can be spread over
    att = token_att.squeeze(-1)
    entropy = -(att * att.clamp_min(eps).log()).sum(-1)
    if selectable is None:
        num_positions = torch.full_like(entropy, att.shape[-1])
    else:
        num_positions = selectable.sum(-1).type_as(entropy)
    return entropy/num_positions.clamp_min(2).log()


@dataclass
class ConstantSchedule:
    p: float

    closed_loop = False

    def value(self, step):
        return self.p

    def state(self):
        return {}


@dataclass
class ExponentialSchedule:
    p0: float
    p_final: float
    # decay rate per EPOCH (not per step), so that the realized schedule does
    # not depend on the batch size or the dataset size
    gamma: float
    steps_per_epoch: int

    closed_loop = False

    def value(self, step):
        return self.p_final + (self.p0 - self.p_final) * math.exp(-self.gamma * step/self.steps_per_epoch)

    def state(self):
        return {}


@dataclass
class CosineSchedule:
    p0: float
    p_final: float
    total_steps: int

    closed_loop = False

    def value(self, step):
        s = min(step/max(self.total_steps, 1), 1.0)
        return self.p_final + (self.p0 - self.p_final) * (1 + math.cos(math.pi * s))/2

    def state(self):
        return {}


@dataclass
class LinearSchedule:
    p0: float
    p_final: float
    total_steps: int

    closed_loop = False

    def value(self, step):
        s = min(step/max(self.total_steps, 1), 1.0)
        return self.p0 + (self.p_final - self.p0) * s

    def state(self):
        return {}


@dataclass
class PIController:
    # Closed-loop noise level: a proportional-integral controller on an
    # exponential moving average of a degeneracy signal measured on the model.
    #   jsd:     disagreement between the attention-based and the
    #            rationale-based predictor on the CLEAN rationale. Noise goes
    #            up when the two predictors decouple.
    #   entropy: normalized entropy of the generator's attention. Noise goes
    #            up when the attention collapses onto few tokens.
    signal: str
    target: float
    kp: float
    ki: float
    ema: float
    p_min: float
    p_max: float
    p_init: Optional[float] = None
    probe_every: int = 10
    signal_ema: Optional[float] = field(default = None, init = False)
    integral: float = field(default = 0.0, init = False)
    p: float = field(default = 0.0, init = False)

    closed_loop = True

    def __post_init__(self):
        if self.signal not in SIGNALS:
            raise ValueError(f'Unknown controller signal {self.signal}')
        if not 0 <= self.p_min <= self.p_max:
            raise ValueError('Controller requires 0 <= p_min <= p_max')
        self.p = self.p_min if self.p_init is None else min(max(self.p_init, self.p_min), self.p_max)
        # start the integral term so that zero error holds the initial level
        self.integral = (self.p - self.p_min)/self.ki if self.ki > 0 else 0.0

    def error(self):
        if self.signal == 'entropy':
            return self.target - self.signal_ema
        return self.signal_ema - self.target

    def needs_probe(self, step):
        # the jsd signal costs a forward pass of the rationale predictor
        return self.signal == 'jsd' and step % self.probe_every == 0

    def update(self, observation):
        self.signal_ema = observation if self.signal_ema is None else self.ema * self.signal_ema + (1 - self.ema) * observation
        error = self.error()
        if self.ki > 0:
            # anti-windup: the integral term alone can never leave [p_min, p_max]
            self.integral = min(max(self.integral + error, 0.0), (self.p_max - self.p_min)/self.ki)
        self.p = min(max(self.p_min + self.kp * error + self.ki * self.integral, self.p_min), self.p_max)
        return self.p

    def value(self, step):
        return self.p

    def state(self):
        return {"signal_ema": self.signal_ema, "integral": self.integral}


def create_noise_schedule(args, steps_per_epoch, total_steps):
    if args.noise_schedule == 'constant':
        return ConstantSchedule(p = args.noise_p)
    if args.noise_schedule == 'closed_loop':
        return PIController(
            signal = args.ctrl_signal,
            target = args.ctrl_target,
            kp = args.ctrl_kp,
            ki = args.ctrl_ki,
            ema = args.ctrl_ema,
            p_min = args.noise_p_min,
            p_max = args.noise_p_max,
            p_init = args.noise_p,
            probe_every = args.ctrl_probe_every
        )
    if args.noise_p0 is None:
        raise ValueError(f'--noise_p0 is required for the {args.noise_schedule} schedule')
    if args.noise_schedule == 'exponential':
        return ExponentialSchedule(p0 = args.noise_p0, p_final = args.noise_p, gamma = args.noise_gamma, steps_per_epoch = steps_per_epoch)
    if args.noise_schedule == 'cosine':
        return CosineSchedule(p0 = args.noise_p0, p_final = args.noise_p, total_steps = total_steps)
    if args.noise_schedule == 'linear':
        return LinearSchedule(p0 = args.noise_p0, p_final = args.noise_p, total_steps = total_steps)
    raise ValueError(f'Unknown noise schedule {args.noise_schedule}')
