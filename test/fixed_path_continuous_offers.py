"""Refine alternating offers over continuous longitudinal controls."""

from dataclasses import dataclass
from functools import lru_cache
from math import log

import numpy as np
from scipy.optimize import minimize


FEASIBILITY_TOLERANCE_S = 1e-8


@dataclass
class Proposal:
    delays: tuple
    parameters: np.ndarray
    position: np.ndarray
    speed: np.ndarray
    entry: np.ndarray
    exit: np.ndarray
    finish: np.ndarray
    ramp_acceleration: np.ndarray

    def execute(self, steps):
        return np.stack((self.position[:, 1:steps+1], self.speed[:, 1:steps+1]), axis=-1).transpose(1, 0, 2)


class ContinuousOfferSearch:
    def __init__(self, scene, state, banks, safe, *, first, candidates, times, step,
                 accel, decel, gap, advance, crossing_time, speed_limit=np.inf):
        self.scene = scene
        self.state = state
        self.candidates = candidates
        self.times = times
        self.step = step
        self.accel = accel
        self.decel = decel
        self.gap = gap
        self.crossing_time = crossing_time
        self.speed_limit = speed_limit
        self.cleared = bool(np.any(state[:, 0] >= scene.exit))
        self.min_speed, self.max_speed = candidates[:, 1].min(), candidates[:, 1].max()
        self.first = first
        self.independent_best = np.array([bank[4].min() for bank in banks])
        self.ids = np.argwhere(safe)
        self.finish = np.stack([banks[i][4][self.ids[:, i]] for i in range(2)], axis=1)
        entry = np.stack([banks[i][2][self.ids[:, i]] for i in range(2)], axis=1)
        leave = np.stack([banks[i][3][self.ids[:, i]] for i in range(2)], axis=1)
        self.orders = np.full(len(self.ids), -1)
        if not self.cleared:
            for leader in range(2):
                self.orders[np.isfinite(leave[:, leader]) & (leave[:, leader] + gap <= entry[:, 1-leader])] = leader
        self.held = [self.hold_path(i, advance) for i in range(2)]

    def hold_path(self, vehicle, advance):
        position, speed = self.state[vehicle]
        path = [(position, speed)]
        while len(path) <= round(self.candidates[:, 0].max() / self.step) and position < self.scene.entry[vehicle]:
            remaining = max(0.0, self.scene.entry[vehicle] - 0.7 - position)
            target = min(self.speed_limit, speed, np.sqrt(2 * self.decel * remaining))
            position, speed, _ = advance(position, speed, target, self.scene.routes[vehicle].progress[-1])
            path.append((position, speed))
        return np.array(path)

    def profile(self, delays, parameters):
        starts = [min(round(delay/self.step), len(path)-1) for delay, path in zip(delays, self.held)]
        position = np.empty((2, len(self.times)))
        speed = np.empty_like(position)
        ramp = np.empty(2)
        for i, start in enumerate(starts):
            position[i, :start], speed[i, :start] = self.held[i][:start].T
            s0, v0 = self.held[i][start]
            target, fraction = parameters[2*i:2*i+2]
            # Keep the original maximum braking for overspeed at launch.
            fraction = 1.0 if v0 > self.max_speed else fraction
            rate = (self.accel if target >= v0 else self.decel) * fraction
            acceleration = float(np.sign(target-v0) * rate)
            ramp[i] = acceleration
            steps = np.arange(len(self.times)-start)
            full_steps = (np.minimum(steps, np.floor(abs(target-v0)/(rate*self.step)))
                          if rate > 0 else steps)
            duration = full_steps*self.step
            ramp_speed = v0 + acceleration*duration
            tail_steps = steps-full_steps
            # Integrate the ramp on the same grid and interpolate the final speed change
            # with the same trapezoid used by the discrete controller.
            tail = np.where(tail_steps > 0, (ramp_speed+target)*self.step/2
                            + (tail_steps-1)*target*self.step, 0)
            position[i, start:] = np.minimum(self.scene.routes[i].progress[-1],
                                             s0 + (v0+ramp_speed)*duration/2 + tail)
            speed[i, start:] = np.where(tail_steps > 0, target, ramp_speed)
        entry = self.crossing_time(self.times, position, self.scene.entry[:, None])
        exit = self.crossing_time(self.times, position, self.scene.exit[:, None])
        goal = self.crossing_time(self.times, position, self.scene.goal[:, None])
        finish = np.where(np.isfinite(goal), goal, self.times[-1]+np.maximum(0, self.scene.goal-position[:, -1])/2)
        return Proposal(delays, parameters, position, speed, entry, exit, finish, ramp)

    def order(self, proposal):
        if self.cleared:
            return -1
        for i in range(2):
            if np.isfinite(proposal.exit[i]) and proposal.exit[i]+self.gap <= proposal.entry[1-i]+FEASIBILITY_TOLERANCE_S:
                return i
        return -1 if np.isinf(proposal.entry).all() else None

    def refine(self, seed, actor, responder_limit):
        leader = self.order(seed)
        if leader is None or (leader == -1 and not self.cleared):
            return seed

        @lru_cache(maxsize=16)
        def cached(parameters):
            return self.profile(seed.delays, np.asarray(parameters))

        def loss(parameters):
            proposal = cached(tuple(parameters))
            return proposal.finish[actor] + 1e-5 * proposal.finish.sum()

        def constraints(parameters):
            proposal = cached(tuple(parameters))
            values = []
            if np.isfinite(responder_limit):
                values.append(responder_limit - proposal.finish[1-actor])
            if leader != -1:
                entry = min(proposal.entry[1-leader], self.times[-1]+1.0)
                exit = min(proposal.exit[leader], self.times[-1]+2.0)
                values.append(entry - exit - self.gap)
            return np.asarray(values)

        result = minimize(loss, seed.parameters, method="SLSQP",
                          bounds=[(self.min_speed, self.max_speed), (0.0, 1.0)] * 2,
                          constraints={"type": "ineq", "fun": constraints},
                          options={"maxiter": 16, "ftol": 1e-7, "eps": 1e-5})
        proposal = cached(tuple(result.x))
        if self.order(proposal) is not None and proposal.finish[1-actor] <= responder_limit+FEASIBILITY_TOLERANCE_S:
            return min((seed, proposal), key=lambda offer: (offer.finish[actor], offer.finish.sum()))
        return seed

    def select(self, delta):
        def seed_from(index):
            delays, speeds = self.candidates[self.ids[index]].T
            return self.profile(tuple(delays), np.column_stack((speeds, [1.0, 1.0])).ravel())

        def best_index(indices, actor):
            return indices[np.lexsort((self.finish[indices].sum(axis=1), self.finish[indices, actor]))[0]]

        # If both vehicles can reach their goals at their individual earliest times,
        # continuous controls cannot improve on applying the maximum allowed acceleration immediately.
        fastest = np.flatnonzero(np.all(np.abs(self.finish-self.independent_best) <= 1e-9, axis=1))
        if len(fastest):
            return seed_from(fastest[0])

        discounted_wait = -2.0 * log(delta) if delta > 0 else np.inf
        continuation = None
        for actor in (self.first, 1-self.first, self.first):
            responder_limit = np.inf if continuation is None else continuation.finish[1-actor]+discounted_wait
            feasible = np.flatnonzero(self.finish[:, 1-actor] <= responder_limit+FEASIBILITY_TOLERANCE_S)
            options = [] if continuation is None else [continuation]
            for order in (-1, 0, 1):
                group = feasible[self.orders[feasible] == order]
                if len(group):
                    seed = seed_from(best_index(group, actor))
                    if self.order(seed) is not None:
                        options.append(self.refine(seed, actor, responder_limit))
            if not options:
                return None
            continuation = min(options, key=lambda offer: (offer.finish[actor], offer.finish.sum()))
        return continuation
