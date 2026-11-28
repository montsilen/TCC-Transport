"""Joint offer feasibility for two fixed paths through a lane drop."""

from functools import lru_cache

import numpy as np
from scipy.optimize import minimize

import run_fixed_path_ttc_trial as trial
from fixed_path_continuous_offers import ContinuousOfferSearch, FEASIBILITY_TOLERANCE_S


FORECAST_STRIDE = 2  # 0.2 s. The independent outcome scorer uses 0.02 s.
SEPARATION_TOLERANCE_M = 1e-6


def poses(route, progress):
    angle = np.interp(progress, route.progress, route.angles)
    return tuple(np.interp(progress, route.progress, route.points[:, i]) for i in (0, 1)) + (
        np.cos(angle), np.sin(angle))


def separation(a, b, sizes):
    """Maximum separating-axis gap, positive exactly for separated rectangles."""
    ax, ay, ac, asi = a
    bx, by, bc, bsi = b
    dx, dy = bx - ax, by - ay
    cosine = np.abs(ac * bc + asi * bsi)
    sine = np.abs(ac * bsi - asi * bc)
    (la, wa), (lb, wb) = np.asarray(sizes) / 2
    return np.maximum.reduce([
        np.abs(dx * ac + dy * asi) - la - lb * cosine - wb * sine,
        np.abs(-dx * asi + dy * ac) - wa - lb * sine - wb * cosine,
        np.abs(dx * bc + dy * bsi) - lb - la * cosine - wa * sine,
        np.abs(-dx * bsi + dy * bc) - wb - la * sine - wa * cosine,
    ])


def clearance_until_finish(separation_m, a, b, goal):
    complete = (a >= goal[0]) & (b >= goal[1])
    # Include the first sample at or beyond joint completion.
    return np.min(np.where(complete[..., :-1], np.inf, separation_m[..., 1:]), axis=-1)


class MergePolicy:
    def feasible(self, scene, state, banks):
        a, b = (bank[0][:, ::FORECAST_STRIDE] for bank in banks)
        pa = tuple(x[:, None, :] for x in poses(scene.routes[0], a))
        pb = tuple(x[None, :, :] for x in poses(scene.routes[1], b))
        margin = separation(pa, pb, [route.size for route in scene.routes])
        return clearance_until_finish(margin, a[:, None, :], b[None, :, :],
                                      scene.goal) > SEPARATION_TOLERANCE_M

    def mpc(self, banks, safe):
        # Two profiles with the same merge slot can have different following gaps.
        cost = banks[0][5][:, None] + banks[1][5][None, :]
        i, j = np.unravel_index(np.argmin(np.where(safe, cost, np.inf)), safe.shape)
        return int(i), int(j)

    def emergency(self, scene, state, steps, speed_limit):
        position, speed = state.T.copy()
        end = [route.progress[-1] for route in scene.routes]
        history = []
        for _ in range(steps):
            position, speed, _ = trial.advance(position, speed, 0.0, end)
            history.append(np.column_stack((position, speed)))
        return np.array(history)


class MergeOfferSearch(ContinuousOfferSearch):
    def __init__(self, scene, state, banks, safe, **settings):
        super().__init__(scene, state, banks, safe, **settings)
        if not self.cleared:
            a, b = (banks[i][3][self.ids[:, i]] for i in (0, 1))
            self.orders = np.where(np.isinf(a) & np.isinf(b), -1, np.where(a <= b, 0, 1))

    def margin(self, proposal):
        a, b = proposal.position[:, ::FORECAST_STRIDE]
        routes = self.scene.routes
        distance = separation(poses(routes[0], a), poses(routes[1], b), [r.size for r in routes])
        return clearance_until_finish(distance, a, b, self.scene.goal)

    def order(self, proposal):
        if self.margin(proposal) <= SEPARATION_TOLERANCE_M:
            return None
        if self.cleared or np.isinf(proposal.exit).all():
            return -1
        return int(np.argmin(proposal.exit))

    def refine(self, seed, actor, responder_limit):
        @lru_cache(maxsize=16)
        def cached(parameters):
            return self.profile(seed.delays, np.asarray(parameters))

        def loss(parameters):
            proposal = cached(tuple(parameters))
            return proposal.finish[actor] + 1e-5 * proposal.finish.sum()

        def constraints(parameters):
            proposal = cached(tuple(parameters))
            margins = [self.margin(proposal) - 2 * SEPARATION_TOLERANCE_M]
            if np.isfinite(responder_limit):
                margins.append(responder_limit - proposal.finish[1 - actor])
            return np.asarray(margins)

        result = minimize(loss, seed.parameters, method="SLSQP",
                          bounds=[(self.min_speed, self.max_speed), (0.0, 1.0)] * 2,
                          constraints={"type": "ineq", "fun": constraints},
                          options={"maxiter": 16, "ftol": 1e-7, "eps": 1e-5})
        proposal = cached(tuple(result.x))
        if self.order(proposal) is not None and proposal.finish[1 - actor] <= responder_limit + FEASIBILITY_TOLERANCE_S:
            return min((seed, proposal), key=lambda p: (p.finish[actor], p.finish.sum()))
        return seed


MergePolicy.continuous_search = MergeOfferSearch
