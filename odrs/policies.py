import gurobipy as grb
import attr
from collections import namedtuple
import abc
from . import odrsstate
import math


class Policy(abc.ABC):
    @abc.abstractmethod
    def apply(self, state: odrsstate.State, p_params: odrsstate.ProblemParams) -> None:
        pass


@attr.s(frozen=True, kw_only=True)
class StarFinder(object):
    time_disc = attr.ib(type=float)

    def __call__(self, time: float, crnt_state: odrsstate.State, p_params: odrsstate.ProblemParams) -> None:
        star_pair = StarFinder.find_monge_star(time, crnt_state, p_params, self.time_disc)
        while star_pair is not None:
            crnt_state.add_assignment(odrsstate.Assignment(passenger=star_pair.passenger, driver=star_pair.driver,
                                                           time=time))
            star_pair = StarFinder.find_monge_star(time, crnt_state, p_params, self.time_disc)
        return

    @staticmethod
    def find_monge_star(time: float, state: odrsstate.State, p_params: odrsstate.ProblemParams, delta: float):
        if not state.passengers or not state. drivers:
            return None

        Match = namedtuple('Match', ['passenger', 'driver'])
        oldest_p = StarFinder.find_oldest(state.passengers)
        oldest_d = StarFinder.find_oldest(state.drivers)
        if oldest_p is None or oldest_d is None:
            return None

        for p in oldest_p:
            for d in oldest_d:
                dist = p_params.dist(p.loc, d.loc)
                mar_p = p_params.pw_cost.marg(time - p.arrival, delta=delta)
                mar_d = p_params.dw_cost.marg(time - d.arrival, delta=delta)
                if 2 * dist <= mar_p and 2 * dist <= mar_d:
                    return Match(p, d)
        return None

    @staticmethod
    def find_oldest(set_agents):
        oldest = None
        oldest_arr = None
        for a in set_agents:
            if oldest_arr is None or a.arrival < oldest_arr:
                oldest = {a}
                oldest_arr = a.arrival
            elif a.arrival == oldest_arr:
                oldest.add(a)
        return oldest


def mincost_match(time: float, state: odrsstate.State, p_params: odrsstate.ProblemParams) -> None:
    if state.passengers and state.drivers:
        mm_assigns = find_mm(state, p_params)
        for (p, d) in mm_assigns:
            state.add_assignment(odrsstate.Assignment(passenger=p, driver=d, time=time))
    return


def find_mm(state, p_params):
    model = grb.Model()
    model.setParam(grb.GRB.Param.OutputFlag, 0)
    evars = add_vars(model, state, p_params)
    add_constr(model, state, evars)
    model.optimize()
    assignments = get_assignments(evars)
    return assignments


def add_mm_constr(model, evars, n_match):
    lhs = grb.LinExpr()
    for v in evars.edge_vars.values():
        lhs.add(v, 1.0)

    model.addConstr(lhs, grb.GRB.GREATER_EQUAL, n_match)
    model.update()
    return


def add_vars(model, state, p_params):
    edge_vars = {}
    imbalance = len(state.passengers) - len(state.drivers)
    for p in state.passengers:
        for d in state.drivers:
            dist = p_params.dist(p.loc, d.loc)
            edge_vars[(p, d)] = model.addVar(lb=0.0, ub=1.0, obj=dist, vtype=grb.GRB.BINARY)

    if imbalance > 0:
        for p in state.passengers:
            edge_vars[(p, None)] = model.addVar(lb=0.0, ub=1.0, obj=0.0, vtype=grb.GRB.BINARY)
    elif imbalance < 0:
        for d in state.drivers:
            edge_vars[(None, d)] = model.addVar(lb=0.0, ub=1.0, obj=0.0, vtype=grb.GRB.BINARY)
    model.update()
    return edge_vars


def add_constr(model, state, edge_vars):
    imbalance = len(state.passengers) - len(state.drivers)
    for p in state.passengers:
        lhs = grb.LinExpr()
        for d in state.drivers:
            lhs.add(edge_vars[(p, d)], 1.0)

        if imbalance > 0:
            lhs.add(edge_vars[(p, None)], 1.0)
        model.addConstr(lhs, grb.GRB.EQUAL, 1)

    for d in state.drivers:
        lhs = grb.LinExpr()
        for p in state.passengers:
            lhs.add(edge_vars[(p, d)], 1.0)

        if imbalance < 0:
            lhs.add(edge_vars[(None, d)], 1.0)
        model.addConstr(lhs, grb.GRB.EQUAL, 1)

    if imbalance > 0:
        lhs = grb.LinExpr()
        for p in state.passengers:
            lhs.add(edge_vars[(p, None)], 1.0)
        model.addConstr(lhs, grb.GRB.EQUAL, imbalance)
    elif imbalance < 0:
        lhs = grb.LinExpr()
        for d in state.drivers:
            lhs.add(edge_vars[(None, d)], 1.0)
        model.addConstr(lhs, grb.GRB.EQUAL, -imbalance)
    return


def get_assignments(edge_vars):
    assignments = set()
    for ((p, d), v) in edge_vars.items():
        if p is not None and d is not None:
            value = v.getAttr(grb.GRB.Attr.X)
            if abs(value - 1) < 0.0001:
                assignments.add((p, d))
    return assignments


@attr.s(frozen=True, kw_only=True)
class SupLinCost(object):
    alpha = attr.ib(type=float, default=1)
    power = attr.ib(type=float, default=1.5)

    def __call__(self, t: float) -> float:
        return (self.alpha * t) ** self.power

    def marg(self, t: float, delta: float):
        return self.__call__(t + delta) - self.__call__(t)


def euclidean_dist(loc1: odrsstate.Location, loc2: odrsstate.Location) -> float:
    return math.sqrt((loc1[0] - loc2[0]) ** 2 + (loc1[1] - loc2[1]) ** 2)


def mincost_batch_policy_gen(disc: float):
    while True:
        yield (disc, mincost_match)


def compose(policy_first: odrsstate.Policy, policy_second: odrsstate.Policy) -> odrsstate.Policy:
    def new_policy(time: float, state: odrsstate.State, params: odrsstate.ProblemParams):
        policy_first(time, state, params)
        policy_second(time, state, params)

    return new_policy


def star_batch_policy_gen(star_disc: float, iter_between: int):
    i = 1
    while True:
        if i == 0:
            yield (star_disc, compose(mincost_match, StarFinder(time_disc=star_disc)))
        else:
            yield (star_disc, StarFinder(time_disc=star_disc))
        i = (i + 1) % iter_between
