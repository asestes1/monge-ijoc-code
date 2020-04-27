import numpy
import scipy.stats
import typing
import attr
import simpy
import operator

PASSENGER_CODE = 1
DRIVER_CODE = 2

Location = typing.Tuple[float, float]
DistanceMeasure = typing.Callable[[Location, Location], float]


@attr.s(frozen=True, kw_only=True)
class ProblemParams(object):
    pw_cost = attr.ib(type=typing.Callable[[float], float])
    dw_cost = attr.ib(type=typing.Callable[[float], float])
    dist = attr.ib(type=DistanceMeasure)


@attr.s(frozen=True, kw_only=True)
class TimeLocation(object):
    loc = attr.ib(type=Location)
    time = attr.ib(type=float)


@attr.s(frozen=True, kw_only=True)
class Agent(object):
    agentid = attr.ib(type=int)
    loc = attr.ib(type=Location)
    arrival = attr.ib(type=float)


@attr.s(frozen=True, kw_only=True)
class Assignment(object):
    passenger = attr.ib(type=typing.Optional[Agent])
    driver = attr.ib(type=typing.Optional[Agent])
    time = attr.ib(type=float)

@attr.s(frozen=True, kw_only=True)
class CostResult(object):
    pw_cost = attr.ib(type=float)
    dw_cost = attr.ib(type=float)
    dist_cost = attr.ib(type=float)

    def total_cost(self):
        return self.pw_cost+self.dw_cost+self.dist_cost

@attr.s(kw_only=True)
class State(object):
    nextid = attr.ib(type=int, default=0)
    passengers = attr.ib(type=typing.Set[Agent], converter=set, default=set())
    drivers = attr.ib(type=typing.Set[Agent], converter=set, default=set())
    assignments = attr.ib(type=typing.Set[Assignment], converter=set, default=set())

    def __str__(self):
        my_str = "P:\n"
        for p in sorted(self.passengers, key=operator.attrgetter("arrival")):
            my_str += "\t" + str(p) + "\n"

        my_str += "D:\n"
        for d in sorted(self.drivers, key=operator.attrgetter("arrival")):
            my_str += "\t" + str(d) + "\n"

        my_str += "M:\n"
        for assign in sorted(self.assignments, key=operator.attrgetter('time')):
            my_str += "\t" + str(assign.time) + "\t" + str(assign.passenger) + "\t" + str(assign.driver) + "\n"
        return my_str

    def add_assignment(self, assignment: Assignment):
        p = assignment.passenger
        d = assignment.driver
        t = assignment.time
        if p is not None and p not in self.passengers:
            raise ValueError("Error in assignment: passenger does not exist or has already been assigned a driver.")
        if d is not None and d not in self.drivers:
            raise ValueError("Error in assignment: driver does not exist or has already been assigned a passenger.")
        if t < p.arrival or t < d.arrival:
            raise ValueError("Passenger assigned before arrival")

        if p is not None:
            self.passengers.remove(p)
        if d is not None:
            self.drivers.remove(d)
        self.assignments.add(assignment)

    def add_agent(self, time: float, loc: Location, typecode: int) -> None:
        if typecode == DRIVER_CODE:
            self.add_driver(time=time, loc=loc)
        elif typecode == PASSENGER_CODE:
            self.add_passenger(time=time, loc=loc)
        return

    def add_passenger(self, time: float, loc: Location) -> None:
        self.passengers.add(Agent(agentid=self.nextid, arrival=time, loc=loc))
        self.nextid += 1
        return

    def add_driver(self, time: float, loc: Location) -> None:
        self.drivers.add(Agent(agentid=self.nextid, arrival=time, loc=loc))
        self.nextid += 1
        return

    def copy(self) -> 'State':
        copy_pass = set(self.passengers)
        copy_drivers = set(self.drivers)
        copy_assign = set(self.assignments)
        return State(passengers=copy_pass, drivers=copy_drivers, assignments=copy_assign, nextid=self.nextid)

    def calc_costs(self, time: float, params: ProblemParams) -> CostResult:
        pw_costs = 0
        dw_costs = 0
        dist_costs = 0
        for a in self.assignments:
            pw_costs += params.pw_cost(a.time-a.passenger.arrival)
            dw_costs += params.dw_cost(a.time - a.driver.arrival)
            dist_costs += params.dist(a.driver.loc, a.passenger.loc)

        for p in self.passengers:
            pw_costs += params.pw_cost(time - p.arrival)

        for d in self.drivers:
            dw_costs += params.dw_cost(time - d.arrival)
        return CostResult(pw_cost=pw_costs, dw_cost=dw_costs, dist_cost=dist_costs)


Policy = typing.Callable[[float, State, ProblemParams], None]
TimePolicy = typing.Tuple[float, Policy]


def agent_process(env: simpy.Environment, arrival_generator: typing.Generator[TimeLocation, None, None], typecode: int,
                  state: State):
    for a in arrival_generator:
        yield env.timeout(a.time)
        state.add_agent(time=env.now, loc=a.loc, typecode=typecode)


def policy_process(env: simpy.Environment, state: State, policy_generator: typing.Generator[TimePolicy, None, None],
                   params: ProblemParams):
    for t, p in policy_generator:
        yield env.timeout(t)
        p(env.now, state, params)


def runsim(env: simpy.Environment, passengers: typing.Generator[TimeLocation, None, None],
           drivers: typing.Generator[TimeLocation, None, None],
           policy_generator: typing.Generator[TimePolicy, None, None],
           params: ProblemParams,
           state: State):
    env.process(agent_process(env=env, arrival_generator=passengers, typecode=PASSENGER_CODE, state=state))
    env.process(agent_process(env=env, arrival_generator=drivers, typecode=DRIVER_CODE, state=state))
    env.process(policy_process(env=env, state=state, policy_generator=policy_generator, params=params))


def scipy_rvs_to_time(scipy_var: scipy.stats.rv_continuous,
                      random_state: typing.Optional[numpy.random.RandomState] = None) -> typing.Generator[
    float, None, None]:
    while True:
        yield float(scipy_var.rvs(1, random_state=random_state))


def generate_arrivals(interarrival_dist: typing.Generator[float, None, None],
                      loc_dist: typing.Generator[Location, None, None]) -> typing.Generator[TimeLocation, None, None]:
    for t, l in zip(interarrival_dist, loc_dist):
        yield TimeLocation(time=t, loc=l)


def mv_uniform_loc(bounds: typing.Tuple[typing.Tuple[float, float], typing.Tuple[float, float]],
                   random_state: typing.Optional[numpy.random.RandomState] = None) -> typing.Generator[
    Location, None, None]:
    while True:
        x_coordinate = scipy.stats.uniform.rvs(size=1, loc=bounds[0][0], scale=bounds[0][1] - bounds[0][0],
                                               random_state=random_state)
        y_coordinate = scipy.stats.uniform.rvs(size=1, loc=bounds[1][0], scale=bounds[1][1] - bounds[1][0],
                                               random_state=random_state)
        yield (float(x_coordinate), float(y_coordinate))


def normal_loc(mean: numpy.ndarray, cov: numpy.ndarray,
               random_state: typing.Optional[numpy.random.RandomState] = None) -> typing.Generator[
    Location, None, None]:
    while True:
        loc_array = scipy.stats.multivariate_normal.rvs(size=1, mean=mean, cov=cov, random_state=random_state)
        yield (float(loc_array[0]), float(loc_array[1]))


def mix_gens(probs: typing.List[float], gens: typing.List[typing.Generator],
             random_state: typing.Optional[numpy.random.RandomState] = None) -> typing.Generator:
    mixture_dist = scipy.stats.rv_discrete(a=0, b=len(probs) - 1, values=(range(0, len(probs)), probs))
    while True:
        next_sample = mixture_dist.rvs(size=1, random_state=random_state)
        yield next(gens[int(next_sample)])
