import sys
import os
import simpy
import scipy.stats
import numpy
import pandas

sys.path.append(os.path.abspath(os.path.join(__file__, os.pardir)))
import odrs.odrsstate as state
import odrs.policies as policies

time_horizon = 128.001
arrival_rate = 1.0
power = 1.5
subdivision = 2
mean_1 = (0.0, 0.0)
cov_1 = ((1, 0), (0, 1))
mean_2 = (5, 5)
cov_2 = ((1, 0), (0, 1))
prob_1 = 0.5
prob_2 = 0.5

dist = policies.euclidean_dist

fields = ['disc', 'mean_1', 'cov_1', 'mean_2', 'cov_2', 'prob_1', 'prob_2', 'subdivision', 'alpha', 'power',
          'timehorizon', 'arrival_rate', ' seed', 'method_name', 'dw_cost', 'pw_cost', 'dist_cost', 'total_cost']
results_dict = {f: [] for f in fields}
for alpha in [1 / 256, 1 / 128, 0.03125, .0625, 0.125, 0.25]:
    waitcost = policies.SupLinCost(alpha=alpha, power=power)
    params = state.ProblemParams(pw_cost=waitcost, dw_cost=waitcost, dist=dist)
    for disc in [0.03125, 0.0625, 0.125, 0.25, 0.5, 1, 2, 4, 8, 16, 32, 64, 128]:
        method_names = []
        print(alpha, disc)
        for seed in range(0, 50):
            for method, method_name in zip([policies.mincost_batch_policy_gen(disc=disc),
                                            policies.star_batch_policy_gen(star_disc=disc / subdivision,
                                                                           iter_between=subdivision)],
                                           ['nostar', 'star']):
                random_state = numpy.random.RandomState(seed=seed)
                normal_gen_1 = state.normal_loc(mean=numpy.array(mean_1), cov=numpy.array(cov_1),
                                                random_state=random_state)
                normal_gen_2 = state.normal_loc(mean=numpy.array(mean_2), cov=numpy.array(cov_2),
                                                random_state=random_state)
                mix_dist = state.mix_gens(probs=[prob_1, prob_2], gens=[normal_gen_1, normal_gen_2],
                                          random_state=random_state)
                interarrival_gen = state.scipy_rvs_to_time(scipy.stats.expon(loc=0, scale=1 / arrival_rate),
                                                           random_state=random_state)
                agent_gen = state.generate_arrivals(interarrival_dist=interarrival_gen, loc_dist=mix_dist)

                myenv = simpy.Environment()
                mystate = state.State()
                state.runsim(myenv, passengers=agent_gen, drivers=agent_gen, policy_generator=method,
                             params=params, state=mystate)
                myenv.run(until=time_horizon)
                costs = mystate.calc_costs(time=time_horizon, params=params)
                for k, v in zip(fields, [disc, mean_1, cov_1, mean_2, cov_2, prob_1, prob_2, subdivision, alpha, power,
                                         time_horizon, arrival_rate, seed, method_name, costs.dw_cost, costs.pw_cost,
                                         costs.dist_cost, costs.total_cost()]):
                    results_dict[k].append(v)

output_file = os.path.abspath(os.path.join(os.path.join(os.path.join(os.path.join(__file__, os.pardir),os.pardir),
                                                        "output"), "results.csv"))
pandas.DataFrame(data=results_dict).to_csv(output_file, index=False)
