# monge-ijoc-code
This repository contains code and results for the paper, "Monge Properties, Optimal Greedy Policies, and Policy
 Improvement for the Dynamic Stochastic Transportation Problem," which has been accepted for publication in the
 INFORMS Journal on Computing.

This repository is divided into two folders. The folder `odrs` contains the code defining the simulation. The file 
`odrsstate.py` defines the passenger and driver arrival simulation, while the file `policies.py` defines the policies
that are used in this simulation. In particular, it defines a greedy maximum matching policy and a variant policy
that alternates between looking for Monge stars and the greedy policy.

The `examples` folder contains a single file `example_run.py` that contains the code used to generate the results 
displayed in the paper. The output from this script is stored in `output\results.csv`. The `results.csv` file is a
`.csv` file. Each row of the file contains results for a single run of the simulation. The fields are as follows:
- `disc`, the time between applying the greedy matching policy. 
- `mean_1`, the mean of the first multivariate normal distribution used in a mixture distribution that determines passenger/driver arrival
locations.
- `cov_1`, the covariance of the first multivariate normal distribution used in a mixture distribution that determines passenger/driver arrival
locations.
- `mean_2`, the mean of the second multivariate normal distribution used in a mixture distribution that determines passenger/driver arrival
locations.
- `cov_2`, the covariance of the second multivariate normal distribution used in a mixture distribution that determines passenger/driver arrival
locations.
- `prob_1`, the probability of the first multivariate normal distribution in the mixture distribution.
- `prob_2`, the probability of the second multivariate normal distribution in the mixture distribution.
- `subdivision`, how frequently the Monge star policy is run for everytime the greedy matching policy is run. 
- `alpha`, the coefficent alpha in the cost function.
- `power`, the exponent in the cost function.
- `timehorizon`, the length of the time horizon.
- `arrival_rate`, the rate at which agents arrrive.
- `seed`, the random seed used in this trial.
- `method_name`, the name of the policy used in this trial. 
- `dw_cost`, the total incurred costs associated with drivers waiting in this trial.
- `pw_cost`, the total incurred costs associated with passengers waiting in this trial.
- `dist_cost`, the total incurred costs associated with distance between assigned pairs in this trial.
- `total_cost`, the total costs incurred in this trial.



This code requires several packages to run correctly. Most of these are listed in the `requirements.txt` file. If you
 are using a python distribution with `pip`, you can use the command `pip install -r requirements.txt` to install these
 packages. In addition, this code makes use of the python interface to the Gurobi commercial software. In order to use
  this, you must have a Gurobi license present on your machine, and you must have installed Gurobi in your python
  environment. Please see `www.gurobi.com` for information on installing and using Gurobi. 