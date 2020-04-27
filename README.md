# monge-ijoc-code
This repository contains code and results for the paper, "Monge Properties, Optimal Greedy Policies, and Policy
 Improvement for the Dynamic Stochastic Transportation Problem," which has been accepted for publication in the
 INFORMS Journal on Computing.

This repository is divided into two folders. The folder `odrs` contains the code defining the simulation. The file 
`odrsstate.py` defines the passenger and driver arrival simulation, while the file `policies.py` defines the policies
that are used in this simulation. In particular, it defines a greedy maximum matching policy and a variant policy
that alternates between looking for Monge stars and the greedy policy.

The `examples` folder contains a single file `example_run.py` that contains the code used to generate the results 
displayed in the paper. The output from this script is stored in `output\results.csv`.

This code requires several packages to run correctly. Most of these are listed in the `requirements.txt` file. If you
 are using a python distribution with `pip`, you can use the command `pip install -r requirements.txt` to install these
 packages. In addition, this code makes use of the python interface to the Gurobi commercial software. In order to use
  this, you must have a Gurobi license present on your machine, and you must have installed Gurobi in your python
  environment. Please see `www.gurobi.com` for information on installing and using Gurobi. 