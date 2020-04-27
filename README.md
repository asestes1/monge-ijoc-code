# monge-ijoc-code
This repository contains code and results for the paper, "Monge Properties, Optimal Greedy Policies, and Policy
 Improvement for the Dynamic Stochastic Transportation Problem," which has been accepted for publication in the
 INFORMS Journal on Computing.

This repository is divided into two folders. The folder `odrs` contains the code defining the simulation. The file 
`odrsstate.py` defines the passenger and driver arrival simulation, while the file `policies.py` defines the policies
that are used in this simulation. In particular, it defines a greedy maximum matching policy and a variant policy
that alternates between looking for Monge stars and the greedy policy.

The `examples` folder contains a single file `example_run.py` that 