## Notes Matteo
your_project/
├── run.py
├── gev_config3.csv
├── gev_config2.csv
├── gev_config.csv
├── models/
│   		└── ESGMotgageModel.py
└── output/
    		└── test/

folder
cd /Users/matteovalle/Documents/GitHub/vallematteo.github.io/vallematteo.github.io/flood_credit_risk_abm

Run on *Norma* environment

TO ADD agents during the run:\
run.py --tot_step 10 --grid_radius 50 --acpt_score 200 --folder 'output/test1' --settings 2 --repeats 3

TO NOT ADD new agents during the run:\
run.py --tot_step 10 --grid_radius 50 --acpt_score 200 --folder 'output/test1' --settings 2 --repeats 3 --add_agents 
To ensure that new agents are not added during the simulation, you need to set the --add_agents parameter to False

### Issues / Not clear yet
-  s_i: unclear the function: is the shere of income coefficient that remain unspent. also s_i or s_i(I_d) ?? unclear notation in the pdf
- acpt_score unclear what is the function of this parameter. IT'S THE ACCEPTANCE SCORE FROM THE BANK SIDE
- tot_num_agents_his??? not clear the meaning of the parameter


### Meeting 29 nov 2024
- check the random generation of the GEV. I can Set it, but we don't know if she set anything.
- check the bandwidth for the violin plot
- check the kde for income generation
- default threshold has effect on the plots??

gev_list in model it seems it's constant across all the agents. it changes only in time

find_flood_map() that find the flood risk of the agents (constant across all the aagents) f_d

### AGENTS UPDATE PROCESS:
model.self.gev it is the realization of the gev itself

1) gev --> fklein, fgrote, fmiddel, feklein
2) assign one of the following to f_d: fklein, fgrote, fmiddel, feklein --> f_d
3) self.f = 0.3 * self.f + 0.5 * self.f_d (moving average to consider recovery) 

self.f flood impact (on the value of the house)
self.f_d flood impact (absolute value)
self.v House value
self.c collateral
(also look at Table 3.3 p 33)

### INCOME SHOCK PROCESS:
now income shock is constant for every agent
from algorithm 2: (assuming β=0.1)
r_e = 0.1 + β · tanh(GEV /100)
r_e is the margin for the layoffs of the agent: p_RV < r_e, the agent is laid-off.
p_RV randomize across the agents the layoffs (p_RV is random number across all the agents)
UNCLEAR: why we generate a new rv in update_s_d() (default status) when the first one is already generated in update_s_e() (employment status)

A NICE DYNAMIC to create would be to have more skilled job at the 

What parameters are subject of calibration?