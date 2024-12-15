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

### Solved Issues
- s_i: unclear the function: is the shere of income coefficient that remain unused. also s_i or s_i(I_d) ?? unclear notation in the pdf
- acpt_score unclear what is the function of this parameter. IT'S THE ACCEPTANCE SCORE FROM THE BANK SIDE
- What parameters are subject of calibration? There is no calibration, we generate from a density that coems from ING Data
- UNCLEAR: why we generate a new rv in update_s_d() (default status) when the first one is already generated in update_s_e() (employment status)? it's 2 different variables and must be named differently in the code. in particular the update of the 
### Issues / Not clear yet
- tot_num_agents_his??? not clear the meaning of the parameter
- Expected minimum score and the maximum score?? how to calculate them
- why if the initialization of the agents is random uniform, the defaults are more concentrated in the more risky areas?
- subproblem of the one above: which of the two conditions trigger the failure event? is more commont the hard or the soft margin?

### Meeting 29 nov 2024
- check the random generation of the GEV. I can Set it, but we don't know if she set anything.
- check the bandwidth for the violin plot
- check the kde for income generation
- default threshold has effect on the plots??


### AGENTS UPDATE PROCESS:
model.self.gev it is the realization of the gev itself
find_flood_map() that find the flood risk scenario of all the agents --> f_d (constant across all the agents)
1) gev --> fklein, fgrote, fmiddel, feklein
2) assign one of the following to f_d: fklein, fgrote, fmiddel, feklein --> f_d
3) self.f = 0.3 * self.f + 0.5 * self.f_d (moving average to consider recovery) 

self.f flood impact (on the value of the house)
self.f_d flood impact (absolute value)
self.v House value
self.c collateral
(also look at Table 3.3 p 33)

### Mechanism of the income shock process:
now income shock is constant for every agent
from algorithm 2: (assuming β=0.1)
r_e = 0.1 + β · tanh(GEV /100)
r_e is the margin for the layoffs of the agent: p_RV < r_e, the agent is laid-off.
p_RV randomize across the agents the layoffs (p_RV is random number across all the agents)


### Meeting 6 Dec 2024
*Alternative to the KDE estimation:*
income ~ lognormal
seniority ~ uniform looking at age distribution in the netherlands
r_cap ~ uniform 




TO DO:
For next time series results of the kolmogorov complexity.
THINK ABOUT:
Climate gentrification is the following phenomena:
Initialization now it's random, but if we modify it so that people do NOT join severely hit area it's more realistic. leading to progresssive abandon of the peripheral areas (also severely hitten) of the region.
Would be nice to disentagle the effect of the flood itself in terms of income and property damage and the network effect that dives the prices of the houses down.


### Meeting 13 Dec 2024
final version for the parameter initialization:
-----new_join == False: (initial agents)-----
v, sp = nvm_kde.sample(1)[0] #housing market data kde
income = burr.rvs(c=3.30, d=0.45, loc=-12.76, scale=3101.46)
r_cap = burr.rvs(c=9.42, d=0.14, loc=-0.11, scale=73.40)
seniority = np.random.beta(a=0.88, b=2.79) * 606.34 + 1.0

TO DO:
- Run the Kolmogorov Complexity via BDM analysis
- (add a stopping parameter)
- read carefully calibrated pd meaning: "calibrated PD is not clear the meaning of calibrated ??"
- Meaning of kolmogorov complexity: explaination and application sto a system such as and ABM
- how many agents default by hard margin and how many default by soft margin
- think about parallelization, especially the JAX RL by jakob foerster's lab

### Results folders tracker
- test 3 
- test 4 600 steps reps 10 and gevconfig3.csv used to store the result of the new way to initialize agents
- test 6 1200 steps, reps = 1 and gevconfig4.csv = 600 set1 and then divergent


