## Notes Matteo
folder
cd /Users/matteovalle/Documents/GitHub/vallematteo.github.io/vallematteo.github.io/flood_credit_risk_abm


TO ADD agents during the run:\
run.py --tot_step 10 --grid_radius 50 --acpt_score 200 --folder 'output/test1' --settings 2 --repeats 3

TO NOT ADD new agents during the run:\
run.py --tot_step 10 --grid_radius 50 --acpt_score 200 --folder 'output/test1' --settings 2 --repeats 3 --add_agents 
To ensure that new agents are not added during the simulation, you need to set the --add_agents parameter to False

### Issues
-  s_i: unclear the function: is the shere of income coefficient that remain unspent. also s_i or s_i(I_d) ?? unclear notation in the pdf
- acpt_score unclear what is the function of this parameter. IT'S THE ACCEPTANCE SCORE FROM THE BANK SIDE
### Not clear yet




your_project/
├── run.py
├── gev_config3.csv
├── gev_config2.csv
├── gev_config.csv
├── models/
│   		└── ESGMotgageModel.py
└── output/
    		└── test/



### Meeting 29 nov 2024
- check the random generation of the GEV. I can Set it, but we don't know if she set anything.
- check the bandwidth for the violin plot
- check the kde for income generation
- default threshold has effect on the plots??

