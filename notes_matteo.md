## Notes Matteo

cd /Users/matteovalle/Documents/GitHub/vallematteo.github.io/vallematteo.github.io/flood_credit_risk_abm


TO RUN THE MODEL:
run.py --tot_step 10 --grid_radius 50 --acpt_score 200 --folder 'output/test' --settings 2 --repeats 3 --add_agents

run.py --tot_step 10 --grid_radius 50 --acpt_score 200 --folder 'output/test' --settings 2 --repeats 3  #if you don't want agents

### Unresolved issues
-  s_i: unclear the function: is the shere of income coefficient that remain unspent. also s_i or s_i(I_d) ?? unclear notation
- 'gev_config3.csv' is missing for new agents
- acpt_score unclear what is the function of this parameter
- also in model.py it seems i need two files for the KDE. stored in folder preparation:  ['kde_model_3d.joblib', 'kde_model_nvm.joblib'] line 32 model.py


your_project/
├── run.py
├── gev_config3.csv
├── gev_config2.csv
├── gev_config.csv
├── models/
│   		└── ESGMotgageModel.py
└── output/
    		└── test/
