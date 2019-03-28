SHELL=/bin/bash -O expand_aliases
# DATA_PATH=/Users/qualia/Code/infomercial/data
DATA_PATH=/home/stitch/Code/infomercial/data/

# --------------------------------------------------------------------------
# 3-28-2019
#
# Testing the CL with a short one hot bandit exp
exp1:
	parallel -j 1 -v \
			--joblog '$(DATA_PATH)/exp1.log' \
			--nice 19 --delay 2 --colsep ',' \
			'info_bandit.py --env_name BanditOneHot2-v0 --num_episodes=10 --policy_mode='meta' --tie_break='next' --tie_threshold=1e-8 --lr=.1 --save=$(DATA_PATH)/exp1_{1}.pkl --seed_value={1}' ::: 1 2

# Run several bandits with the same parameters drawn from some hand tuning
# done in `exp_info_bandit.ipynb`
#
# lr = .1; epsilon = 1e-8
# N_trials = 10000; M_exps = 50
exp2:
	parallel -j 40 -v \
			--joblog '$(DATA_PATH)/exp2.log' \
			--nice 19 --delay 2 --colsep ',' \
			'info_bandit.py --env_name {2}-v0 --num_episodes=10000 --policy_mode='meta' --tie_break='next' --tie_threshold=1e-8 --lr=.1 --save=$(DATA_PATH)/exp2_{2}_{1}.pkl --seed_value={1}' ::: {1..50} ::: BanditOneHot2-v0 BanditOneHot10-v0 BanditOneHot121-v0 BanditOneHot1000-v0 BanditOneHigh2 BanditOneHigh10 BanditOneHigh121 BanditOneHigh1000 BanditHardAndSparse2 BanditHardAndSparse10 BanditHardAndSparse121 BanditHardAndSparse1000