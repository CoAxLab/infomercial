SHELL=/bin/bash -O expand_aliases
DATA_PATH=/Users/qualia/Code/infomercial/data


# 3-28-2019
#
# Testing the CL with a short one hot bandit exp
exp1:
	parallel -j 1 -v \
			--joblog '$(DATA_PATH)/exp1.log' \
			--nice 19 --delay 2 --colsep ',' \
			'info_bandit.py --env_name BanditOneHot2-v0 --num_episodes=10 --policy_mode='meta' --tie_break='next' --tie_threshold=1e-8 --lr=.1 --save=data/exp1_{1}.pkl --seed_value={1}' ::: 1 2