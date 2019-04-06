SHELL=/bin/bash -O expand_aliases
DATA_PATH=/Users/qualia/Code/infomercial/data
# DATA_PATH=/home/stitch/Code/infomercial/data/

# ----------------------------------------------------------------------------
# 3-28-2019
#
# Testing the CL with a short one hot bandit exp
exp1:
	parallel -j 1 -v \
			--joblog '$(DATA_PATH)/exp1.log' \
			--nice 19 --delay 2 --colsep ',' \
			'meta_bandit.py --env_name BanditOneHot2-v0 --num_episodes=10 --tie_break='next' --tie_threshold=1e-8 --lr=.1 --save=$(DATA_PATH)/exp1_{1}.pkl --interactive=False --debug=True --seed_value={1}' ::: 1 2

# As a first real exp, run several bandits with the same parameters 
# drawn from some hand tuning. See where were at, overall.
# Tuning done in `exp_meta_bandit.ipynb`. Not explicitly doc'ed.
#
# lr = .1; epsilon = 1e-8
# N_trials = 10000; M_exps = 50

# RESULTS: 
# - One hot solved easily. Skipping these in future exps. 
# - One high was solved with p_best=1, or near to that, but the solutions 
#   were not as fast or stable as some of my hand-tuned runs w/ different lr. 
# - Sparse was never solved to p_best=1. 2 arm came close. 
#   Others were at chance.
exp2:
	parallel -j 40 -v \
			--joblog '$(DATA_PATH)/exp2.log' \
			--nice 19 --delay 2 --colsep ',' \
			'meta_bandit.py --env_name {2} --num_episodes=10000  --tie_break='next' --tie_threshold=1e-8 --lr=.1 --save=$(DATA_PATH)/exp2_{2}_{1}.pkl --interactive=False --seed_value={1}' ::: {1..50} ::: BanditOneHot2-v0 BanditOneHot10-v0 BanditOneHot121-v0 BanditOneHot1000-v0 BanditOneHigh2-v0 BanditOneHigh10-v0 BanditOneHigh121-v0 BanditOneHigh1000-v0 BanditHardAndSparse2-v0 BanditHardAndSparse10-v0 BanditHardAndSparse121-v0 BanditHardAndSparse1000-v0

# ----------------------------------------------------------------------------
# 3-29-2019
# Try some parameter variations starting based on exp2, which was about 2/3 of
# the way there in terms of solving all the bandits.

# Lesser lr improves sparse, but destroys performance on one high. Gain on sparse are not really great.
# Lesser threshold also hamrd one high. Does improve sparse.

# hot take: high lr good. play with theshold more....

# lr = 0.01
exp3:
	parallel -j 40 -v \
			--joblog '$(DATA_PATH)/exp3.log' \
			--nice 19 --delay 2 --colsep ',' \
			'meta_bandit.py --env_name {2} --num_episodes=10000  --tie_break='next' --tie_threshold=1e-8 --lr=.01 --save=$(DATA_PATH)/exp3_{2}_{1}.pkl --interactive=False --seed_value={1}' ::: {1..50} ::: BanditOneHigh2-v0 BanditOneHigh10-v0 BanditOneHigh121-v0 BanditOneHigh1000-v0 BanditHardAndSparse2-v0 BanditHardAndSparse10-v0 BanditHardAndSparse121-v0 BanditHardAndSparse1000-v0

# lr = 0.001
exp4:
	parallel -j 40 -v \
			--joblog '$(DATA_PATH)/exp4.log' \
			--nice 19 --delay 2 --colsep ',' \
			'meta_bandit.py --env_name {2} --num_episodes=10000 --tie_break='next' --tie_threshold=1e-8 --lr=0.001 --save=$(DATA_PATH)/exp4_{2}_{1}.pkl --interactive=False --seed_value={1}' ::: {1..50} ::: BanditOneHigh2-v0 BanditOneHigh10-v0 BanditOneHigh121-v0 BanditOneHigh1000-v0 BanditHardAndSparse2-v0 BanditHardAndSparse10-v0 BanditHardAndSparse121-v0 BanditHardAndSparse1000-v0

# lr = 0.000001
exp5:
	parallel -j 40 -v \
			--joblog '$(DATA_PATH)/exp5.log' \
			--nice 19 --delay 2 --colsep ',' \
			'meta_bandit.py --env_name {2} --num_episodes=10000 --tie_break='next' --tie_threshold=1e-8 --lr=0.000001 --save=$(DATA_PATH)/exp5_{2}_{1}.pkl --interactive=False --seed_value={1}' ::: {1..50} ::: BanditOneHigh2-v0 BanditOneHigh10-v0 BanditOneHigh121-v0 BanditOneHigh1000-v0 BanditHardAndSparse2-v0 BanditHardAndSparse10-v0 BanditHardAndSparse121-v0 BanditHardAndSparse1000-v0

# tie_threshold = 1e-9
exp6:
	parallel -j 40 -v \
			--joblog '$(DATA_PATH)/exp6.log' \
			--nice 19 --delay 2 --colsep ',' \
			'meta_bandit.py --env_name {2} --num_episodes=10000 --tie_break='next' --tie_threshold=1e-9 --lr=.1 --save=$(DATA_PATH)/exp6_{2}_{1}.pkl --interactive=False --seed_value={1}' ::: {1..50} ::: BanditOneHigh2-v0 BanditOneHigh10-v0 BanditOneHigh121-v0 BanditOneHigh1000-v0 BanditHardAndSparse2-v0 BanditHardAndSparse10-v0 BanditHardAndSparse121-v0 BanditHardAndSparse1000-v0

# lr=0.01, tie_threshold = 1e-9
exp7:
	parallel -j 40 -v \
			--joblog '$(DATA_PATH)/exp7.log' \
			--nice 19 --delay 2 --colsep ',' \
			'meta_bandit.py --env_name {2} --num_episodes=10000 --tie_break='next' --tie_threshold=1e-9 --lr=.01 --save=$(DATA_PATH)/exp7_{2}_{1}.pkl --interactive=False --seed_value={1}' ::: {1..50} ::: BanditOneHigh2-v0 BanditOneHigh10-v0 BanditOneHigh121-v0 BanditOneHigh1000-v0 BanditHardAndSparse2-v0 BanditHardAndSparse10-v0 BanditHardAndSparse121-v0 BanditHardAndSparse1000-v0

# lr=0.000001, tie_threshold = 1e-9
exp8:
	parallel -j 40 -v \
			--joblog '$(DATA_PATH)/exp8.log' \
			--nice 19 --delay 2 --colsep ',' \
			'meta_bandit.py --env_name {2} --num_episodes=10000 --tie_break='next' --tie_threshold=1e-9 --lr=0.000001 --save=$(DATA_PATH)/exp8_{2}_{1}.pkl --interactive=False --seed_value={1}' ::: {1..50} ::: BanditOneHigh2-v0 BanditOneHigh10-v0 BanditOneHigh121-v0 BanditOneHigh1000-v0 BanditHardAndSparse2-v0 BanditHardAndSparse10-v0 BanditHardAndSparse121-v0 BanditHardAndSparse1000-v0

# ----------------------------------------------------------------------------
# 3-31-2019
# Quest for on true parameter set continues....
# After exp3-8, trying high lr with more-ahem-exploration of the tie_threshold.

# lr = .1; tie_threshold = 1e-9
exp9:
	parallel -j 40 -v \
			--joblog '$(DATA_PATH)/exp9.log' \
			--nice 19 --delay 2 --colsep ',' \
			'meta_bandit.py --env_name {2} --num_episodes=10000 --tie_break='next' --tie_threshold=1e-9 --lr=0.1 --save=$(DATA_PATH)/exp9_{2}_{1}.pkl --interactive=False --seed_value={1}' ::: {1..50} ::: BanditOneHigh2-v0 BanditOneHigh10-v0 BanditOneHigh121-v0 BanditOneHigh1000-v0 BanditHardAndSparse2-v0 BanditHardAndSparse10-v0 BanditHardAndSparse121-v0 BanditHardAndSparse1000-v0

# lr = .1; tie_threshold = 1e-10
#
# Sum: exp9-10 decreasing threshold not helpful w/ lr = 1. I expected the opposite.
exp10:
	parallel -j 40 -v \
			--joblog '$(DATA_PATH)/exp10.log' \
			--nice 19 --delay 2 --colsep ',' \
			'meta_bandit.py --env_name {2} --num_episodes=10000 --tie_break='next' --tie_threshold=1e-10 --lr=0.1 --save=$(DATA_PATH)/exp10_{2}_{1}.pkl --interactive=False --seed_value={1}' ::: {1..50} ::: BanditOneHigh2-v0 BanditOneHigh10-v0 BanditOneHigh121-v0 BanditOneHigh1000-v0 BanditHardAndSparse2-v0 BanditHardAndSparse10-v0 BanditHardAndSparse121-v0 BanditHardAndSparse1000-v0


# lr = .2; tie_threshold = 1e-8
# 
# Sum:  OneHigh121 shows an approach to 1, that a large LOSS in p_best with learning. First time I've seen a loss. Don't really understand how that can be!
exp11:
	parallel -j 40 -v \
			--joblog '$(DATA_PATH)/exp11.log' \
			--nice 19 --delay 2 --colsep ',' \
			'meta_bandit.py --env_name {2} --num_episodes=10000 --tie_break='next' --tie_threshold=1e-8 --lr=0.2 --save=$(DATA_PATH)/exp11_{2}_{1}.pkl --interactive=False --seed_value={1}' ::: {1..50} ::: BanditOneHigh2-v0 BanditOneHigh10-v0 BanditOneHigh121-v0 BanditOneHigh1000-v0 BanditHardAndSparse2-v0 BanditHardAndSparse10-v0 BanditHardAndSparse121-v0 BanditHardAndSparse1000-v0

# lr = .2; tie_threshold = 1e-9
#
# 121 again shows a loss (see exp11) however it is MUCH more severe here.
exp12:
	parallel -j 40 -v \
			--joblog '$(DATA_PATH)/exp12.log' \
			--nice 19 --delay 2 --colsep ',' \
			'meta_bandit.py --env_name {2} --num_episodes=10000 --tie_break='next' --tie_threshold=1e-9 --lr=0.2 --save=$(DATA_PATH)/exp12_{2}_{1}.pkl --interactive=False --seed_value={1}' ::: {1..50} ::: BanditOneHigh2-v0 BanditOneHigh10-v0 BanditOneHigh121-v0 BanditOneHigh1000-v0 BanditHardAndSparse2-v0 BanditHardAndSparse10-v0 BanditHardAndSparse121-v0 BanditHardAndSparse1000-v0

# lr = .2; tie_threshold = 1e-10
# 
# Sum: loss on 121 again. No improvement otherwise.
# lr = 0.2 is just too high?
exp13:
	parallel -j 40 -v \
			--joblog '$(DATA_PATH)/exp13.log' \
			--nice 19 --delay 2 --colsep ',' \
			'meta_bandit.py --env_name {2} --num_episodes=10000 --tie_break='next' --tie_threshold=1e-10 --lr=0.2 --save=$(DATA_PATH)/exp13_{2}_{1}.pkl --interactive=False --seed_value={1}' ::: {1..50} ::: BanditOneHigh2-v0 BanditOneHigh10-v0 BanditOneHigh121-v0 BanditOneHigh1000-v0 BanditHardAndSparse2-v0 BanditHardAndSparse10-v0 BanditHardAndSparse121-v0 BanditHardAndSparse1000-v0

# lr = .1; tie_threshold = 1e-6
#
# Sum: oneHigh all solved. thresh was too low! No sparse solved. 
#
# - Maybe should I start thinking about the lr/thresh ratio. 
# Exploring that way?
exp14:
	parallel -j 40 -v \
			--joblog '$(DATA_PATH)/exp14.log' \
			--nice 19 --delay 2 --colsep ',' \
			'meta_bandit.py --env_name {2} --num_episodes=10000 --tie_break='next' --tie_threshold=1e-6 --lr=0.1 --save=$(DATA_PATH)/exp14_{2}_{1}.pkl --interactive=False --seed_value={1}' ::: {1..50} ::: BanditOneHigh2-v0 BanditOneHigh10-v0 BanditOneHigh121-v0 BanditOneHigh1000-v0 BanditHardAndSparse2-v0 BanditHardAndSparse10-v0 BanditHardAndSparse121-v0 BanditHardAndSparse1000-v0

# lr = .1; tie_threshold = 1e-4
#
# Sum: OneHigh NEARLY solved VERY fast. Back to near linear answers for this task! Some instability. Do a run w/ tie_threshold = 1e-3.
# 
# Sparse still looking quite poor; In single trials I was nailing these...
# not sure what I was doing different. 
#
# - The dist around p_best is consistent enough that single trials are a
# reasonable path to faster tuning. Esp for Sparse.
exp15:
	parallel -j 40 -v \
			--joblog '$(DATA_PATH)/exp15.log' \
			--nice 19 --delay 2 --colsep ',' \
			'meta_bandit.py --env_name {2} --num_episodes=10000 --tie_break='next' --tie_threshold=1e-4 --lr=0.1 --save=$(DATA_PATH)/exp15_{2}_{1}.pkl --interactive=False --seed_value={1}' ::: {1..50} ::: BanditOneHigh2-v0 BanditOneHigh10-v0 BanditOneHigh121-v0 BanditOneHigh1000-v0 BanditHardAndSparse2-v0 BanditHardAndSparse10-v0 BanditHardAndSparse121-v0 BanditHardAndSparse1000-v0

# ----------------------------------------------------------------------------
# 4-1-2019
# lr = .1; tie_threshold = 1e-3
# 
# Sum: On Onehigh instability over last few thousand trials increased 
# compared to exp15. I was hoping for the opposite. Try 1e-5 next. 
# Sparse still poorly.
exp16:
	parallel -j 40 -v \
			--joblog '$(DATA_PATH)/exp16.log' \
			--nice 19 --delay 2 --colsep ',' \
			'meta_bandit.py --env_name {2} --num_episodes=10000 --tie_break='next' --tie_threshold=1e-3 --lr=0.1 --save=$(DATA_PATH)/exp16_{2}_{1}.pkl --interactive=False --seed_value={1}' ::: {1..50} ::: BanditOneHigh2-v0 BanditOneHigh10-v0 BanditOneHigh121-v0 BanditOneHigh1000-v0 BanditHardAndSparse2-v0 BanditHardAndSparse10-v0 BanditHardAndSparse121-v0 BanditHardAndSparse1000-v0


# lr = .1; tie_threshold = 1e-5
# 
# Sum: For oneHigh, performance improved compared to exp16. 1000 still not quit perfect but its p_best > 0.95. OneHigh2,10,121 all quickly converged.
# Sparse still poorly. ...Keep this thresh? Try a very low lr? (low lr worked 
# well in hand tuning, IIRC).
exp17:
	parallel -j 40 -v \
			--joblog '$(DATA_PATH)/exp17.log' \
			--nice 19 --delay 2 --colsep ',' \
			'meta_bandit.py --env_name {2} --num_episodes=10000 --tie_break='next' --tie_threshold=1e-5 --lr=0.1 --save=$(DATA_PATH)/exp17_{2}_{1}.pkl --interactive=False --seed_value={1}' ::: {1..50} ::: BanditOneHigh2-v0 BanditOneHigh10-v0 BanditOneHigh121-v0 BanditOneHigh1000-v0 BanditHardAndSparse2-v0 BanditHardAndSparse10-v0 BanditHardAndSparse121-v0 BanditHardAndSparse1000-v0

# lr = .000001; tie_threshold = 1e-5
# 
# Sum: oneHigh2,10,121 quickly find p_best. oneHigh1000 never converged
# Sparse2 converged. Sparse10  p_best=0.8 or so. 121 and 1000 are at chance.
# Confused. Re-visit tuning runs. What is happening? Why are sparse solns poss.
# in these exps. What did I do diff?
# 
# Overall, tie_threshold seems to be more critical that lr is....
# Will need to do a full sensitivity test. 
# 
# Should also plot p_best for all bandits/exp so far....
exp18:
	parallel -j 40 -v \
			--joblog '$(DATA_PATH)/exp18.log' \
			--nice 19 --delay 2 --colsep ',' \
			'meta_bandit.py --env_name {2} --num_episodes=10000 --tie_break='next' --tie_threshold=1e-5 --lr=0.000001 --save=$(DATA_PATH)/exp18_{2}_{1}.pkl --interactive=False --seed_value={1}' ::: {1..50} ::: BanditOneHigh2-v0 BanditOneHigh10-v0 BanditOneHigh121-v0 BanditOneHigh1000-v0 BanditHardAndSparse2-v0 BanditHardAndSparse10-v0 BanditHardAndSparse121-v0 BanditHardAndSparse1000-v0

# ---------------------------------------------------------------------------
# 4-2-2019
# Testing the CL for epsilon_bandit.py
exp19:
	parallel -j 1 -v \
			--joblog '$(DATA_PATH)/exp19.log' \
			--nice 19 --delay 2 --colsep ',' \
			'epsilon_bandit.py --env_name BanditOneHigh2-v0 --num_episodes=10 --epsilon=0.1 --epsilon_decay_tau=0.001 --lr=.1 --save=$(DATA_PATH)/exp19_{1}.pkl --interactive=False --debug=True --seed_value={1}' ::: 1 2

# Testing the CL for beta_bandit.py
exp20:
	parallel -j 1 -v \
			--joblog '$(DATA_PATH)/exp20.log' \
			--nice 19 --delay 2 --colsep ',' \
			'beta_bandit.py --env_name BanditOneHigh2-v0 --num_episodes=10 --beta=1.0 --tie_break='next' --tie_threshold=1e-8 --lr=.1 --save=$(DATA_PATH)/exp20_{1}.pkl --interactive=False --debug=True --seed_value={1}' ::: 1 2

# ---------------------------------------------------------------------------
# 4-5-2019
# Tune some models
exp21:
	-rm -rf $(DATA_PATH)/exp21/*
	tune_bandit.py $(DATA_PATH)/exp21 \
		--exp_name='epsilon_bandit' \
		--env_name=BanditOneHigh10-v0 \
		--num_episodes=10000 \
		--num_samples=10 \
		--training_iteration=1000 \
		--perturbation_interval=10 \
		--epsilon='(.01, .99)' \
		--epsilon_decay_tau='(0.0001, 0.1)' \
		--lr='(1e-6, 1e-1)'