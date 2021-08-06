SHELL=/bin/bash -O expand_aliases
# DATA_PATH=/Users/qualia/Code/infomercial/data
# DATA_PATH=/Volumes/Data/infomercial/data
DATA_PATH=/home/stitch/Code/infomercial/data/

# ----------------------------------------------------------------------------
# Test recipes for various agents/parameters
# 
# Note: This should run fine with main/HEAD. It is kept up to date, in other
# words. Or, well, it should be.

# WSLS tester - change as needed
test1:
	-rm -rf $(DATA_PATH)/test1*
	parallel -j 1 -v \
			--nice 19 --delay 2 --colsep ',' \
			'wsls_bandit.py --env_name=BanditOneHigh4-v0 --num_episodes=40 --tie_break='next' --tie_threshold=1e-4 --mode='KL' --lr_R=.1 --log_dir=$(DATA_PATH)/test1/run{1} --master_seed={1}' ::: {0..100}

# softbeta tester - change as needed
test2:
	-rm -rf $(DATA_PATH)/test2*
	parallel -j 1 -v \
			--nice 19 --delay 2 --colsep ',' \
			'softbeta_bandit.py --env_name=BanditOneHigh4-v0 --num_episodes=40 --temp=0.2 --beta=0.5 --bonus=0 --lr_R=.1 --log_dir=$(DATA_PATH)/test2/run{1}  --master_seed={1}' ::: {0..100}

# softcount tester - change as needed
test3:
	-rm -rf $(DATA_PATH)/test3*
	parallel -j 1 -v \
			--nice 19 --delay 2 --colsep ',' \
			'softcount_bandit.py --env_name=BanditOneHigh4-v0 --num_episodes=40 --temp=0.2 --beta=0.5 --lr_R=.1 --mode='UCB' --log_dir=$(DATA_PATH)/test3/run{1} --master_seed={1}' ::: {0..100}

# softentropy tester - change as needed
test4:
	-rm -rf $(DATA_PATH)/test4*
	parallel -j 1 -v \
			--nice 19 --delay 2 --colsep ',' \
			'softentropy_bandit.py --env_name=BanditOneHigh4-v0 --num_episodes=40 --temp=0.2 --beta=0.5 --lr_R=.1 --log_dir=$(DATA_PATH)/test4/run{1} --master_seed={1}' ::: {0..100}

# epgreedy tester - change as needed
test5:
	-rm -rf $(DATA_PATH)/test5*
	parallel -j 1 -v \
			--nice 19 --delay 2 --colsep ',' \
			'epsilon_bandit.py --env_name=BanditOneHigh4-v0 --num_episodes=40 --epsilon=0.1 --epsilon_decay_tau=0 --lr_R=.1 --log_dir=$(DATA_PATH)/test5/run{1} --master_seed={1}' ::: {0..100}

# ----------------------------------------------------------------------------
# 3-28-2019
#
# Testing the CL with a short one hot bandit exp
exp1:
	parallel -j 1 -v \
			--joblog '$(DATA_PATH)/exp1.log' \
			--nice 19 --delay 2 --colsep ',' \
			'wsls_bandit.py --env_name BanditOneHot2-v0 --num_episodes=10 --tie_break='next' --tie_threshold=1e-8 --lr_R=.1 --save=$(DATA_PATH)/exp1_{1}.pkl --interactive=False --debug=True --master_seed={1}' ::: 1 2

# As a first real exp, run several bandits with the same parameters 
# drawn from some hand tuning. See where were at, overall.
# Tuning done in `exp_wsls_bandit.ipynb`. Not explicitly doc'ed.
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
			'wsls_bandit.py --env_name {2} --num_episodes=10000  --tie_break='next' --tie_threshold=1e-8 --lr=.1 --save=$(DATA_PATH)/exp2_{2}_{1}.pkl --interactive=False --master_seed={1}' ::: {1..50} ::: BanditOneHot2-v0 BanditOneHot10-v0 BanditOneHot121-v0 BanditOneHot1000-v0 BanditOneHigh2-v0 BanditOneHigh10-v0 BanditOneHigh121-v0 BanditOneHigh1000-v0 BanditHardAndSparse2-v0 BanditHardAndSparse10-v0 BanditHardAndSparse121-v0 BanditHardAndSparse1000-v0

# ----------------------------------------------------------------------------
# 3-29-2019
# Try some parameter variations starting based on exp2, which was about 2/3 of
# the way there in terms of solving all the bandits.

# Lesser lr improves sparse, but destroys performance on one high. Gain on sparse are not really great.
# Lesser threshold also hamrd one high. Does improve sparse.

# SUM: high lr good. play with theshold more....
# - See notebooks/exp21-8_analysis.ipynb for full analysis
# - as well as individual notebooks/exp2_analysis.ipynb
#                         notebooks/exp3_analysis.ipynb
#                         and so on

# lr = 0.01
exp3:
	parallel -j 40 -v \
			--joblog '$(DATA_PATH)/exp3.log' \
			--nice 19 --delay 2 --colsep ',' \
			'wsls_bandit.py --env_name {2} --num_episodes=10000  --tie_break='next' --tie_threshold=1e-8 --lr=.01 --save=$(DATA_PATH)/exp3_{2}_{1}.pkl --interactive=False --master_seed={1}' ::: {1..50} ::: BanditOneHigh2-v0 BanditOneHigh10-v0 BanditOneHigh121-v0 BanditOneHigh1000-v0 BanditHardAndSparse2-v0 BanditHardAndSparse10-v0 BanditHardAndSparse121-v0 BanditHardAndSparse1000-v0

# lr = 0.001
exp4:
	parallel -j 40 -v \
			--joblog '$(DATA_PATH)/exp4.log' \
			--nice 19 --delay 2 --colsep ',' \
			'wsls_bandit.py --env_name {2} --num_episodes=10000 --tie_break='next' --tie_threshold=1e-8 --lr=0.001 --save=$(DATA_PATH)/exp4_{2}_{1}.pkl --interactive=False --master_seed={1}' ::: {1..50} ::: BanditOneHigh2-v0 BanditOneHigh10-v0 BanditOneHigh121-v0 BanditOneHigh1000-v0 BanditHardAndSparse2-v0 BanditHardAndSparse10-v0 BanditHardAndSparse121-v0 BanditHardAndSparse1000-v0

# lr = 0.000001
exp5:
	parallel -j 40 -v \
			--joblog '$(DATA_PATH)/exp5.log' \
			--nice 19 --delay 2 --colsep ',' \
			'wsls_bandit.py --env_name {2} --num_episodes=10000 --tie_break='next' --tie_threshold=1e-8 --lr=0.000001 --save=$(DATA_PATH)/exp5_{2}_{1}.pkl --interactive=False --master_seed={1}' ::: {1..50} ::: BanditOneHigh2-v0 BanditOneHigh10-v0 BanditOneHigh121-v0 BanditOneHigh1000-v0 BanditHardAndSparse2-v0 BanditHardAndSparse10-v0 BanditHardAndSparse121-v0 BanditHardAndSparse1000-v0

# tie_threshold = 1e-9
exp6:
	parallel -j 40 -v \
			--joblog '$(DATA_PATH)/exp6.log' \
			--nice 19 --delay 2 --colsep ',' \
			'wsls_bandit.py --env_name {2} --num_episodes=10000 --tie_break='next' --tie_threshold=1e-9 --lr=.1 --save=$(DATA_PATH)/exp6_{2}_{1}.pkl --interactive=False --master_seed={1}' ::: {1..50} ::: BanditOneHigh2-v0 BanditOneHigh10-v0 BanditOneHigh121-v0 BanditOneHigh1000-v0 BanditHardAndSparse2-v0 BanditHardAndSparse10-v0 BanditHardAndSparse121-v0 BanditHardAndSparse1000-v0

# lr=0.01, tie_threshold = 1e-9
exp7:
	parallel -j 40 -v \
			--joblog '$(DATA_PATH)/exp7.log' \
			--nice 19 --delay 2 --colsep ',' \
			'wsls_bandit.py --env_name {2} --num_episodes=10000 --tie_break='next' --tie_threshold=1e-9 --lr=.01 --save=$(DATA_PATH)/exp7_{2}_{1}.pkl --interactive=False --master_seed={1}' ::: {1..50} ::: BanditOneHigh2-v0 BanditOneHigh10-v0 BanditOneHigh121-v0 BanditOneHigh1000-v0 BanditHardAndSparse2-v0 BanditHardAndSparse10-v0 BanditHardAndSparse121-v0 BanditHardAndSparse1000-v0

# lr=0.000001, tie_threshold = 1e-9
exp8:
	parallel -j 40 -v \
			--joblog '$(DATA_PATH)/exp8.log' \
			--nice 19 --delay 2 --colsep ',' \
			'wsls_bandit.py --env_name {2} --num_episodes=10000 --tie_break='next' --tie_threshold=1e-9 --lr=0.000001 --save=$(DATA_PATH)/exp8_{2}_{1}.pkl --interactive=False --master_seed={1}' ::: {1..50} ::: BanditOneHigh2-v0 BanditOneHigh10-v0 BanditOneHigh121-v0 BanditOneHigh1000-v0 BanditHardAndSparse2-v0 BanditHardAndSparse10-v0 BanditHardAndSparse121-v0 BanditHardAndSparse1000-v0

# ----------------------------------------------------------------------------
# 3-31-2019
# Quest for on true parameter set continues....
# After exp3-8, trying high lr with more-ahem-exploration of the tie_threshold.

# lr = .1; tie_threshold = 1e-9
# - See notebooks/exp9_analysis.ipynb for full analysis
exp9:
	parallel -j 40 -v \
			--joblog '$(DATA_PATH)/exp9.log' \
			--nice 19 --delay 2 --colsep ',' \
			'wsls_bandit.py --env_name {2} --num_episodes=10000 --tie_break='next' --tie_threshold=1e-9 --lr=0.1 --save=$(DATA_PATH)/exp9_{2}_{1}.pkl --interactive=False --master_seed={1}' ::: {1..50} ::: BanditOneHigh2-v0 BanditOneHigh10-v0 BanditOneHigh121-v0 BanditOneHigh1000-v0 BanditHardAndSparse2-v0 BanditHardAndSparse10-v0 BanditHardAndSparse121-v0 BanditHardAndSparse1000-v0

# lr = .1; tie_threshold = 1e-10
#
# SUM: exp9-10 decreasing threshold not helpful w/ lr = 1. I expected the opposite.
# - See notebooks/exp10_analysis.ipynb for full analysis
exp10:
	parallel -j 40 -v \
			--joblog '$(DATA_PATH)/exp10.log' \
			--nice 19 --delay 2 --colsep ',' \
			'wsls_bandit.py --env_name {2} --num_episodes=10000 --tie_break='next' --tie_threshold=1e-10 --lr=0.1 --save=$(DATA_PATH)/exp10_{2}_{1}.pkl --interactive=False --master_seed={1}' ::: {1..50} ::: BanditOneHigh2-v0 BanditOneHigh10-v0 BanditOneHigh121-v0 BanditOneHigh1000-v0 BanditHardAndSparse2-v0 BanditHardAndSparse10-v0 BanditHardAndSparse121-v0 BanditHardAndSparse1000-v0


# lr = .2; tie_threshold = 1e-8
# 
# SUM:  OneHigh121 shows an approach to 1, that a large LOSS in p_best with learning. First time I've seen a loss. Don't really understand how that can be!
# - See notebooks/exp11_analysis.ipynb for full analysis
exp11:
	parallel -j 40 -v \
			--joblog '$(DATA_PATH)/exp11.log' \
			--nice 19 --delay 2 --colsep ',' \
			'wsls_bandit.py --env_name {2} --num_episodes=10000 --tie_break='next' --tie_threshold=1e-8 --lr=0.2 --save=$(DATA_PATH)/exp11_{2}_{1}.pkl --interactive=False --master_seed={1}' ::: {1..50} ::: BanditOneHigh2-v0 BanditOneHigh10-v0 BanditOneHigh121-v0 BanditOneHigh1000-v0 BanditHardAndSparse2-v0 BanditHardAndSparse10-v0 BanditHardAndSparse121-v0 BanditHardAndSparse1000-v0

# lr = .2; tie_threshold = 1e-9
#
# 121 again shows a loss (see exp11) however it is MUCH more severe here.
# - See notebooks/exp12_analysis.ipynb for full analysis
exp12:
	parallel -j 40 -v \
			--joblog '$(DATA_PATH)/exp12.log' \
			--nice 19 --delay 2 --colsep ',' \
			'wsls_bandit.py --env_name {2} --num_episodes=10000 --tie_break='next' --tie_threshold=1e-9 --lr=0.2 --save=$(DATA_PATH)/exp12_{2}_{1}.pkl --interactive=False --master_seed={1}' ::: {1..50} ::: BanditOneHigh2-v0 BanditOneHigh10-v0 BanditOneHigh121-v0 BanditOneHigh1000-v0 BanditHardAndSparse2-v0 BanditHardAndSparse10-v0 BanditHardAndSparse121-v0 BanditHardAndSparse1000-v0

# lr = .2; tie_threshold = 1e-10
# 
# SUM: loss on 121 again. No improvement otherwise.
# lr = 0.2 is just too high?
# - See notebooks/exp13_analysis.ipynb for full analysis
exp13:
	parallel -j 40 -v \
			--joblog '$(DATA_PATH)/exp13.log' \
			--nice 19 --delay 2 --colsep ',' \
			'wsls_bandit.py --env_name {2} --num_episodes=10000 --tie_break='next' --tie_threshold=1e-10 --lr=0.2 --save=$(DATA_PATH)/exp13_{2}_{1}.pkl --interactive=False --master_seed={1}' ::: {1..50} ::: BanditOneHigh2-v0 BanditOneHigh10-v0 BanditOneHigh121-v0 BanditOneHigh1000-v0 BanditHardAndSparse2-v0 BanditHardAndSparse10-v0 BanditHardAndSparse121-v0 BanditHardAndSparse1000-v0

# lr = .1; tie_threshold = 1e-6
#
# SUM: oneHigh all solved. thresh was too low! No sparse solved. 
#
# - Maybe should I start thinking about the lr/thresh ratio. 
# Exploring that way?
# - See notebooks/exp14_analysis.ipynb for full analysis
exp14:
	parallel -j 40 -v \
			--joblog '$(DATA_PATH)/exp14.log' \
			--nice 19 --delay 2 --colsep ',' \
			'wsls_bandit.py --env_name {2} --num_episodes=10000 --tie_break='next' --tie_threshold=1e-6 --lr=0.1 --save=$(DATA_PATH)/exp14_{2}_{1}.pkl --interactive=False --master_seed={1}' ::: {1..50} ::: BanditOneHigh2-v0 BanditOneHigh10-v0 BanditOneHigh121-v0 BanditOneHigh1000-v0 BanditHardAndSparse2-v0 BanditHardAndSparse10-v0 BanditHardAndSparse121-v0 BanditHardAndSparse1000-v0

# lr = .1; tie_threshold = 1e-4
#
# SUM: OneHigh NEARLY solved VERY fast. Back to near linear answers for this task! Some instability. Do a run w/ tie_threshold = 1e-3.
# 
# Sparse still looking quite poor; In single trials I was nailing these...
# not sure what I was doing different. 
#
# - The dist around p_best is consistent enough that single trials are a
# reasonable path to faster tuning. Esp for Sparse.
#
# - See notebooks/exp15_analysis.ipynb for full analysis
exp15:
	parallel -j 40 -v \
			--joblog '$(DATA_PATH)/exp15.log' \
			--nice 19 --delay 2 --colsep ',' \
			'wsls_bandit.py --env_name {2} --num_episodes=10000 --tie_break='next' --tie_threshold=1e-4 --lr=0.1 --save=$(DATA_PATH)/exp15_{2}_{1}.pkl --interactive=False --master_seed={1}' ::: {1..50} ::: BanditOneHigh2-v0 BanditOneHigh10-v0 BanditOneHigh121-v0 BanditOneHigh1000-v0 BanditHardAndSparse2-v0 BanditHardAndSparse10-v0 BanditHardAndSparse121-v0 BanditHardAndSparse1000-v0

# ----------------------------------------------------------------------------
# 4-1-2019
# lr = .1; tie_threshold = 1e-3
# 
# SUM: On Onehigh instability over last few thousand trials increased 
# compared to exp15. I was hoping for the opposite. Try 1e-5 next. 
# Sparse still poorly.
# - See notebooks/exp16_analysis.ipynb for full analysis
exp16:
	parallel -j 40 -v \
			--joblog '$(DATA_PATH)/exp16.log' \
			--nice 19 --delay 2 --colsep ',' \
			'wsls_bandit.py --env_name {2} --num_episodes=10000 --tie_break='next' --tie_threshold=1e-3 --lr=0.1 --save=$(DATA_PATH)/exp16_{2}_{1}.pkl --interactive=False --master_seed={1}' ::: {1..50} ::: BanditOneHigh2-v0 BanditOneHigh10-v0 BanditOneHigh121-v0 BanditOneHigh1000-v0 BanditHardAndSparse2-v0 BanditHardAndSparse10-v0 BanditHardAndSparse121-v0 BanditHardAndSparse1000-v0


# lr = .1; tie_threshold = 1e-5
# 
# SUM: For oneHigh, performance improved compared to exp16. 1000 still not quit perfect but its p_best > 0.95. OneHigh2,10,121 all quickly converged.
# Sparse still poorly. ...Keep this thresh? Try a very low lr? (low lr worked 
# well in hand tuning, IIRC).
# - See notebooks/exp17_analysis.ipynb for full analysis
exp17:
	parallel -j 40 -v \
			--joblog '$(DATA_PATH)/exp17.log' \
			--nice 19 --delay 2 --colsep ',' \
			'wsls_bandit.py --env_name {2} --num_episodes=10000 --tie_break='next' --tie_threshold=1e-5 --lr=0.1 --save=$(DATA_PATH)/exp17_{2}_{1}.pkl --interactive=False --master_seed={1}' ::: {1..50} ::: BanditOneHigh2-v0 BanditOneHigh10-v0 BanditOneHigh121-v0 BanditOneHigh1000-v0 BanditHardAndSparse2-v0 BanditHardAndSparse10-v0 BanditHardAndSparse121-v0 BanditHardAndSparse1000-v0

# lr = .000001; tie_threshold = 1e-5
# 
# SUM: oneHigh2,10,121 quickly find p_best. oneHigh1000 never converged
# Sparse2 converged. Sparse10  p_best=0.8 or so. 121 and 1000 are at chance.
# Confused. Re-visit tuning runs. What is happening? Why are sparse solns poss.
# in these exps. What did I do diff?
# 
# Overall, tie_threshold seems to be more critical that lr is....
# Will need to do a full sensitivity test. 
# 
# Should also plot p_best for all bandits/exp so far....
#
# - See notebooks/exp18_analysis.ipynb for full analysis
exp18:
	parallel -j 40 -v \
			--joblog '$(DATA_PATH)/exp18.log' \
			--nice 19 --delay 2 --colsep ',' \
			'wsls_bandit.py --env_name {2} --num_episodes=10000 --tie_break='next' --tie_threshold=1e-5 --lr=0.000001 --save=$(DATA_PATH)/exp18_{2}_{1}.pkl --interactive=False --master_seed={1}' ::: {1..50} ::: BanditOneHigh2-v0 BanditOneHigh10-v0 BanditOneHigh121-v0 BanditOneHigh1000-v0 BanditHardAndSparse2-v0 BanditHardAndSparse10-v0 BanditHardAndSparse121-v0 BanditHardAndSparse1000-v0

# ---------------------------------------------------------------------------
# 4-2-2019
# Testing the CL for epsilon_bandit.py
exp19:
	parallel -j 1 -v \
			--joblog '$(DATA_PATH)/exp19.log' \
			--nice 19 --delay 2 --colsep ',' \
			'epsilon_bandit.py --env_name BanditOneHigh2-v0 --num_episodes=10 --epsilon=0.1 --epsilon_decay_tau=0.001 --lr=.1 --save=$(DATA_PATH)/exp19_{1}.pkl --interactive=False --debug=True --master_seed={1}' ::: 1 2

# Testing the CL for beta_bandit.py
exp20:
	parallel -j 1 -v \
			--joblog '$(DATA_PATH)/exp20.log' \
			--nice 19 --delay 2 --colsep ',' \
			'beta_bandit.py --env_name BanditOneHigh2-v0 --num_episodes=10 --beta=1.0 --tie_break='next' --tie_threshold=1e-8 --lr=.1 --save=$(DATA_PATH)/exp20_{1}.pkl --interactive=False --debug=True --master_seed={1}' ::: 1 2

# ---------------------------------------------------------------------------
# 4-5-2019
# Tune epsilon_bandit: a quick test run
exp21:
	-rm -rf $(DATA_PATH)/exp21/*
	tune_bandit.py $(DATA_PATH)/exp21 \
		--exp_name='epsilon_bandit' \
		--env_name=BanditOneHigh10-v0 \
		--num_episodes=10 \
		--num_samples=10 \
		--training_iteration=20 \
		--perturbation_interval=2 \
		--epsilon='(.01, .99)' \
		--epsilon_decay_tau='(0.0001, 0.1)' \
		--lr='(1e-6, 1e-1)'

# Tune epsilon_bandit: first real try
# SUM: converged on having essentially no exploration.
#  'epsilon': 0.013352806530529619,
#  'epsilon_decay_tau': 0.08101846019197038,
#  'lr': 0.004639344318990854,
#  'total_R': 583.0
# It never find the best arm as a result.
# 
# - See notebooks/exp21_analysis.ipynb for full analysis

# I'm not sure if that's a problem with the method or
# the code. To try and diagnos going to run a 
# the other exps to see anything sensible happens.

# Note: to save space I deleted the detailed run data in exp22/*
# kept only exp22_best.pkl and exp22_sorted.pkl
# 
# - See notebooks/exp22_analysis.ipynb for full analysis
exp22:
	-rm -rf $(DATA_PATH)/exp22/*
	tune_bandit.py $(DATA_PATH)/exp22 \
		--exp_name='epsilon_bandit' \
		--env_name=BanditOneHigh1000-v0 \
		--num_episodes=3000 \
		--num_samples=120 \
		--training_iteration=100 \
		--perturbation_interval=1 \
		--epsilon='(.01, .99)' \
		--epsilon_decay_tau='(0.0001, 0.1)' \
		--lr='(1e-6, 1e-1)'

# ---------------------------------------------------------------------------
# 4-8-2019
# c37259fc6ba12e2ca8f49da1457218664a8b36ff
# First real opt for meta

# SUM: after running for 2 days this never converged. No sure why? PBT config?
# Model problems? 
# 
# - See notebooks/exp23_analysis.ipynb for full analysis
#
# NEXT: maoving to a simple random search, just go make some quick progress in 
# studing the alt models. Will revist PBT later...
exp23:
	-rm -rf $(DATA_PATH)/exp23/*
	tune_bandit.py $(DATA_PATH)/exp23 \
		--exp_name='wsls_bandit' \
		--env_name=BanditOneHigh1000-v0 \
		--num_episodes=3000 \
		--num_samples=120 \
		--training_iteration=100 \
		--perturbation_interval=1 \
		--tie_threshold='(1e-1, .1e-10)' \
		--lr='(1e-1, 1e-6)'

# First real opt for beta
exp24:
	-rm -rf $(DATA_PATH)/exp24/*
	tune_bandit.py $(DATA_PATH)/exp24 \
		--exp_name='beta_bandit' \
		--env_name=BanditOneHigh1000-v0 \
		--num_episodes=3000 \
		--num_samples=120 \
		--training_iteration=100 \
		--perturbation_interval=1 \
		--beta='(1e-3, 1e1)' \
		--lr='(1e-1, 1e-6)'


# ---------------------------------------------------------------------------
# 4-10-2019
# 2890330c3ac7b6e4aa2c58c31bf2860f04876c99

# Branched from master -> random_search. 
# Try: Random search for 100 draws, with 3 resamples
# 
# Opt beta
# SUM: Best params solved all oneHot/oneHigh. No progress on HardAndSparse
# The top 50 parameters (or 500) all give qbout equal performance on the 
# OneHigh tasks.
# Sensitivy was to beta, but it is complex. Very little to lr directly. 
# But there is probably a beta/lr interaction, given beta's complex trend,
# but hard to sus it out.
# Far as I can tell between beta: 0.04-0.08 gives the best total_R.
# 
# Meant to set beta between 0.1 - 10. Need to rerun this over a wider range.
# The sampling for lr is off too. Hmm....
# exp25:
# 	-rm -rf $(DATA_PATH)/exp25/*
# 	tune_bandit.py $(DATA_PATH)/exp25 \
# 		--exp_name='beta_bandit' \
# 		--env_name=BanditOneHigh1000-v0 \
# 		--num_episodes=3000 \
# 		--num_samples=500 \
# 		--beta='(0.001, 2)' \
# 		--lr='(0.001, 0.2)'

# # opt meta
# exp26:
# 	-rm -rf $(DATA_PATH)/exp26/*
# 	tune_bandit.py $(DATA_PATH)/exp26 \
# 		--exp_name='wsls_bandit' \
# 		--env_name=BanditOneHigh1000-v0 \
# 		--num_episodes=3000 \
# 		--num_samples=500 \
# 		--verbose=True \
# 		--tie_threshold='(1e-8, 0.1)' \
# 		--lr='(0.001, 0.2)'

# # opt epsilon
# exp27:
# 	-rm -rf $(DATA_PATH)/exp27/*
# 	tune_bandit.py $(DATA_PATH)/exp27 \
# 		--exp_name='epsilon_bandit' \
# 		--env_name=BanditOneHigh1000-v0 \
# 		--num_episodes=3000 \
# 		--num_samples=500 \
# 		--epsilon='(0.01, 0.99)' \
# 		--epsilon_decay_tau='(0.0001, 0.01)' \
# 		--lr='(0.001, 0.2)'

# ---------------------------------------------------------------------------
# 4-15-2019
# e929bc945e9bc55ae4b751c9a3b9d81062758a36
# Re-ran exp25-7 using a random search method that actually searches the asked 
# for parameter ranges. `ray` has a big bug that limits my ability to actually
# search params....
#
# SUM: only meta converged on the best. the other two were mess. this is at odds
# with the prior runs at exp25-7 with ray and PBT (even given the bug on search
# range). Clearly random search either needs many more samples or I need to
# move to a smarter tuning system. 

# Next: As a quick test, to keep things going, am re-run beta and epsilon w/ 5 times the samples overnight.

# opt beta
# - See notebooks/exp25_analysis.ipynb for full analysis
exp25:
	-rm -rf $(DATA_PATH)/exp25/*
	tune_bandit.py random $(DATA_PATH)/exp25 \
		--exp_name='beta_bandit' \
		--env_name=BanditOneHigh1000-v0 \
		--num_episodes=3000 \
		--num_samples=500 \
		--num_processes=40 \
		--beta='(0.001, 2)' \
		--lr='(0.001, 0.2)'

# opt meta
# - See notebooks/exp26_analysis.ipynb for full analysis
exp26:
	-rm -rf $(DATA_PATH)/exp26/*
	tune_bandit.py random $(DATA_PATH)/exp26 \
		--exp_name='wsls_bandit' \
		--env_name=BanditOneHigh1000-v0 \
		--num_episodes=3000 \
		--num_samples=500 \
		--num_processes=40 \
		--verbose=True \
		--tie_threshold='(1e-8, 0.1)' \
		--lr='(0.001, 0.2)'

# opt epsilon
# - See notebooks/exp27_analysis.ipynb for full analysis
exp27:
	-rm -rf $(DATA_PATH)/exp27/*
	tune_bandit.py random $(DATA_PATH)/exp27 \
		--exp_name='epsilon_bandit' \
		--env_name=BanditOneHigh1000-v0 \
		--num_episodes=3000 \
		--num_samples=500 \
		--num_processes=40 \
		--epsilon='(0.01, 0.99)' \
		--epsilon_decay_tau='(0.0001, 0.01)' \
		--lr='(0.001, 0.2)'

# ---------------------------------------------------------------------------
# 4-15-2019
# 5x more samples compared to 25, 27
# 18ebf12316a04bd0a3e76b394ab538475d77b737

# SUM: with a broader HP search, both beta and ep found the best arm. 

# beta
# - See notebooks/exp28_analysis.ipynb for full analysis
exp28:
	-rm -rf $(DATA_PATH)/exp28/*
	tune_bandit.py random $(DATA_PATH)/exp28 \
		--exp_name='beta_bandit' \
		--env_name=BanditOneHigh1000-v0 \
		--num_episodes=3000 \
		--num_samples=2500 \
		--num_processes=40 \
		--beta='(0.001, 2)' \
		--lr='(0.001, 0.2)'

# epsilon
# - See notebooks/exp29_analysis.ipynb for full analysis	
exp29:
	-rm -rf $(DATA_PATH)/exp29/*
	tune_bandit.py random $(DATA_PATH)/exp29 \
		--exp_name='epsilon_bandit' \
		--env_name=BanditOneHigh1000-v0 \
		--num_episodes=3000 \
		--num_samples=2500 \
		--num_processes=40 \
		--epsilon='(0.01, 0.99)' \
		--epsilon_decay_tau='(0.0001, 0.01)' \
		--lr='(0.001, 0.2)'

# ---------------------------------------------------------------------------
# 4-16-2019
# 5x more samples compared to 26
# d7ff0eb34e36c2b83cd65d23db2a45ccf39c0e34

# opt meta
# - See notebooks/exp30_analysis.ipynb for full analysis
exp30:
	-rm -rf $(DATA_PATH)/exp30/*
	tune_bandit.py random $(DATA_PATH)/exp30 \
		--exp_name='wsls_bandit' \
		--env_name=BanditOneHigh1000-v0 \
		--num_episodes=3000 \
		--num_samples=2500 \
		--num_processes=40 \
		--verbose=True \
		--tie_threshold='(1e-8, 0.1)' \
		--lr='(0.001, 0.2)'

# ---------------------------------------------------------------------------
# 4-16-2019
# Try a few bandits and algs w/ PBT tuning. How does it do overall
# and compared to random search (above)

# opt meta

# BanditOneHigh1000
exp31:
	-rm -rf $(DATA_PATH)/exp31/*
	tune_bandit.py pbt $(DATA_PATH)/exp31 \
		--exp_name='wsls_bandit' \
		--env_name=BanditOneHigh1000-v0 \
		--num_iterations=10 \
		--top_threshold=0.25 \
		--num_episodes=3000 \
		--num_samples=2500 \
		--num_processes=40 \
		--verbose=True \
		--tie_threshold='(1e-8, 0.1)' \
		--lr='(0.001, 0.2)'

# BanditOneHigh10
exp32:
	-rm -rf $(DATA_PATH)/exp32/*
	tune_bandit.py pbt $(DATA_PATH)/exp32 \
		--exp_name='wsls_bandit' \
		--env_name=BanditOneHigh10-v0 \
		--num_iterations=10 \
		--top_threshold=0.25 \
		--num_episodes=3000 \
		--num_samples=2500 \
		--num_processes=40 \
		--verbose=True \
		--tie_threshold='(1e-8, 0.1)' \
		--lr='(0.001, 0.2)'

# BanditHardAndSparse2
exp33:
	-rm -rf $(DATA_PATH)/exp33/*
	tune_bandit.py pbt $(DATA_PATH)/exp33 \
		--exp_name='wsls_bandit' \
		--env_name=BanditHardAndSparse2-v0 \
		--num_iterations=10 \
		--top_threshold=0.25 \
		--num_episodes=3000 \
		--num_samples=2500 \
		--num_processes=40 \
		--verbose=True \
		--tie_threshold='(1e-8, 0.1)' \
		--lr='(0.001, 0.2)'

# BanditHardAndSparse10
exp34:
	-rm -rf $(DATA_PATH)/exp34/*
	tune_bandit.py pbt $(DATA_PATH)/exp34 \
		--exp_name='wsls_bandit' \
		--env_name=BanditHardAndSparse10-v0 \
		--num_iterations=10 \
		--top_threshold=0.25 \
		--num_episodes=3000 \
		--num_samples=2500 \
		--num_processes=40 \
		--verbose=True \
		--tie_threshold='(1e-8, 0.1)' \
		--lr='(0.001, 0.2)'


# BanditHardAndSparse121
exp35:
	-rm -rf $(DATA_PATH)/exp35/*
	tune_bandit.py pbt $(DATA_PATH)/exp35 \
		--exp_name='wsls_bandit' \
		--env_name=BanditHardAndSparse121-v0 \
		--num_iterations=10 \
		--top_threshold=0.25 \
		--num_episodes=3000 \
		--num_samples=2500 \
		--num_processes=40 \
		--verbose=True \
		--tie_threshold='(1e-8, 0.1)' \
		--lr='(0.001, 0.2)'

# BanditHardAndSparse1000
exp36:
	-rm -rf $(DATA_PATH)/exp36/*
	tune_bandit.py pbt $(DATA_PATH)/exp36 \
		--exp_name='wsls_bandit' \
		--env_name=BanditHardAndSparse1000-v0 \
		--num_iterations=10 \
		--top_threshold=0.25 \
		--num_episodes=3000 \
		--num_samples=2500 \
		--num_processes=40 \
		--verbose=True \
		--tie_threshold='(1e-8, 0.1)' \
		--lr='(0.001, 0.2)'


# --------------------------------------------------------------------------
# 4-17-2019
# Based on Harper's [1] work connecting Bayes inference w/ evo game theory I
# developed a hyper opt scheme based on replictor dynamics.
#
# [1]: Harper, Marc. “The Replicator Equation as an Inference Dynamic.” 
# ArXiv:0911.1763 [Cs, Math], November 9, 2009. http://arxiv.org/abs/0911.1763.
#
# Let's test it on some hard bandits using my meta_policy.

# BanditHardAndSparse10
# SUM: best params did learn the best arm
# - See notebooks/exp37_analysis.ipynb for full analysis
exp37:
	-rm -rf $(DATA_PATH)/exp37/*
	tune_bandit.py replicator $(DATA_PATH)/exp37 \
		--exp_name='wsls_bandit' \
		--env_name=BanditHardAndSparse10-v0 \
		--num_iterations=10 \
		--num_episodes=100 \
		--num_replicators=120 \
		--num_processes=40 \
		--verbose=True \
		--tie_threshold='(1e-8, 0.1)' \
		--lr='(0.001, 0.2)'


# BanditHardAndSparse121
# SUM: best params did NOT learn the best arm
# - See notebooks/exp38_analysis.ipynb for full analysis
exp38:
	-rm -rf $(DATA_PATH)/exp38/*
	tune_bandit.py replicator $(DATA_PATH)/exp38 \
		--exp_name='wsls_bandit' \
		--env_name=BanditHardAndSparse121-v0 \
		--num_iterations=10 \
		--num_episodes=1210 \
		--num_replicators=120 \
		--num_processes=40 \
		--verbose=True \
		--tie_threshold='(1e-8, 0.1)' \
		--lr='(0.001, 0.2)'


# BanditHardAndSparse1000
# SUM: best params did NOT learn the best arm
# - See notebooks/exp39_analysis.ipynb for full analysis
exp39:
	-rm -rf $(DATA_PATH)/exp39/*
	tune_bandit.py replicator $(DATA_PATH)/exp39 \
		--exp_name='wsls_bandit' \
		--env_name=BanditHardAndSparse1000-v0 \
		--num_iterations=10 \
		--num_episodes=10000 \
		--num_replicators=120 \
		--num_processes=40 \
		--verbose=True \
		--tie_threshold='(1e-8, 0.1)' \
		--lr='(0.001, 0.2)'

# --------------------------------------------------------------------------
# 4-17-2019
# 
# Repeat exp38 with a much larger pop, BanditHardAndSparse121
# SUM: best params did NOT learn the best arm
# - See notebooks/exp40_analysis.ipynb for full analysis
exp40:
	-rm -rf $(DATA_PATH)/exp40/*
	tune_bandit.py replicator $(DATA_PATH)/exp40 \
		--exp_name='wsls_bandit' \
		--env_name=BanditHardAndSparse121-v0 \
		--num_iterations=10 \
		--num_episodes=1210 \
		--num_replicators=400 \
		--num_processes=40 \
		--verbose=True \
		--tie_threshold='(1e-8, 0.1)' \
		--lr='(0.001, 0.2)'


# --------------------------------------------------------------------------
# 4-18-2019
# 
# Tune meta to maximize total_E (not total_R as in the previous experiments)

# BanditHardAndSparse2
# - See notebooks/exp41_analysis.ipynb for full analysis
exp41:
	-rm -rf $(DATA_PATH)/exp41/*
	tune_bandit.py replicator $(DATA_PATH)/exp41 \
		--exp_name='wsls_bandit' \
		--env_name=BanditHardAndSparse2-v0 \
		--num_iterations=10 \
		--num_episodes=20 \
		--num_replicators=400 \
		--num_processes=40 \
		--metric=total_E \
		--tie_threshold='(1e-8, 0.1)' \
		--lr='(0.001, 0.2)'

# BanditHardAndSparse10
# - See notebooks/exp42_analysis.ipynb for full analysis
exp42:
	-rm -rf $(DATA_PATH)/exp42/*
	tune_bandit.py replicator $(DATA_PATH)/exp42 \
		--exp_name='wsls_bandit' \
		--env_name=BanditHardAndSparse10-v0 \
		--num_iterations=10 \
		--num_episodes=100 \
		--num_replicators=400 \
		--num_processes=40 \
		--metric=total_E \
		--tie_threshold='(1e-8, 0.1)' \
		--lr='(0.001, 0.2)'

# BanditOneHigh2
# - See notebooks/exp43_analysis.ipynb for full analysis
exp43:
	-rm -rf $(DATA_PATH)/exp43/*
	tune_bandit.py replicator $(DATA_PATH)/exp43 \
		--exp_name='wsls_bandit' \
		--env_name=BanditOneHigh2-v0 \
		--num_iterations=10 \
		--num_episodes=20 \
		--num_replicators=400 \
		--num_processes=40 \
		--metric=total_E \
		--tie_threshold='(1e-8, 0.1)' \
		--lr='(0.001, 0.2)'

# BanditOneHigh10
# - See notebooks/exp44_analysis.ipynb for full analysis
exp44:
	-rm -rf $(DATA_PATH)/exp44/*
	tune_bandit.py replicator $(DATA_PATH)/exp44 \
		--exp_name='wsls_bandit' \
		--env_name=BanditOneHigh10-v0 \
		--num_iterations=10 \
		--num_episodes=100 \
		--num_replicators=400 \
		--num_processes=40 \
		--metric=total_E \
		--tie_threshold='(1e-8, 0.1)' \
		--lr='(0.001, 0.2)'

# BanditOneHigh121
# - See notebooks/exp45_analysis.ipynb for full analysis
exp45:
	-rm -rf $(DATA_PATH)/exp45/*
	tune_bandit.py replicator $(DATA_PATH)/exp45 \
		--exp_name='wsls_bandit' \
		--env_name=BanditOneHigh121-v0 \
		--num_iterations=10 \
		--num_episodes=1210 \
		--num_replicators=400 \
		--num_processes=40 \
		--metric=total_E \
		--tie_threshold='(1e-8, 0.1)' \
		--lr='(0.001, 0.2)'


# --------------------------------------------------------------------------
# 4-18-2019
# Play with num_replicators versus num_iterations (the number of replications) 
# in an easy task.

# SUM (exp46-52): all opts found the best arm, generally within the first 20 
# episodes. 
# - The range of lr was (0.0039834751368258295, .1421784912409777)
# - The range of tie_threshold was (0.005942132325156814, 0.1431450391147704)
# - The range of total_R was (43.0, 88.0)
# - num_iteration < 4 gave the worst result, both in terms of convergence 
#   speed and total_R
# - num_replicators did not seem to matter; this env may be to easy?
# - See notebooks/exp46-52_analysis.ipynb for full analysis

# NEXT: 
# 1. try these w/ metric=total_E just to see if it gives a similiar range of effects.
# 2. try a harder env

# BanditOneHigh10:
# --num_iterations=16; --num_replicators=40
exp46:
	-rm -rf $(DATA_PATH)/exp46/*
	tune_bandit.py replicator $(DATA_PATH)/exp46 \
		--exp_name='wsls_bandit' \
		--env_name=BanditOneHigh10-v0 \
		--num_iterations=16 \
		--num_episodes=100 \
		--num_replicators=40 \
		--num_processes=40 \
		--tie_threshold='(1e-8, 0.1)' \
		--lr='(0.001, 0.2)'

# --num_iterations=2; --num_replicators=400
exp47:
	-rm -rf $(DATA_PATH)/exp47/*
	tune_bandit.py replicator $(DATA_PATH)/exp47 \
		--exp_name='wsls_bandit' \
		--env_name=BanditOneHigh10-v0 \
		--num_iterations=2 \
		--num_episodes=100 \
		--num_replicators=400 \
		--num_processes=40 \
		--tie_threshold='(1e-8, 0.1)' \
		--lr='(0.001, 0.2)'

# --num_iterations=4; --num_replicators=400
exp48:
	-rm -rf $(DATA_PATH)/exp48/*
	tune_bandit.py replicator $(DATA_PATH)/exp48 \
		--exp_name='wsls_bandit' \
		--env_name=BanditOneHigh10-v0 \
		--num_iterations=4 \
		--num_episodes=100 \
		--num_replicators=400 \
		--num_processes=40 \
		--tie_threshold='(1e-8, 0.1)' \
		--lr='(0.001, 0.2)'

# --num_iterations=8; --num_replicators=400
exp49:
	-rm -rf $(DATA_PATH)/exp49/*
	tune_bandit.py replicator $(DATA_PATH)/exp49 \
		--exp_name='wsls_bandit' \
		--env_name=BanditOneHigh10-v0 \
		--num_iterations=8 \
		--num_episodes=100 \
		--num_replicators=400 \
		--num_processes=40 \
		--tie_threshold='(1e-8, 0.1)' \
		--lr='(0.001, 0.2)'

# --num_iterations=16; --num_replicators=400
exp50:
	-rm -rf $(DATA_PATH)/exp50/*
	tune_bandit.py replicator $(DATA_PATH)/exp50 \
		--exp_name='wsls_bandit' \
		--env_name=BanditOneHigh10-v0 \
		--num_iterations=16 \
		--num_episodes=100 \
		--num_replicators=400 \
		--num_processes=40 \
		--tie_threshold='(1e-8, 0.1)' \
		--lr='(0.001, 0.2)'

# --num_iterations=8; --num_replicators=40
exp51:
	-rm -rf $(DATA_PATH)/exp51/*
	tune_bandit.py replicator $(DATA_PATH)/exp51 \
		--exp_name='wsls_bandit' \
		--env_name=BanditOneHigh10-v0 \
		--num_iterations=8 \
		--num_episodes=100 \
		--num_replicators=40 \
		--num_processes=40 \
		--tie_threshold='(1e-8, 0.1)' \
		--lr='(0.001, 0.2)'

# --num_iterations=4; --num_replicators=40
exp52:
	-rm -rf $(DATA_PATH)/exp52/*
	tune_bandit.py replicator $(DATA_PATH)/exp52 \
		--exp_name='wsls_bandit' \
		--env_name=BanditOneHigh10-v0 \
		--num_iterations=4 \
		--num_episodes=100 \
		--num_replicators=40 \
		--num_processes=40 \
		--tie_threshold='(1e-8, 0.1)' \
		--lr='(0.001, 0.2)'


# ----------------------------------------------------------------------------
# 4-18-2019
#
# Repeat of exp46-52 BUT with --metric=total_E.
#
# SUM (exp43-59): all opts found the best arm, _always_ within 
#   the first 20 episodes. 
# - The range of lr was (0.0026347871766753592, 0.3318698314514748)
# - The range of tie_threshold was (0.0019689062371460136, 0.07024629597387025)
# - The range of total_R was (73.0, 74.0); the highest total_R was less
#   here then in exp46-53.
# - num_iteration had no effect.
# - num_replicators may have had a slight effect. Low N makes it unclear.
# - See notebooks/exp53-59_analysis.ipynb for full analysis

# BanditOneHigh10:
# --num_iterations=16; --num_replicators=40
exp53:
	tune_bandit.py replicator $(DATA_PATH)/exp53 \
		--exp_name='wsls_bandit' \
		--env_name=BanditOneHigh10-v0 \
		--num_iterations=16 \
		--num_episodes=100 \
		--num_replicators=40 \
		--num_processes=40 \
		--metric=total_E \
		--tie_threshold='(1e-8, 0.1)' \
		--lr='(0.001, 0.2)'

# --num_iterations=2; --num_replicators=400
exp54:
	tune_bandit.py replicator $(DATA_PATH)/exp54 \
		--exp_name='wsls_bandit' \
		--env_name=BanditOneHigh10-v0 \
		--num_iterations=2 \
		--num_episodes=100 \
		--num_replicators=400 \
		--num_processes=40 \
		--metric=total_E \
		--tie_threshold='(1e-8, 0.1)' \
		--lr='(0.001, 0.2)'

# --num_iterations=4; --num_replicators=400
exp55:
	tune_bandit.py replicator $(DATA_PATH)/exp55 \
		--exp_name='wsls_bandit' \
		--env_name=BanditOneHigh10-v0 \
		--num_iterations=4 \
		--num_episodes=100 \
		--num_replicators=400 \
		--num_processes=40 \
		--metric=total_E \
		--tie_threshold='(1e-8, 0.1)' \
		--lr='(0.001, 0.2)'

# --num_iterations=8; --num_replicators=400
exp56:
	tune_bandit.py replicator $(DATA_PATH)/exp56 \
		--exp_name='wsls_bandit' \
		--env_name=BanditOneHigh10-v0 \
		--num_iterations=8 \
		--num_episodes=100 \
		--num_replicators=400 \
		--num_processes=40 \
		--metric=total_E \
		--tie_threshold='(1e-8, 0.1)' \
		--lr='(0.001, 0.2)'

# --num_iterations=16; --num_replicators=400
exp57:
	tune_bandit.py replicator $(DATA_PATH)/exp57 \
		--exp_name='wsls_bandit' \
		--env_name=BanditOneHigh10-v0 \
		--num_iterations=16 \
		--num_episodes=100 \
		--num_replicators=400 \
		--num_processes=40 \
		--metric=total_E \
		--tie_threshold='(1e-8, 0.1)' \
		--lr='(0.001, 0.2)'

# --num_iterations=8; --num_replicators=40
exp58:
	tune_bandit.py replicator $(DATA_PATH)/exp58 \
		--exp_name='wsls_bandit' \
		--env_name=BanditOneHigh10-v0 \
		--num_iterations=8 \
		--num_episodes=100 \
		--num_replicators=40 \
		--num_processes=40 \
		--metric=total_E \
		--tie_threshold='(1e-8, 0.1)' \
		--lr='(0.001, 0.2)'

# --num_iterations=4; --num_replicators=40
exp59:
	tune_bandit.py replicator $(DATA_PATH)/exp59 \
		--exp_name='wsls_bandit' \
		--env_name=BanditOneHigh10-v0 \
		--num_iterations=4 \
		--num_episodes=100 \
		--num_replicators=40 \
		--num_processes=40 \
		--metric=total_E \
		--tie_threshold='(1e-8, 0.1)' \
		--lr='(0.001, 0.2)'

# ----------------------------------------------------------------------------
# 4-18-2019
#
# Repeat of exp46-52 BUT with MUCH HARDER env, using 
#   --env_name=BanditHardAndSparse121
# and much 100X bandit size for episode number, meaning
#   --num_episodes=12100

# SUM (exp60-66): NO opts searches found the best arm. 
# - See notebooks/exp60-66_analysis.ipynb for full analysis
#
# NEXT: 100X didn't seem to help? For eff. returning to 20X
#       Try more replicators? I've solved this before, intermittently w/ hand 
#       tuning...; Try forcing tie_treshold to small value? 
#       I think that helped before?

# --num_iterations=16; --num_replicators=40
exp60:
	tune_bandit.py replicator $(DATA_PATH)/exp60 \
		--exp_name='wsls_bandit' \
		--env_name=BanditHardAndSparse121-v0 \
		--num_iterations=16 \
		--num_episodes=12100 \
		--num_replicators=40 \
		--num_processes=40 \
		--tie_threshold='(1e-8, 0.1)' \
		--lr='(0.001, 0.2)'

# --num_iterations=2; --num_replicators=400
exp61:
	tune_bandit.py replicator $(DATA_PATH)/exp61 \
		--exp_name='wsls_bandit' \
		--env_name=BanditHardAndSparse121-v0 \
		--num_iterations=2 \
		--num_episodes=12100 \
		--num_replicators=400 \
		--num_processes=40 \
		--tie_threshold='(1e-8, 0.1)' \
		--lr='(0.001, 0.2)'

# --num_iterations=4; --num_replicators=400
exp62:
	tune_bandit.py replicator $(DATA_PATH)/exp62 \
		--exp_name='wsls_bandit' \
		--env_name=BanditHardAndSparse121-v0 \
		--num_iterations=4 \
		--num_episodes=12100 \
		--num_replicators=400 \
		--num_processes=40 \
		--tie_threshold='(1e-8, 0.1)' \
		--lr='(0.001, 0.2)'

# --num_iterations=8; --num_replicators=400
exp63:
	tune_bandit.py replicator $(DATA_PATH)/exp63 \
		--exp_name='wsls_bandit' \
		--env_name=BanditHardAndSparse121-v0 \
		--num_iterations=8 \
		--num_episodes=12100 \
		--num_replicators=400 \
		--num_processes=40 \
		--tie_threshold='(1e-8, 0.1)' \
		--lr='(0.001, 0.2)'

# --num_iterations=16; --num_replicators=400
exp64:
	tune_bandit.py replicator $(DATA_PATH)/exp64 \
		--exp_name='wsls_bandit' \
		--env_name=BanditHardAndSparse121-v0 \
		--num_iterations=16 \
		--num_episodes=12100 \
		--num_replicators=400 \
		--num_processes=40 \
		--tie_threshold='(1e-8, 0.1)' \
		--lr='(0.001, 0.2)'

# --num_iterations=8; --num_replicators=40
exp65:
	tune_bandit.py replicator $(DATA_PATH)/exp65 \
		--exp_name='wsls_bandit' \
		--env_name=BanditHardAndSparse121-v0 \
		--num_iterations=8 \
		--num_episodes=12100 \
		--num_replicators=40 \
		--num_processes=40 \
		--tie_threshold='(1e-8, 0.1)' \
		--lr='(0.001, 0.2)'

# --num_iterations=4; --num_replicators=40
exp66:
	tune_bandit.py replicator $(DATA_PATH)/exp66 \
		--exp_name='wsls_bandit' \
		--env_name=BanditHardAndSparse121-v0 \
		--num_iterations=4 \
		--num_episodes=12100 \
		--num_replicators=40 \
		--num_processes=40 \
		--tie_threshold='(1e-8, 0.1)' \
		--lr='(0.001, 0.2)'

# ----------------------------------------------------------------------------
# 4-19-2019
# 
# Try to solve BanditHardAndSparse121 w/ many more replicators? 
# - Limit tie_thereshold to small values: (1e-10, 1e-6)
# - Fix lr = 0.1 as it's been consisntently near this value in many tuning runs.
#
# SUM: NO opts searches found the best arm. 
exp67:
	tune_bandit.py replicator $(DATA_PATH)/exp67 \
		--exp_name='wsls_bandit' \
		--env_name=BanditHardAndSparse121-v0 \
		--num_iterations=10 \
		--num_episodes=1210 \
		--num_replicators=1200 \
		--num_processes=40 \
		--tie_threshold='(1e-10, 1e-6)' \
		--lr=0.1


# ----------------------------------------------------------------------------
# 4-19-2019
# Made a variation of replicator where the size of the
# pertrubation is tuned by meta-learning. 
# Let's try that here... on a repeat of exp67
# Other changes:
#   --tie_threshold='(1e-10, 1e-3)' 
#   --num_replicators=120

# SUM: NO opts searches found the best arm. Even using the meta-approach,
# which should offer high var in the replicator population, the range of final
# best HP is very low. Basically two values.... I've seen this before but 
# ignored it.... there a bug? If the system is doing poorly variance should stay
# high. If not, why not?
exp68:
	git checkout 694c965821aa2facfacebdf8a0d346ea5ca51b85  
	tune_bandit.py replicator $(DATA_PATH)/exp68 \
		--exp_name='wsls_bandit' \
		--env_name=BanditHardAndSparse121-v0 \
		--num_iterations=50 \
		--num_episodes=1210 \
		--num_replicators=120 \
		--num_processes=40 \
		--tie_threshold='(1e-10, 1e-3)' \
		--lr='(0.001, 0.2)'
	git checkout master


# ----------------------------------------------------------------------------
# 4-19-2019
# meta_learning option in replicator merged to master....
#
# In the process, a major issue w/ replicator was found.

# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
# commit 117a458cb184b9bd2ab92a7f5dd65adbdb5a2fb9
# Author: Erik Peterson <Erik.Exists@gmail.com>
# Date:   Fri Apr 19 14:34:53 2019 -0700

#     MAJOR FIX: there were two bugs in replicator
    
#     1. configs was not getting masked when having children
#     2. childrens params were getting wrongly copied due to a copy-by-ref issue.
    
#     ALL PREV. RUN w/ REPLICATOR ARE QUESTIONALBLE.
    
#     DISREGARD THEM.
# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

# NEED TO RERUN several experiments above.
# First let's try the most recent run again. See how things look now.

# SUM: For both 69 and 70, NO opts searches found the best arm. 
# - However the final params had much higher variance, consistent
#   with a more proper search.

# Without meta tuning of perturbation
exp69:
	tune_bandit.py replicator $(DATA_PATH)/exp69 \
		--exp_name='wsls_bandit' \
		--env_name=BanditHardAndSparse121-v0 \
		--num_iterations=50 \
		--num_episodes=1210 \
		--num_replicators=120 \
		--num_processes=40 \
		--perturbation=0.1 \
		--tie_threshold='(1e-10, 1e-3)' \
		--lr='(0.001, 0.2)'

# w/ tuning
exp70:
	tune_bandit.py replicator $(DATA_PATH)/exp70 \
		--exp_name='wsls_bandit' \
		--env_name=BanditHardAndSparse121-v0 \
		--num_iterations=50 \
		--num_episodes=1210 \
		--num_replicators=120 \
		--num_processes=40 \
		--perturbation=meta \
		--tie_threshold='(1e-10, 1e-3)' \
		--lr='(0.001, 0.2)'

# ----------------------------------------------------------------------------
# 4-22-2019
# 
# Run easy and progressivily harder sims to confirm that the replicator
# is working as intended. Also, let's compare to matched results w/ random
# search. 
# Without meta tuning of perturbation

# ------ #
# 2 ARMs #
# ------ #

# BanditOneHigh2
exp71:
	tune_bandit.py replicator $(DATA_PATH)/exp71 \
		--exp_name='wsls_bandit' \
		--env_name=BanditOneHigh2-v0 \
		--num_iterations=20 \
		--num_episodes=20 \
		--num_replicators=120 \
		--num_processes=40 \
		--perturbation=0.1 \
		--tie_threshold='(1e-10, 1e-3)' \
		--lr='(0.001, 0.2)'

exp72:
	tune_bandit.py random $(DATA_PATH)/exp72 \
		--exp_name='wsls_bandit' \
		--env_name=BanditOneHigh2-v0 \
        --num_episodes=20 \
        --num_samples=2400 \
        --num_processes=40 \
		--tie_threshold='(1e-10, 1e-3)' \
		--lr='(0.001, 0.2)'

# BanditHardAndSparse2
exp73:
	tune_bandit.py replicator $(DATA_PATH)/exp73 \
		--exp_name='wsls_bandit' \
		--env_name=BanditHardAndSparse2-v0 \
		--num_iterations=20 \
		--num_episodes=20 \
		--num_replicators=120 \
		--num_processes=40 \
		--perturbation=0.1 \
		--tie_threshold='(1e-10, 1e-3)' \
		--lr='(0.001, 0.2)'

exp74:
	tune_bandit.py random $(DATA_PATH)/exp74 \
		--exp_name='wsls_bandit' \
		--env_name=BanditHardAndSparse2-v0 \
        --num_episodes=20 \
        --num_samples=2400 \
        --num_processes=40 \
		--tie_threshold='(1e-10, 1e-3)' \
		--lr='(0.001, 0.2)'

# ------- #
# 10 ARMs #
# ------- #

# BanditOneHigh10
exp75:
	tune_bandit.py replicator $(DATA_PATH)/exp75 \
		--exp_name='wsls_bandit' \
		--env_name=BanditOneHigh10-v0 \
		--num_iterations=20 \
		--num_episodes=100 \
		--num_replicators=120 \
		--num_processes=40 \
		--perturbation=0.1 \
		--tie_threshold='(1e-10, 1e-3)' \
		--lr='(0.001, 0.2)'

exp76:
	tune_bandit.py random $(DATA_PATH)/exp76 \
		--exp_name='wsls_bandit' \
		--env_name=BanditOneHigh10-v0 \
        --num_episodes=100 \
        --num_samples=2400 \
        --num_processes=40 \
		--tie_threshold='(1e-10, 1e-3)' \
		--lr='(0.001, 0.2)'

# BanditHardAndSparse10
exp77:
	tune_bandit.py replicator $(DATA_PATH)/exp77 \
		--exp_name='wsls_bandit' \
		--env_name=BanditHardAndSparse10-v0 \
		--num_iterations=20 \
		--num_episodes=100 \
		--num_replicators=120 \
		--num_processes=40 \
		--perturbation=0.1 \
		--tie_threshold='(1e-10, 1e-3)' \
		--lr='(0.001, 0.2)'

exp78:
	tune_bandit.py random $(DATA_PATH)/exp78 \
		--exp_name='wsls_bandit' \
		--env_name=BanditHardAndSparse10-v0 \
        --num_episodes=100 \
        --num_samples=2400 \
        --num_processes=40 \
		--tie_threshold='(1e-10, 1e-3)' \
		--lr='(0.001, 0.2)'


# -------- #
# 121 ARMs #
# -------- #

# BanditOneHigh121
exp79:
	tune_bandit.py replicator $(DATA_PATH)/exp79 \
		--exp_name='wsls_bandit' \
		--env_name=BanditOneHigh121-v0 \
		--num_iterations=20 \
		--num_episodes=1210 \
		--num_replicators=120 \
		--num_processes=40 \
		--perturbation=0.1 \
		--tie_threshold='(1e-10, 1e-3)' \
		--lr='(0.001, 0.2)'

exp80:
	tune_bandit.py random $(DATA_PATH)/exp80 \
		--exp_name='wsls_bandit' \
		--env_name=BanditOneHigh121-v0 \
        --num_episodes=1210 \
        --num_samples=2400 \
        --num_processes=40 \
		--tie_threshold='(1e-10, 1e-3)' \
		--lr='(0.001, 0.2)'

# BanditHardAndSparse121
# Run already. See exp69-70.


# --------- #
# 1000 ARMs #
# --------- #

# TODO: this is where you left off prior to resuming editing the main draft. 
# - Next direct step it to analyze the exp below. You are trying to tune HP and validate 
#   replicator. So far replicator looks good. 
# - Need to tune for beta and epsilon, and set up a comparison.
# - Need to explore max for total_E in meta as well. (Been using total_R as the metric)
# - **Overall aim** is to compare performance, and see
#   if meta is (much, a lot, a little, less) than epsilon or beta.
# - Use p_best as the performance metric. Note it reaches some criterion. How often it 
#   (re)crosses this criterion, as measure of stability.
# - May need to convert beta to be stochastic or ....
# - Or add an H/noisy sampling method, 'Soft Actor-Critic:  Off-Policy Maximum Entropy Deep Reinforcement Learning with a Stochastic...'; I want to make clear how wastefully noisy sampling is. What's the best test?


# BanditOneHigh121
exp81:
	tune_bandit.py replicator $(DATA_PATH)/exp81 \
		--exp_name='wsls_bandit' \
		--env_name=BanditOneHigh1000-v0 \
		--num_iterations=20 \
		--num_episodes=2500 \
		--num_replicators=120 \
		--num_processes=40 \
		--perturbation=0.1 \
		--tie_threshold='(1e-10, 1e-3)' \
		--lr='(0.001, 0.2)'

exp82:
	tune_bandit.py random $(DATA_PATH)/exp82 \
		--exp_name='wsls_bandit' \
		--env_name=BanditOneHigh1000-v0 \
        --num_episodes=2500 \
        --num_samples=2400 \
        --num_processes=40 \
		--tie_threshold='(1e-10, 1e-3)' \
		--lr='(0.001, 0.2)'

# ---------------------------------------------------------------------------
# 5-15-2019
# 27ce394690c3683b262a8d01afaa6523aaed697c

# ---------------------------
# MAJOR CHANGE TO META_BANDIT
# ---------------------------

# Prev to this change, both R and E value were learned by TD(0). 
# This was done so their learning rules were consistent. 
# But, E doesn't need a delta learning rule at all!
# The Bellman eq is sufficient (and optimal)
# So I adapted the code for E learning so it is just Bellman, with a learning
# rate (lr_E). 
# In practice, this change required the lr for for R and E are seperated. 
#
# As a result, the API for `wsls_bandit` is now changed. 
#
# !!! THIS BREAKS THE API OF ALL EXPS PREVIOUS TO THIS ONE !!!
#
# For usage examples see `notebooks/exptest_wsls_bandit.ipynb`.

# Run some new exps with the Bellman form

# BanditOneHigh121
# SUM: top half of best params solve i. Exploration was truncated. That's
# OK here. Might not be O in harder tasks
exp83:
	tune_bandit.py replicator $(DATA_PATH)/exp83 \
		--exp_name='wsls_bandit' \
		--env_name=BanditOneHigh121-v0 \
		--num_iterations=20 \
		--num_episodes=605 \
		--num_replicators=120 \
		--num_processes=40 \
		--perturbation=0.1 \
		--tie_threshold='(1e-10, 0.01)' \
		--lr_R='(0.0000001, 0.2)' \
		--lr_E='(0.0000001, 0.2)'


# HardAndSparse121
# SUM: for 84 and 85 the best soln discovered was don't explore at all. Stick
# to arm 0.
# Should I try running with total value, or max E as the objective?

# NOTE: I can solve this by hand-tuning. How to make the meta-opt work right?
exp84:
	tune_bandit.py replicator $(DATA_PATH)/exp84 \
		--exp_name='wsls_bandit' \
		--env_name=BanditHardAndSparse121-v0 \
		--num_iterations=20 \
		--num_episodes=12100 \
		--num_replicators=120 \
		--num_processes=40 \
		--perturbation=0.1 \
		--tie_threshold='(1e-10, 0.01)' \
		--lr_R='(0.0000001, 0.2)' \
		--lr_E='(0.0000001, 0.2)'

# HardAndSparse121 
# 10X more samples compared to 84
# SUM: see above
exp85:
	tune_bandit.py replicator $(DATA_PATH)/exp85 \
		--exp_name='wsls_bandit' \
		--env_name=BanditHardAndSparse121-v0 \
		--num_iterations=20 \
		--num_episodes=121000 \
		--num_replicators=120 \
		--num_processes=40 \
		--perturbation=0.1 \
		--tie_threshold='(1e-10, 0.01)' \
		--lr_R='(0.0000001, 0.2)' \
		--lr_E='(0.0000001, 0.2)'


# HardAndSparse121 
# metric: total_E
# SUM: no improvement from 85. Huh.
exp86:
	tune_bandit.py replicator $(DATA_PATH)/exp86 \
		--exp_name='wsls_bandit' \
		--env_name=BanditHardAndSparse121-v0 \
		--metric=total_E \
		--num_iterations=20 \
		--num_episodes=121000 \
		--num_replicators=120 \
		--num_processes=40 \
		--perturbation=0.1 \
		--tie_threshold='(1e-10, 0.01)' \
		--lr_R='(0.0000001, 0.2)' \
		--lr_E='(0.0000001, 0.2)'

# HardAndSparse121 
# metric: total_E_R
# SUM: still no exploration at all. Something is wrong, with replicator. 
# With meta. I don't understand this result. Focusing on E should lead to 
# extended exploration! 
exp87:
	tune_bandit.py replicator $(DATA_PATH)/exp87 \
		--exp_name='wsls_bandit' \
		--env_name=BanditHardAndSparse121-v0 \
		--metric=total_E_R \
		--num_iterations=40 \
		--num_episodes=121000 \
		--num_replicators=1200 \
		--num_processes=40 \
		--perturbation=0.1 \
		--tie_threshold='(1e-10, 0.01)' \
		--lr_R='(0.0000001, 0.2)' \
		--lr_E='(0.0000001, 0.2)'


# ---------------------------------------------------------------------------
# 5-20-2019
# 7b6353092468ef9645f655a470b25ecbc42c4fc5

# Fixed bug in E/R init that was preventing exploration in low R conditions.
# Re-running last few attempts are HardAndSparse121....

# --metric=total_R 
exp88:
	tune_bandit.py replicator $(DATA_PATH)/exp88 \
		--exp_name='wsls_bandit' \
		--env_name=BanditHardAndSparse121-v0 \
		--metric=total_R \
		--num_iterations=40 \
		--num_episodes=60500 \
		--num_replicators=120 \
		--num_processes=40 \
		--perturbation=0.1 \
		--tie_threshold='(1e-10, 0.01)' \
		--lr_R='(0.0000001, 0.2)' \
		--lr_E='(0.0000001, 0.2)'

# --metric=total_E 
exp89:
	tune_bandit.py replicator $(DATA_PATH)/exp89 \
		--exp_name='wsls_bandit' \
		--env_name=BanditHardAndSparse121-v0 \
		--metric=total_E \
		--num_iterations=40 \
		--num_episodes=60500 \
		--num_replicators=120 \
		--num_processes=40 \
		--perturbation=0.1 \
		--tie_threshold='(1e-10, 0.01)' \
		--lr_R='(0.0000001, 0.2)' \
		--lr_E='(0.0000001, 0.2)'

# --metric=total_E_R 
exp90:
	tune_bandit.py replicator $(DATA_PATH)/exp90 \
		--exp_name='wsls_bandit' \
		--env_name=BanditHardAndSparse121-v0 \
		--metric=total_E_R \
		--num_iterations=40 \
		--num_episodes=60500 \
		--num_replicators=120 \
		--num_processes=40 \
		--perturbation=0.1 \
		--tie_threshold='(1e-10, 0.01)' \
		--lr_R='(0.0000001, 0.2)' \
		--lr_E='(0.0000001, 0.2)'


# --metric=total_R --perturbation=meta
exp91:
	tune_bandit.py replicator $(DATA_PATH)/exp91 \
		--exp_name='wsls_bandit' \
		--env_name=BanditHardAndSparse121-v0 \
		--metric=total_R \
		--num_iterations=40 \
		--num_episodes=60500 \
		--num_replicators=120 \
		--num_processes=40 \
		--perturbation=meta \
		--tie_threshold='(1e-10, 0.01)' \
		--lr_R='(0.0000001, 0.2)' \
		--lr_E='(0.0000001, 0.2)'

# random search
exp92:
	tune_bandit.py random $(DATA_PATH)/exp92 \
		--exp_name='wsls_bandit' \
		--env_name=BanditHardAndSparse121-v0 \
        --num_episodes=60500 \
        --num_samples=4800 \
        --num_processes=40 \
		--tie_threshold='(1e-10, 0.01)' \
		--lr_R='(0.0000001, 0.2)' \
		--lr_E='(0.0000001, 0.2)'


# ----------------------------------------------------------------------------
# As of commit
# 
# fb03b7758741c50c8c8249a4a8dc81c3d636e8c6
#
# E learning is strictly greedy, which means the lr_E param is redundant for
# single trial bandit tasks AND HAS BEEN REMOVED. When working with mult step
# envs, re-introducing a gamma time horizon may be needed. Not for 1 step 
# bandits.
# 
# THIS CHANGE BREAKS ALL PAST EXPS ABOVE.
# ----------------------------------------------------------------------------
# 
# SUM: in the exptest_wsls_bandit notebook I confirmed all oneHigh bandits
# are still easily solved.

# Now let's re-try some Sparse problems

# w/ random search

# SUM: 93 and 94 found good solutions. 
#      95 result made no obvious progress on the problem.

exp93:
	tune_bandit.py random $(DATA_PATH)/exp93 \
		--exp_name='wsls_bandit' \
		--env_name=BanditHardAndSparse2-v0 \
        --num_episodes=2000 \
        --num_samples=1000 \
        --num_processes=40 \
		--tie_threshold='(1e-16, 0.01)' \
		--lr_R='(0.0000001, 0.2)' \

exp94:
	tune_bandit.py random $(DATA_PATH)/exp94 \
		--exp_name='wsls_bandit' \
		--env_name=BanditHardAndSparse10-v0 \
        --num_episodes=50000 \
        --num_samples=1000 \
        --num_processes=40 \
		--tie_threshold='(1e-16, 0.01)' \
		--lr_R='(0.0000001, 0.2)' \

exp95:
	tune_bandit.py random $(DATA_PATH)/exp95 \
		--exp_name='wsls_bandit' \
		--env_name=BanditHardAndSparse121-v0 \
        --num_episodes=605000 \
        --num_samples=1000 \
        --num_processes=40 \
		--tie_threshold='(1e-16, 0.01)' \
		--lr_R='(0.000000001, 0.2)' \


# ---------------------------------------------------------------------------
# 6-14-2019
# c8b4c07ef3374f1580ef250686a770e051db7012

# First full round of tuning for draft:
# https://www.biorxiv.org/content/10.1101/671362v1

# PLAN:
	# # Bandits for Fig 2.
	# - Traditional - BanditOneHigh10 
	#   + 10 arm: 20, 80, ....
	# - 2 mode - BanditTwoHigh10
	#   + 10 arm: 20, 80, ..., 80, ....
	# - Rand (1 winner) - BanditUniform121
	#   + 121 (or 10?) arm: rand: 20-75, .... 80, ....
	# - Sparse - BanditHardAndSparse10
	#   + 10 arm: 0.01, 0.02, ....

	# # Agents
	# - ep (no decay)
	# - beta (det)
	# - beta (softmax)
	# - meta 
	# - random (baseline)

	# # Exp plan
	# - Tune each independently for each bandit. Random search ~1000 samples.
	# - Do 25 runs/seeds for each tune bandit and env combination. Maybe do more later, depending on initial variability.


# SUM:
# - For softbeta, temp was not tuned and the default is far to high. Needs tuning for all bandits. *Not discussing softbeta further below.*
#
# BanditOneHigh10:
# - Found stable correct params for all models.
#
# BanditTwoHigh10
# - meta: saw both arms, settled on one quickly, held (biased) value for both
# - ep: saw both, settled on one, then swtiched.
# - beta: only saw on arm
#
# BanditUniform121
# - a harder task than expected. No agent found a stable solution.
# - meta/ep got close?
# - beta was terrible 
#
# BanditHardAndSparse10
# - meta no progress (....hand tuning can solve this)
# - ep: no progress
# - beta: good progress (!), though soln was not stable 
#   did not expect beta to do well here! In retro, makes sense?


# ---------------
# BanditOneHigh10
exp96:
	tune_bandit.py random $(DATA_PATH)/exp96 \
		--exp_name='wsls_bandit' \
		--env_name=BanditOneHigh10-v0 \
        --num_episodes=100 \
        --num_samples=1000 \
        --num_processes=40 \
		--tie_threshold='(1e-16, 0.01)' \
		--lr_R='(0.000000001, 0.5)' 

exp97:
	tune_bandit.py random $(DATA_PATH)/exp97 \
		--exp_name='epsilon_bandit' \
		--env_name=BanditOneHigh10-v0 \
        --num_episodes=100 \
        --num_samples=1000 \
		--num_processes=40 \
		--epsilon='(0.01, 0.99)' \
		--lr_R='(0.000000001, 0.2)' 

exp98:
	tune_bandit.py random $(DATA_PATH)/exp98 \
		--exp_name='beta_bandit' \
		--env_name=BanditOneHigh10-v0 \
		--num_episodes=100 \
        --num_samples=1000 \
		--num_processes=40 \
		--beta='(1e-3, 2)' \
		--lr_R='(0.000000001, 0.2)' 

exp99:
	tune_bandit.py random $(DATA_PATH)/exp99 \
		--exp_name='softbeta_bandit' \
		--env_name=BanditOneHigh10-v0 \
		--num_episodes=100 \
        --num_samples=1000 \
		--num_processes=40 \
		--beta='(1e-3, 2)' \
		--lr_R='(0.000000001, 0.2)' 


# ---------------
# BanditTwoHigh10
exp100:
	tune_bandit.py random $(DATA_PATH)/exp100 \
		--exp_name='wsls_bandit' \
		--env_name=BanditTwoHigh10-v0 \
        --num_episodes=100 \
        --num_samples=1000 \
        --num_processes=40 \
		--tie_threshold='(1e-16, 0.01)' \
		--lr_R='(0.000000001, 0.5)' 

exp101:
	tune_bandit.py random $(DATA_PATH)/exp101 \
		--exp_name='epsilon_bandit' \
		--env_name=BanditTwoHigh10-v0 \
        --num_episodes=100 \
        --num_samples=1000 \
		--num_processes=40 \
		--epsilon='(0.01, 0.99)' \
		--lr_R='(0.000000001, 0.2)' 

exp102:
	tune_bandit.py random $(DATA_PATH)/exp102 \
		--exp_name='beta_bandit' \
		--env_name=BanditTwoHigh10-v0 \
		--num_episodes=100 \
        --num_samples=1000 \
		--num_processes=40 \
		--beta='(1e-3, 2)' \
		--lr_R='(0.000000001, 0.2)' 

exp103:
	tune_bandit.py random $(DATA_PATH)/exp103 \
		--exp_name='softbeta_bandit' \
		--env_name=BanditTwoHigh10-v0 \
		--num_episodes=100 \
        --num_samples=1000 \
		--num_processes=40 \
		--beta='(1e-3, 2)' \
		--lr_R='(0.000000001, 0.2)' 


# ---------------
# BanditUniform121
exp104:
	tune_bandit.py random $(DATA_PATH)/exp104 \
		--exp_name='wsls_bandit' \
		--env_name=BanditUniform121-v0 \
        --num_episodes=60500 \
        --num_samples=1000 \
        --num_processes=40 \
		--tie_threshold='(1e-16, 0.01)' \
		--lr_R='(0.000000001, 0.5)' 

exp105:
	tune_bandit.py random $(DATA_PATH)/exp105 \
		--exp_name='epsilon_bandit' \
		--env_name=BanditUniform121-v0 \
        --num_episodes=60500 \
        --num_samples=1000 \
		--num_processes=40 \
		--epsilon='(0.01, 0.99)' \
		--lr_R='(0.000000001, 0.2)' 

exp106:
	tune_bandit.py random $(DATA_PATH)/exp106 \
		--exp_name='beta_bandit' \
		--env_name=BanditUniform121-v0 \
		--num_episodes=60500 \
        --num_samples=1000 \
		--num_processes=40 \
		--beta='(1e-3, 2)' \
		--lr_R='(0.000000001, 0.2)' 

exp107:
	tune_bandit.py random $(DATA_PATH)/exp107 \
		--exp_name='softbeta_bandit' \
		--env_name=BanditUniform121-v0 \
		--num_episodes=60500 \
        --num_samples=1000 \
		--num_processes=40 \
		--beta='(1e-3, 2)' \
		--lr_R='(0.000000001, 0.2)' 

# ---------------------
# BanditHardAndSparse10
exp108:
	tune_bandit.py random $(DATA_PATH)/exp108 \
		--exp_name='wsls_bandit' \
		--env_name=BanditHardAndSparse10-v0 \
        --num_episodes=50000 \
        --num_samples=1000 \
        --num_processes=40 \
		--tie_threshold='(1e-16, 0.01)' \
		--lr_R='(0.0000001, 0.2)' 

exp109:
	tune_bandit.py random $(DATA_PATH)/exp109 \
		--exp_name='epsilon_bandit' \
		--env_name=BanditHardAndSparse10-v0 \
        --num_episodes=50000 \
        --num_samples=1000 \
		--num_processes=40 \
		--epsilon='(0.01, 0.99)' \
		--lr_R='(0.000000001, 0.2)' 

exp110:
	tune_bandit.py random $(DATA_PATH)/exp110 \
		--exp_name='beta_bandit' \
		--env_name=BanditHardAndSparse10-v0 \
        --num_episodes=50000 \
        --num_samples=1000 \
		--num_processes=40 \
		--beta='(1e-3, 2)' \
		--lr_R='(0.000000001, 0.2)' 

exp111:
	tune_bandit.py random $(DATA_PATH)/exp111 \
		--exp_name='softbeta_bandit' \
		--env_name=BanditHardAndSparse10-v0 \
        --num_episodes=50000 \
        --num_samples=1000 \
		--num_processes=40 \
		--beta='(1e-3, 2)' \
		--lr_R='(0.000000001, 0.2)'
	 

# --------------------------------
# 6-20-2019
# 0a0cafef1d8268a73cf61e43232d468ee6c849e1

# First pass at tuning softbeta's temp param
# Trying BanditOneHigh10

# SUM: learned a stable solution; temp ~ 0.8
exp112:
	tune_bandit.py random $(DATA_PATH)/exp112 \
		--exp_name='softbeta_bandit' \
		--env_name=BanditOneHigh10-v0 \
		--num_episodes=100 \
        --num_samples=1000 \
		--num_processes=40 \
		--beta='(1e-3, 2)' \
		--lr_R='(0.000000001, 0.2)' \
		--temp='(1e-1, 10)'

# --------------------------------
# 6-20-2019
# aca35be1d0b707e56569ed6869a0a5574edf35fe

# Re-run remaining tasks for softbeta, w/ temp tune

# SUM: exp113 learned a stable soln. Saw both arms.
#      exp114 slight progress based on total_R rank. Nothing like a soln.
#      exp115 no progress. p+best = 0 and total_R dist is symmetric
exp113:
	tune_bandit.py random $(DATA_PATH)/exp113 \
		--exp_name='softbeta_bandit' \
		--env_name=BanditTwoHigh10-v0 \
		--num_episodes=100 \
        --num_samples=1000 \
		--num_processes=40 \
		--beta='(1e-3, 2)' \
		--lr_R='(0.000000001, 0.2)' \
		--temp='(1e-1, 10)'

exp114:
	tune_bandit.py random $(DATA_PATH)/exp114 \
		--exp_name='softbeta_bandit' \
		--env_name=BanditUniform121-v0 \
		--num_episodes=60500 \
        --num_samples=1000 \
		--num_processes=40 \
		--beta='(1e-3, 2)' \
		--lr_R='(0.000000001, 0.2)' \
		--temp='(1e-1, 10)'

exp115:
	tune_bandit.py random $(DATA_PATH)/exp115 \
		--exp_name='softbeta_bandit' \
		--env_name=BanditHardAndSparse10-v0 \
        --num_episodes=50000 \
        --num_samples=1000 \
		--num_processes=40 \
		--beta='(1e-3, 2)' \
		--lr_R='(0.000000001, 0.2)' \
		--temp='(1e-1, 10)'

# --------------------------------
# 6-21-2019
# aca35be1d0b707e56569ed6869a0a5574edf35fe

# More tuning for BanditHardAndSparse10

# META:
# Begin search near some good hand tuned params.
# 'lr_R' : 0.0001
# 'tie_threshold' : 0.000000001

# SUM: found a stable soln (a range of them actually) 
exp116:
	tune_bandit.py random $(DATA_PATH)/exp116 \
		--exp_name='wsls_bandit' \
		--env_name=BanditHardAndSparse10-v0 \
        --num_episodes=50000 \
        --num_samples=1000 \
        --num_processes=40 \
		--tie_threshold='(0.0000000005, 0.000000005)' \
		--lr_R='(0.00005, 0.0005)' 

# EP: 
# Try some partitions on ep?
# Move in units of 0.1 but only do 80 samples.
# Looking for hints of progress.

# SUM: *only* exp121 learned the best value, though it's high level of 
#      exploration limited its overall performance to p_best ~= 0.5.
#      Presumably, adding a annealing for epsilon would allow it to converge
#      to p_best -> 1.0.
exp117:
	tune_bandit.py random $(DATA_PATH)/exp117 \
		--exp_name='epsilon_bandit' \
		--env_name=BanditHardAndSparse10-v0 \
        --num_episodes=50000 \
        --num_samples=1000 \
		--num_processes=40 \
		--epsilon='(0.01, 0.1)' \
		--lr_R='(0.00005, 0.0005)'  

exp118:
	tune_bandit.py random $(DATA_PATH)/exp118 \
		--exp_name='epsilon_bandit' \
		--env_name=BanditHardAndSparse10-v0 \
        --num_episodes=50000 \
        --num_samples=80 \
		--num_processes=40 \
		--epsilon='(0.101, 0.2)' \
		--lr_R='(0.00005, 0.0005)'  

exp119:
	tune_bandit.py random $(DATA_PATH)/exp119 \
		--exp_name='epsilon_bandit' \
		--env_name=BanditHardAndSparse10-v0 \
        --num_episodes=50000 \
        --num_samples=80 \
		--num_processes=40 \
		--epsilon='(0.201, 0.3)' \
		--lr_R='(0.00005, 0.0005)'  
	
exp120:
	tune_bandit.py random $(DATA_PATH)/exp120 \
		--exp_name='epsilon_bandit' \
		--env_name=BanditHardAndSparse10-v0 \
        --num_episodes=50000 \
        --num_samples=80 \
		--num_processes=40 \
		--epsilon='(0.301, 0.4)' \
		--lr_R='(0.00005, 0.0005)'  

exp121:
	tune_bandit.py random $(DATA_PATH)/exp121 \
		--exp_name='epsilon_bandit' \
		--env_name=BanditHardAndSparse10-v0 \
        --num_episodes=50000 \
        --num_samples=80 \
		--num_processes=40 \
		--epsilon='(0.401, 0.5)' \
		--lr_R='(0.00005, 0.0005)'  

# -
# SFOTBETA w/ TEMP

# Use good results from exp98 to get it started
# 'beta': 0.3671269035680538,
# 'lr_R': 0.009549127434538021,
# 'total_R': 84.0

# SUM: Bit of a surprise w/ best params for this exp.
#      'beta': 0.38381390290530865
#      'lr_R': 0.009705438535971703
#      'temp': 5.919105536555171
#
# With the temp so high the agent does learn what the best
# arm, but can't express it in a stable way. Anneal beta?
exp122:
	tune_bandit.py random $(DATA_PATH)/exp122 \
		--exp_name='softbeta_bandit' \
		--env_name=BanditOneHigh10-v0 \
		--num_episodes=100 \
        --num_samples=1000 \
		--num_processes=40 \
		--beta='(0.33, 0.39)' \
		--lr_R='(0.008, 0.010)' \
		--temp='(1e-1, 10)'

# --------------------------------------
# 6-24-2019
# Tweak exp110 - beta+hardandsparse10

# Best from exp110: 'beta': 1.53, 'lr_R': 0.049
# Let's search around them (2 fold)

# SUM: still not stable, but this soln (w/ larger)
#      beta has a higher p_best
# 'beta': 2.832753031081456, 'lr_R': 0.05336390016195454
exp123:
	tune_bandit.py random $(DATA_PATH)/exp123 \
		--exp_name='beta_bandit' \
		--env_name=BanditHardAndSparse10-v0 \
        --num_episodes=50000 \
        --num_samples=120 \
		--num_processes=40 \
		--beta='(0.75, 3)' \
		--lr_R='(0.025, 0.1)' 

# ------------------------------------
# 6-24-2019
# 
# Tweak BanditUniform 120 arms: 0.2-0.6, 1 best arm: 0.8
# 8c40c57eec610ec048e2ea5b2f10441ff83f3915
#
# SUM:
# - exp124: found stable soln
# - exp125: found stable soln (low ep)
# - exp126: found stable soln (very eff	)
# - exp127: no soln found. p_best low (temp too)
exp124:
	tune_bandit.py random $(DATA_PATH)/exp124 \
		--exp_name='wsls_bandit' \
		--env_name=BanditUniform121-v0 \
        --num_episodes=60500 \
        --num_samples=1000 \
        --num_processes=40 \
		--tie_threshold='(1e-16, 0.01)' \
		--lr_R='(0.000000001, 0.5)' 

exp125:
	tune_bandit.py random $(DATA_PATH)/exp125 \
		--exp_name='epsilon_bandit' \
		--env_name=BanditUniform121-v0 \
        --num_episodes=60500 \
        --num_samples=1000 \
		--num_processes=40 \
		--epsilon='(0.01, 0.99)' \
		--lr_R='(0.000000001, 0.2)' 

exp126:
	tune_bandit.py random $(DATA_PATH)/exp126 \
		--exp_name='beta_bandit' \
		--env_name=BanditUniform121-v0 \
		--num_episodes=60500 \
        --num_samples=1000 \
		--num_processes=40 \
		--beta='(1e-3, 2)' \
		--lr_R='(0.000000001, 0.2)' 

exp127:
	tune_bandit.py random $(DATA_PATH)/exp127 \
		--exp_name='softbeta_bandit' \
		--env_name=BanditUniform121-v0 \
		--num_episodes=60500 \
        --num_samples=1000 \
		--num_processes=40 \
		--beta='(1e-3, 2)' \
		--lr_R='(0.000000001, 0.2)' \
		--temp='(1e-1, 10)'

# -------------------------------------
# 6-26-2019
# Now that we have optimized HP, need to do some final runs sampling
# 1. env seed
# 2. param sensitivity (for fixed seed) +/- 10% or 50%.

# -
# 1. Sample seeds
# -

# BanditOneHigh10:
# meta: exp96 - learns a stable soln 
#   + 'tie_threshold': 0.0041, 'lr_R': 0.31
exp128:
	parallel -j 40 \
			--joblog '$(DATA_PATH)/exp128.log' \
			--nice 19 --delay 2 --colsep ',' \
			'wsls_bandit.py --env_name=BanditOneHigh10-v0 --num_episodes=500 --tie_break='next' --tie_threshold=0.0041 --lr_R=0.31 --save=$(DATA_PATH)/exp128_{1}.pkl --interactive=False --debug=False --master_seed={1}' ::: {1..100}

# beta: exp98 - learns a stable soln 
#   + 'beta': 0.37, 'lr_R': 0.0095
exp129:
	parallel -j 40 \
			--joblog '$(DATA_PATH)/exp129.log' \
			--nice 19 --delay 2 --colsep ',' \
			'beta_bandit.py --env_name=BanditOneHigh10-v0 --num_episodes=500  --lr_R=0.31 --beta=0.37 --save=$(DATA_PATH)/exp129_{1}.pkl --interactive=False --debug=False --master_seed={1}' ::: {1..100}

# softbeta: exp112 - learns a stable soln 
#   + 'beta': 0.045, 'lr_R': 0.12, 'temp': 0.10
exp130:
	parallel -j 40 \
			--joblog '$(DATA_PATH)/exp130.log' \
			--nice 19 --delay 2 --colsep ',' \
			'softbeta_bandit.py --env_name=BanditOneHigh10-v0 --num_episodes=500  --lr_R=0.12 --beta=0.045 --temp=0.01 --save=$(DATA_PATH)/exp130_{1}.pkl --interactive=False --debug=False --master_seed={1}' ::: {1..100}

# epsilon: exp97 - learns a stable soln 
#   + 'epsilon': 0.078, 'lr_R': 0.12
exp131:
	parallel -j 40 \
			--joblog '$(DATA_PATH)/exp131.log' \
			--nice 19 --delay 2 --colsep ',' \
			'epsilon_bandit.py --env_name=BanditOneHigh10-v0 --num_episodes=500  --lr_R=0.12 --epsilon=0.078 --save=$(DATA_PATH)/exp131_{1}.pkl --interactive=False --debug=False --master_seed={1}' ::: {1..100}

# -
# BanditTwoHigh10
# meta: exp100 - sees both, learns a stable soln
#   + 'tie_threshold': 0.0058, 'lr_R': 0.14
exp132:
	parallel -j 40 \
			--joblog '$(DATA_PATH)/exp132.log' \
			--nice 19 --delay 2 --colsep ',' \
			'wsls_bandit.py --env_name=BanditTwoHigh10-v0 --num_episodes=500 --tie_break='next' --tie_threshold=0.0058 --lr_R=0.14 --save=$(DATA_PATH)/exp132_{1}.pkl --interactive=False --debug=False --master_seed={1}' ::: {1..100}

# beta: exp102 - learns only one arm. Never sees best arm 2
#   + 'beta': 0.025, 'lr_R': 0.073
exp133:
	parallel -j 40 \
			--joblog '$(DATA_PATH)/exp133.log' \
			--nice 19 --delay 2 --colsep ',' \
			'beta_bandit.py --env_name=BanditTwoHigh10-v0 --num_episodes=500  --lr_R=0.073 --beta=0.025 --save=$(DATA_PATH)/exp133_{1}.pkl --interactive=False --debug=False --master_seed={1}' ::: {1..100}

# softbeta: exp113 - sees both (probably?), learns a stable soln
#   + 'beta': 0.010, 'lr_R': 0.17, 'temp': 0.24
exp134:
	parallel -j 40 \
			--joblog '$(DATA_PATH)/exp134.log' \
			--nice 19 --delay 2 --colsep ',' \
			'softbeta_bandit.py --env_name=BanditTwoHigh10-v0 --num_episodes=500  --lr_R=0.17 --beta=0.010 --temp=0.24 --save=$(DATA_PATH)/exp134_{1}.pkl --interactive=False --debug=False --master_seed={1}' ::: {1..100}

# epsilon: exp101 - learns solns, flip flops between them
#   + 'epsilon': 0.078, 'lr_R': 0.12
exp135:
	parallel -j 40 \
			--joblog '$(DATA_PATH)/exp135.log' \
			--nice 19 --delay 2 --colsep ',' \
			'epsilon_bandit.py --env_name=BanditTwoHigh10-v0 --num_episodes=500  --lr_R=0.12 --epsilon=0.078 --save=$(DATA_PATH)/exp135_{1}.pkl --interactive=False --debug=False --master_seed={1}' ::: {1..100}

# -
# BanditUniform121
# meta: exp124 - found stable soln
#   + 'tie_threshold': 0.00031, 'lr_R': 0.14
exp136:
	parallel -j 40 \
			--joblog '$(DATA_PATH)/exp136.log' \
			--nice 19 --delay 2 --colsep ',' \
			'wsls_bandit.py --env_name=BanditUniform121-v0 --num_episodes=120000 --tie_break='next' --tie_threshold=0.00031 --lr_R=0.14 --save=$(DATA_PATH)/exp136_{1}.pkl --interactive=False --debug=False --master_seed={1}' ::: {1..100}

# beta: exp126 - found stable soln (very eff.)
#   + 'beta': 0.090, 'lr_R': 0.061
exp137:
	parallel -j 40 \
			--joblog '$(DATA_PATH)/exp137.log' \
			--nice 19 --delay 2 --colsep ',' \
			'beta_bandit.py --env_name=BanditUniform121-v0 --num_episodes=120000  --lr_R=0.061 --beta=0.090 --save=$(DATA_PATH)/exp137_{1}.pkl --interactive=False --debug=False --master_seed={1}' ::: {1..100}

# softbeta: exp127 - no soln found. p_best low (temp too)
#   + 'beta': 0.60, 'lr_R': 0.097, 'temp': 0.13
exp138:
	parallel -j 40 \
			--joblog '$(DATA_PATH)/exp138.log' \
			--nice 19 --delay 2 --colsep ',' \
			'softbeta_bandit.py --env_name=BanditUniform121-v0 --num_episodes=120000  --lr_R=0.097 --beta=0.60 --temp=0.13 --save=$(DATA_PATH)/exp138_{1}.pkl --interactive=False --debug=False --master_seed={1}' ::: {1..100}

# epsilon: exp125: found stable soln (low ep)
#   + 'epsilon': 0.012, 'lr_R': 0.11
exp139:
	parallel -j 40 \
			--joblog '$(DATA_PATH)/exp139.log' \
			--nice 19 --delay 2 --colsep ',' \
			'epsilon_bandit.py --env_name=BanditUniform121-v0 --num_episodes=120000  --lr_R=0.11 --epsilon=0.012 --save=$(DATA_PATH)/exp139_{1}.pkl --interactive=False --debug=False --master_seed={1}' ::: {1..100}

# -
# HardAndSparse10
# meta: meta: exp116 - learns a stable soln 
#   + 'tie_threshold': 3.76-09, 'lr_R': 0.00021
exp140:
	parallel -j 40 \
			--joblog '$(DATA_PATH)/exp140.log' \
			--nice 19 --delay 2 --colsep ',' \
			'wsls_bandit.py --env_name=BanditHardAndSparse10-v0 --num_episodes=100000 --tie_break='next' --tie_threshold=3.76e-09 --lr_R=0.00021 --save=$(DATA_PATH)/exp140_{1}.pkl --interactive=False --debug=False --master_seed={1}' ::: {1..100}

# beta: exp110 - Close to soln. Not stable. Narrow range?
#   + 'beta': 2.83, 'lr_R': 0.053
exp141:
	parallel -j 40 \
			--joblog '$(DATA_PATH)/exp141.log' \
			--nice 19 --delay 2 --colsep ',' \
			'beta_bandit.py --env_name= -v0 --num_episodes=100000  --lr_R=0.053 --beta=2.83 --save=$(DATA_PATH)/exp141_{1}.pkl --interactive=False --debug=False --master_seed={1}' ::: {1..100}

# softbeta: exp122 - learns the value but needs to high a temp to ever stabilize
#   + 'beta': 0.38, 'lr_R': 0.00971, 'temp': 5.9
exp142:
	parallel -j 40 \
			--joblog '$(DATA_PATH)/exp142.log' \
			--nice 19 --delay 2 --colsep ',' \
			'softbeta_bandit.py --env_name=BanditHardAndSparse10-v0 --num_episodes=100000  --lr_R=0.00971 --beta=0.38 --temp=5.9 --save=$(DATA_PATH)/exp142_{1}.pkl --interactive=False --debug=False --master_seed={1}' ::: {1..100}

# epsilon: exp121 - learns the value, final performance limited by high epsilon
#   + 'epsilon': 0.42, 'lr_R': 0.00043
exp143:
	parallel -j 40 \
			--joblog '$(DATA_PATH)/exp143.log' \
			--nice 19 --delay 2 --colsep ',' \
			'epsilon_bandit.py --env_name=BanditHardAndSparse10-v0 --num_episodes=100000  --lr_R=0.00043 --epsilon=0.42 --save=$(DATA_PATH)/exp143_{1}.pkl --interactive=False --debug=False --master_seed={1}' ::: {1..100}

# -------------------------------------
# 7-1-2019
# 77e4fcc4e4f074ebb22c574aa6685bd0acb80b0d
#
# Meta has been doing real well (exp128-144). Better than I'd expected. 
# To compare to SOA for AI applications let's run some
# epsilon annealing exps. First, we'll need to tune....
#
# For the four envs, tune an annealed epsilon_bandit.  
#

# SUM: - very good performance on 144 and 146.
#      - didn't see second good arm in 145
#      - no progress on sparse (147/8)

# BanditOneHigh10
exp144:
	tune_bandit.py random $(DATA_PATH)/exp144 \
		--exp_name='epsilon_bandit' \
		--env_name=BanditOneHigh10-v0 \
        --num_episodes=100 \
        --num_samples=1000 \
		--num_processes=40 \
		--epsilon='(0.01, 0.99)' \
		--epsilon_decay_tau='(0.000000001, 0.1)' \
		--lr_R='(0.000000001, 0.2)' 

# BanditTwoHigh10
exp145:
	tune_bandit.py random $(DATA_PATH)/exp145 \
		--exp_name='epsilon_bandit' \
		--env_name=BanditTwoHigh10-v0 \
        --num_episodes=100 \
        --num_samples=1000 \
		--num_processes=40 \
		--epsilon='(0.01, 0.99)' \
		--epsilon_decay_tau='(0.000000001, 0.1)' \
		--lr_R='(0.000000001, 0.2)' 

# BanditUniform121
exp146:
	tune_bandit.py random $(DATA_PATH)/exp146 \
		--exp_name='epsilon_bandit' \
		--env_name=BanditUniform121-v0 \
        --num_episodes=60500 \
        --num_samples=1000 \
		--num_processes=40 \
		--epsilon='(0.01, 0.99)' \
		--epsilon_decay_tau='(0.000000001, 0.1)' \
		--lr_R='(0.000000001, 0.2)' 

# BanditHardAndSparse10
# Full ep search
exp147:
	tune_bandit.py random $(DATA_PATH)/exp147 \
		--exp_name='epsilon_bandit' \
		--env_name=BanditHardAndSparse10-v0 \
        --num_episodes=50000 \
        --num_samples=80 \
		--num_processes=40 \
		--epsilon='(0.01, 0.99)' \
		--epsilon_decay_tau='(0.000000001, 0.1)' \
		--lr_R='(0.00005, 0.0005)' 

# Hand-tuned ep range (w/ no annealing).
exp148:
	tune_bandit.py random $(DATA_PATH)/exp148 \
		--exp_name='epsilon_bandit' \
		--env_name=BanditHardAndSparse10-v0 \
        --num_episodes=50000 \
        --num_samples=80 \
		--num_processes=40 \
		--epsilon='(0.401, 0.5)' \
		--epsilon_decay_tau='(0.000000001, 0.1)' \
		--lr_R='(0.00005, 0.0005)' 


# -----------------------------
# Replicate exps for ep decay
# BanditOneHigh10
# {'epsilon': 0.16028547541549285, 'epsilon_decay_tau': 0.07969927623155562, 'lr_R': 0.10688060186632899, 'total_R': 86.0}
exp149:
	parallel -j 40 \
			--joblog '$(DATA_PATH)/exp149.log' \
			--nice 19 --delay 2 --colsep ',' \
			'epsilon_bandit.py --env_name=BanditOneHigh10-v0 --num_episodes=500  --lr_R=0.11 --epsilon=0.16 --epsilon_decay_tau=0.080 --save=$(DATA_PATH)/exp149_{1}.pkl --interactive=False --debug=False --master_seed={1}' ::: {1..100}

# BanditTwoHigh10
# {'epsilon': 0.838651023382445, 'epsilon_decay_tau': 0.07116057540412388, 'lr_R': 0.1885459873244454, 'total_R': 71.0}
exp150:
	parallel -j 40 \
			--joblog '$(DATA_PATH)/exp150.log' \
			--nice 19 --delay 2 --colsep ',' \
			'epsilon_bandit.py --env_name=BanditTwoHigh10-v0 --num_episodes=500  --lr_R=0.19 --epsilon=0.83 --epsilon_decay_tau=0.071 --save=$(DATA_PATH)/exp150_{1}.pkl --interactive=False --debug=False --master_seed={1}' ::: {1..100}

# BanditUniform121
# {'epsilon': 0.5743595655655118, 'epsilon_decay_tau': 0.03268667798766935, 'lr_R': 0.17235910245007333, 'total_R': 48586.0}
exp151:
	parallel -j 40 \
			--joblog '$(DATA_PATH)/exp151.log' \
			--nice 19 --delay 2 --colsep ',' \
			'epsilon_bandit.py --env_name=BanditUniform121-v0 --num_episodes=120000  --lr_R=0.17 --epsilon=0.57 --epsilon_decay_tau=0.032 --save=$(DATA_PATH)/exp151_{1}.pkl --interactive=False --debug=False --master_seed={1}' ::: {1..100}

# BanditHardAndSparse10
# {'epsilon': 0.7666645365811449, 'epsilon_decay_tau': 0.014058030361594634, 'lr_R': 7.504905974098415e-05, 'total_R': 1029.0}
exp152:
	parallel -j 40 \
			--joblog '$(DATA_PATH)/exp152.log' \
			--nice 19 --delay 2 --colsep ',' \
			'epsilon_bandit.py --env_name=BanditHardAndSparse10-v0 --num_episodes=100000 --lr_R=7.50e-05 --epsilon=0.76 --epsilon_decay_tau=0.014 --save=$(DATA_PATH)/exp152_{1}.pkl --interactive=False --debug=False --master_seed={1}' ::: {1..100}


# ------------------------------------------
# Run exps w/ a random 'learner'. Neg control.
# Fix lr_R at 0.1. No way to opt this.

# BanditOneHigh10
exp153:
	parallel -j 40 \
			--joblog '$(DATA_PATH)/exp153.log' \
			--nice 19 --delay 2 --colsep ',' \
			'random_bandit.py --env_name=BanditOneHigh10-v0 --num_episodes=500  --lr_R=0.1 --save=$(DATA_PATH)/exp153_{1}.pkl --interactive=False --debug=False --master_seed={1}' ::: {1..100}

# BanditTwoHigh10
exp154:
	parallel -j 40 \
			--joblog '$(DATA_PATH)/exp154.log' \
			--nice 19 --delay 2 --colsep ',' \
			'random_bandit.py --env_name=BanditTwoHigh10-v0 --num_episodes=500  --lr_R=0.1 --save=$(DATA_PATH)/exp154_{1}.pkl --interactive=False --debug=False --master_seed={1}' ::: {1..100}

# BanditUniform121
exp155:
	parallel -j 40 \
			--joblog '$(DATA_PATH)/exp155.log' \
			--nice 19 --delay 2 --colsep ',' \
			'random_bandit.py --env_name=BanditUniform121-v0 --num_episodes=120000  --lr_R=0.1 --save=$(DATA_PATH)/exp155_{1}.pkl --interactive=False --debug=False --master_seed={1}' ::: {1..100}

# BanditHardAndSparse10
exp156:
	parallel -j 40 \
			--joblog '$(DATA_PATH)/exp156.log' \
			--nice 19 --delay 2 --colsep ',' \
			'random_bandit.py --env_name=BanditHardAndSparse10-v0 --num_episodes=100000 --lr_R=0.1 --save=$(DATA_PATH)/exp156_{1}.pkl --interactive=False --debug=False --master_seed={1}' ::: {1..100}

# ------------------------------------------
# Prior to
# 0228db4dd6a9a57c763d88587b866ac9d22421d1
# the random seed as fixed for all Actors(). 
# This is not idea. So. Rerun the relevant cases below.

# -
# BanditOneHigh10
#
# replicates exp130:
exp157:
	parallel -j 40 \
			--joblog '$(DATA_PATH)/exp157.log' \
			--nice 19 --delay 2 --colsep ',' \
			'softbeta_bandit.py --env_name=BanditOneHigh10-v0 --num_episodes=500  --lr_R=0.12 --beta=0.045 --temp=0.01 --save=$(DATA_PATH)/exp157_{1}.pkl --interactive=False --debug=False --master_seed={1}' ::: {1..100}

# replicates exp131
exp158:
	parallel -j 40 \
			--joblog '$(DATA_PATH)/exp158.log' \
			--nice 19 --delay 2 --colsep ',' \
			'epsilon_bandit.py --env_name=BanditOneHigh10-v0 --num_episodes=500  --lr_R=0.12 --epsilon=0.078 --save=$(DATA_PATH)/exp158_{1}.pkl --interactive=False --debug=False --master_seed={1}' ::: {1..100}

# -
# BanditTwoHigh10

# replicates exp134
exp159:
	parallel -j 40 \
			--joblog '$(DATA_PATH)/exp159.log' \
			--nice 19 --delay 2 --colsep ',' \
			'softbeta_bandit.py --env_name=BanditTwoHigh10-v0 --num_episodes=500  --lr_R=0.17 --beta=0.010 --temp=0.24 --save=$(DATA_PATH)/exp159_{1}.pkl --interactive=False --debug=False --master_seed={1}' ::: {1..100}


# replicates exp135
exp160:
	parallel -j 40 \
			--joblog '$(DATA_PATH)/exp160.log' \
			--nice 19 --delay 2 --colsep ',' \
			'epsilon_bandit.py --env_name=BanditTwoHigh10-v0 --num_episodes=500  --lr_R=0.12 --epsilon=0.078 --save=$(DATA_PATH)/exp160_{1}.pkl --interactive=False --debug=False --master_seed={1}' ::: {1..100}

# -
# BanditUniform121

# replicates exp138
exp161:
	parallel -j 40 \
			--joblog '$(DATA_PATH)/exp161.log' \
			--nice 19 --delay 2 --colsep ',' \
			'softbeta_bandit.py --env_name=BanditUniform121-v0 --num_episodes=120000  --lr_R=0.097 --beta=0.60 --temp=0.13 --save=$(DATA_PATH)/exp161_{1}.pkl --interactive=False --debug=False --master_seed={1}' ::: {1..100}


# replicates exp139
exp162:
	parallel -j 40 \
			--joblog '$(DATA_PATH)/exp162.log' \
			--nice 19 --delay 2 --colsep ',' \
			'epsilon_bandit.py --env_name=BanditUniform121-v0 --num_episodes=120000  --lr_R=0.11 --epsilon=0.012 --save=$(DATA_PATH)/exp162_{1}.pkl --interactive=False --debug=False --master_seed={1}' ::: {1..100}

# -
# HardAndSparse

# replicates exp142
exp163:
	parallel -j 40 \
			--joblog '$(DATA_PATH)/exp163.log' \
			--nice 19 --delay 2 --colsep ',' \
			'softbeta_bandit.py --env_name=BanditHardAndSparse10-v0 --num_episodes=100000  --lr_R=0.00971 --beta=0.38 --temp=5.9 --save=$(DATA_PATH)/exp163_{1}.pkl --interactive=False --debug=False --master_seed={1}' ::: {1..100}

# replicates exp143
exp164:
	parallel -j 40 \
			--joblog '$(DATA_PATH)/exp164.log' \
			--nice 19 --delay 2 --colsep ',' \
			'epsilon_bandit.py --env_name=BanditHardAndSparse10-v0 --num_episodes=100000  --lr_R=0.00043 --epsilon=0.42 --save=$(DATA_PATH)/exp164_{1}.pkl --interactive=False --debug=False --master_seed={1}' ::: {1..100}

# -
# ep-decay experiments (all bandits)

# BanditOneHigh10
# replicates exp149
exp165:
	parallel -j 40 \
			--joblog '$(DATA_PATH)/exp165.log' \
			--nice 19 --delay 2 --colsep ',' \
			'epsilon_bandit.py --env_name=BanditOneHigh10-v0 --num_episodes=500  --lr_R=0.11 --epsilon=0.16 --epsilon_decay_tau=0.080 --save=$(DATA_PATH)/exp165_{1}.pkl --interactive=False --debug=False --master_seed={1}' ::: {1..100}


# BanditTwoHigh10
# replicates exp150
exp166:
	parallel -j 40 \
			--joblog '$(DATA_PATH)/exp166.log' \
			--nice 19 --delay 2 --colsep ',' \
			'epsilon_bandit.py --env_name=BanditTwoHigh10-v0 --num_episodes=500  --lr_R=0.19 --epsilon=0.83 --epsilon_decay_tau=0.071 --save=$(DATA_PATH)/exp166_{1}.pkl --interactive=False --debug=False --master_seed={1}' ::: {1..100}

# BanditUniform121
# replicates exp151
exp167:
	parallel -j 40 \
			--joblog '$(DATA_PATH)/exp167.log' \
			--nice 19 --delay 2 --colsep ',' \
			'epsilon_bandit.py --env_name=BanditUniform121-v0 --num_episodes=120000  --lr_R=0.17 --epsilon=0.57 --epsilon_decay_tau=0.032 --save=$(DATA_PATH)/exp167_{1}.pkl --interactive=False --debug=False --master_seed={1}' ::: {1..100}

# BanditHardAndSparse10
# replicates exp152
exp168:
	parallel -j 40 \
			--joblog '$(DATA_PATH)/exp168.log' \
			--nice 19 --delay 2 --colsep ',' \
			'epsilon_bandit.py --env_name=BanditHardAndSparse10-v0 --num_episodes=100000 --lr_R=7.50e-05 --epsilon=0.76 --epsilon_decay_tau=0.014 --save=$(DATA_PATH)/exp168_{1}.pkl --interactive=False --debug=False --master_seed={1}' ::: {1..100}

# -
# Random

# BanditOneHigh10
# replicates exp153
exp169:
	parallel -j 40 \
			--joblog '$(DATA_PATH)/exp169.log' \
			--nice 19 --delay 2 --colsep ',' \
			'random_bandit.py --env_name=BanditOneHigh10-v0 --num_episodes=500  --lr_R=0.1 --save=$(DATA_PATH)/exp169_{1}.pkl --interactive=False --debug=False --master_seed={1}' ::: {1..100}

# BanditTwoHigh10
# replicates exp154
exp170:
	parallel -j 40 \
			--joblog '$(DATA_PATH)/exp170.log' \
			--nice 19 --delay 2 --colsep ',' \
			'random_bandit.py --env_name=BanditTwoHigh10-v0 --num_episodes=500  --lr_R=0.1 --save=$(DATA_PATH)/exp170_{1}.pkl --interactive=False --debug=False --master_seed={1}' ::: {1..100}

# BanditUniform121
# replicates exp155
exp171:
	parallel -j 40 \
			--joblog '$(DATA_PATH)/exp171.log' \
			--nice 19 --delay 2 --colsep ',' \
			'random_bandit.py --env_name=BanditUniform121-v0 --num_episodes=120000  --lr_R=0.1 --save=$(DATA_PATH)/exp171_{1}.pkl --interactive=False --debug=False --master_seed={1}' ::: {1..100}

# BanditHardAndSparse10
# replicates exp156
exp172:
	parallel -j 40 \
			--joblog '$(DATA_PATH)/exp172.log' \
			--nice 19 --delay 2 --colsep ',' \
			'random_bandit.py --env_name=BanditHardAndSparse10-v0 --num_episodes=100000 --lr_R=0.1 --save=$(DATA_PATH)/exp17_{1}.pkl --interactive=False --debug=False --master_seed={1}' ::: {1..100}



###########################################################################
###########################################################################
###########################################################################
###########################################################################
###########################################################################
# 7-18-2019
#
# Param opt using replicator dynammics.
#
# Use same starting params as random search.
	# exp_name='beta_bandit',
	# env_name='BanditOneHigh10-v0',
	# num_iterations=2,
	# num_episodes=2000,
	# num_replicators=10,
	# num_processes=1,
	# perturbation=0.1,
	# metric="total_R",
	# verbose=False,
	# master_seed=None,
	# **config_kwargs)

# -
# eta

# BanditOneHigh10
exp173:
	tune_bandit.py replicator $(DATA_PATH)/exp173 \
		--exp_name='wsls_bandit' \
		--env_name=BanditOneHigh10-v0 \
        --num_iterations=10 \
        --num_episodes=100 \
		--num_replicators=400 \
		--num_processes=40 \
		--perturbation=0.05 \
		--metric="total_R" \
		--tie_threshold='(1e-5, .1)' \
		--lr_R='(0.000001, 0.4)'

# BanditTwoHigh10
exp174:
	tune_bandit.py replicator $(DATA_PATH)/exp174 \
		--exp_name='wsls_bandit' \
		--env_name=BanditTwoHigh10-v0 \
        --num_iterations=10 \
        --num_episodes=200 \
		--num_replicators=400 \
		--num_processes=40 \
		--perturbation=0.05 \
		--metric="total_R" \
		--tie_threshold='(1e-5, .1)' \
		--lr_R='(0.000001, 0.4)'

# BanditUniform121
exp175:
	tune_bandit.py replicator $(DATA_PATH)/exp175 \
		--exp_name='wsls_bandit' \
		--env_name=BanditUniform121-v0 \
		--num_iterations=10 \
        --num_episodes=60500 \
		--num_replicators=400 \
		--num_processes=40 \
		--perturbation=0.05 \
		--metric="total_R" \
		--tie_threshold='(1e-6, .1)' \
		--lr_R='(0.000001, 0.4)'

# BanditHardAndSparse10
exp176:
	tune_bandit.py replicator $(DATA_PATH)/exp176 \
		--exp_name='wsls_bandit' \
		--env_name=BanditHardAndSparse10-v0 \
		--num_iterations=10 \
        --num_episodes=50000 \
		--num_replicators=400 \
		--num_processes=40 \
		--perturbation=0.05 \
		--metric="total_R" \
		--tie_threshold='(1e-10, 1e-08)' \
		--lr_R='(0.000001, 0.4)' 


# -
# beta
# BanditOneHigh10
exp177:
	tune_bandit.py replicator $(DATA_PATH)/exp177 \
		--exp_name='beta_bandit' \
		--env_name=BanditOneHigh10-v0 \
        --num_iterations=10 \
        --num_episodes=100 \
		--num_replicators=400 \
		--num_processes=40 \
		--perturbation=0.05 \
		--metric="total_R" \
		--beta='(1e-2, 3)' \
		--lr_R='(1e-5, 0.2)' 

# BanditTwoHigh10
exp178:
	tune_bandit.py replicator $(DATA_PATH)/exp178 \
		--exp_name='beta_bandit' \
		--env_name=BanditTwoHigh10-v0 \
        --num_iterations=10 \
        --num_episodes=200 \
		--num_replicators=400 \
		--num_processes=40 \
		--perturbation=0.05 \
		--metric="total_R" \
		--beta='(1e-2, 3)' \
		--lr_R='(1e-5, 0.2)' 

# BanditUniform121
exp179:
	tune_bandit.py replicator $(DATA_PATH)/exp179 \
		--exp_name='beta_bandit' \
		--env_name=BanditUniform121-v0 \
		--num_iterations=10 \
        --num_episodes=60500 \
		--num_replicators=400 \
		--num_processes=40 \
		--perturbation=0.05 \
		--metric="total_R" \
		--beta='(1e-2, 3)' \
		--lr_R='(1e-5, 0.2)' 

# BanditHardAndSparse10
exp180:
	tune_bandit.py replicator $(DATA_PATH)/exp180 \
		--exp_name='beta_bandit' \
		--env_name=BanditHardAndSparse10-v0 \
		--num_iterations=10 \
        --num_episodes=50000 \
		--num_replicators=400 \
		--num_processes=40 \
		--perturbation=0.05 \
		--metric="total_R" \
		--beta='(1e-2, 3)' \
		--lr_R='(1e-5, 0.2)' 

# -
# softbeta
# BanditOneHigh10
exp181:
	tune_bandit.py replicator $(DATA_PATH)/exp181 \
		--exp_name='softbeta_bandit' \
		--env_name=BanditOneHigh10-v0 \
        --num_iterations=10 \
        --num_episodes=100 \
		--num_replicators=400 \
		--num_processes=40 \
		--perturbation=0.05 \
		--metric="total_R" \
		--beta='(1e-3, 3)' \
		--lr_R='(1e-5, 0.2)' \
		--temp='(1e-1, 3)'

# BanditTwoHigh10
exp182:
	tune_bandit.py replicator $(DATA_PATH)/exp182 \
		--exp_name='softbeta_bandit' \
		--env_name=BanditTwoHigh10-v0 \
        --num_iterations=10 \
        --num_episodes=200 \
		--num_replicators=400 \
		--num_processes=40 \
		--perturbation=0.05 \
		--metric="total_R" \
		--beta='(1e-3, 3)' \
		--lr_R='(1e-5, 0.2)' \
		--temp='(1e-1, 3)'

# BanditUniform121
exp183:
	tune_bandit.py replicator $(DATA_PATH)/exp183 \
		--exp_name='softbeta_bandit' \
		--env_name=BanditUniform121-v0 \
		--num_iterations=10 \
        --num_episodes=60500 \
		--num_replicators=400 \
		--num_processes=40 \
		--perturbation=0.05 \
		--metric="total_R" \
		--beta='(1e-3, 3)' \
		--lr_R='(1e-5, 0.2)' \
		--temp='(1e-1, 3)'

# BanditHardAndSparse10
exp184:
	tune_bandit.py replicator $(DATA_PATH)/exp184 \
		--exp_name='softbeta_bandit' \
		--env_name=BanditHardAndSparse10-v0 \
		--num_iterations=10 \
        --num_episodes=50000 \
		--num_replicators=400 \
		--num_processes=40 \
		--perturbation=0.05 \
		--metric="total_R" \
		--beta='(1e-3, 3)' \
		--lr_R='(1e-5, 0.2)' \
		--temp='(1e-1, 6)'
	
# -
# ep:
# BanditOneHigh10
exp185:
	tune_bandit.py replicator $(DATA_PATH)/exp185 \
		--exp_name='epsilon_bandit' \
		--env_name=BanditOneHigh10-v0 \
		--num_iterations=10 \
        --num_episodes=100 \
		--num_replicators=400 \
		--num_processes=40 \
		--perturbation=0.05 \
		--metric="total_R" \
		--epsilon='(0.01, 0.99)' \
		--lr_R='(0.000000001, 0.2)' 

# BanditTwoHigh10
exp186:
	tune_bandit.py replicator $(DATA_PATH)/exp186 \
		--exp_name='epsilon_bandit' \
		--env_name=BanditTwoHigh10-v0 \
        --num_iterations=10 \
        --num_episodes=200 \
		--num_replicators=400 \
		--num_processes=40 \
		--perturbation=0.05 \
		--metric="total_R" \
		--epsilon='(0.01, 0.99)' \
		--lr_R='(0.000000001, 0.2)' 

# BanditUniform121
exp187:
	tune_bandit.py replicator $(DATA_PATH)/exp187 \
		--exp_name='epsilon_bandit' \
		--env_name=BanditUniform121-v0 \
		--num_iterations=10 \
        --num_episodes=60500 \
		--num_replicators=400 \
		--num_processes=40 \
		--perturbation=0.05 \
		--metric="total_R" \
		--epsilon='(0.01, 0.99)' \
		--lr_R='(0.000000001, 0.2)' 

# BanditHardAndSparse10 - full ep
exp188:
	tune_bandit.py replicator $(DATA_PATH)/exp188 \
		--exp_name='epsilon_bandit' \
		--env_name=BanditHardAndSparse10-v0 \
		--num_iterations=10 \
        --num_episodes=50000 \
		--num_replicators=400 \
		--num_processes=40 \
		--perturbation=0.05 \
		--metric="total_R" \
		--epsilon='(0.01, 0.99)' \
		--lr_R='(0.00005, 0.0005)' 

# BanditHardAndSparse10 - select ep
exp189:
	tune_bandit.py replicator $(DATA_PATH)/exp189 \
		--exp_name='epsilon_bandit' \
		--env_name=BanditHardAndSparse10-v0 \
		--num_iterations=10 \
        --num_episodes=50000 \
		--num_replicators=400 \
		--num_processes=40 \
		--perturbation=0.05 \
		--metric="total_R" \
		--epsilon='(0.401, 0.5)' \
		--lr_R='(0.00005, 0.0005)' 

# -
# Annealed ep:
# BanditOneHigh10
exp190:
	tune_bandit.py replicator $(DATA_PATH)/exp190 \
		--exp_name='epsilon_bandit' \
		--env_name=BanditOneHigh10-v0 \
		--num_iterations=10 \
        --num_episodes=100 \
		--num_replicators=400 \
		--num_processes=40 \
		--perturbation=0.05 \
		--metric="total_R" \
		--epsilon='(0.01, 0.99)' \
		--epsilon_decay_tau='(0.000000001, 0.1)' \
		--lr_R='(0.000000001, 0.2)' 

# BanditTwoHigh10
exp191:
	tune_bandit.py replicator $(DATA_PATH)/exp191 \
		--exp_name='epsilon_bandit' \
		--env_name=BanditTwoHigh10-v0 \
        --num_iterations=10 \
        --num_episodes=200 \
		--num_replicators=400 \
		--num_processes=40 \
		--perturbation=0.05 \
		--metric="total_R" \
		--epsilon='(0.01, 0.99)' \
		--epsilon_decay_tau='(0.000000001, 0.1)' \
		--lr_R='(0.000000001, 0.2)' 

# BanditUniform121
exp192:
	tune_bandit.py replicator $(DATA_PATH)/exp192 \
		--exp_name='epsilon_bandit' \
		--env_name=BanditUniform121-v0 \
		--num_iterations=10 \
        --num_episodes=60500 \
		--num_replicators=400 \
		--num_processes=40 \
		--perturbation=0.05 \
		--metric="total_R" \
		--epsilon='(0.01, 0.99)' \
		--epsilon_decay_tau='(0.000000001, 0.1)' \
		--lr_R='(0.000000001, 0.2)' 

# BanditHardAndSparse10
exp193:
	tune_bandit.py replicator $(DATA_PATH)/exp193 \
		--exp_name='epsilon_bandit' \
		--env_name=zs-v0 \
		--num_iterations=10 \
        --num_episodes=50000 \
		--num_replicators=400 \
		--num_processes=40 \
		--perturbation=0.05 \
		--metric="total_R" \
		--epsilon='(0.401, 0.5)' \
		--epsilon_decay_tau='(0.000000001, 0.1)' \
		--lr_R='(0.00005, 0.0005)' 

############################################################################
############################################################################
############################################################################
############################################################################
############################################################################
# 7-22-2019
# Run mult seeds for replicator opt params

# Meta
# BanditOneHigh10
exp194:
	parallel -j 40 \
			--joblog '$(DATA_PATH)/exp194.log' \
			--nice 19 --delay 2 --colsep ',' \
			'wsls_bandit.py --env_name=BanditOneHigh10-v0 --num_episodes=500 --tie_break='next' --tie_threshold=0.053 --lr_R=0.34 --save=$(DATA_PATH)/exp194_{1}.pkl --interactive=False --debug=False --master_seed={1}' ::: {1..100}

# BanditTwoHigh10
exp195:
	parallel -j 40 \
			--joblog '$(DATA_PATH)/exp195.log' \
			--nice 19 --delay 2 --colsep ',' \
			'wsls_bandit.py --env_name=BanditTwoHigh10-v0 --num_episodes=500 --tie_break='next' --tie_threshold=0.0169 --lr_R=0.161 --save=$(DATA_PATH)/exp195_{1}.pkl --interactive=False --debug=False --master_seed={1}' ::: {1..100}

# BanditUniform121
exp196:
	parallel -j 40 \
			--joblog '$(DATA_PATH)/exp196.log' \
			--nice 19 --delay 2 --colsep ',' \
			'wsls_bandit.py --env_name=BanditUniform121-v0 --num_episodes=60500 --tie_break='next' --tie_threshold=0.00355 --lr_R=0.147 --save=$(DATA_PATH)/exp196_{1}.pkl --interactive=False --debug=False --master_seed={1}' ::: {1..100}

# HardAndSparse10
exp197:
	parallel -j 40 \
			--joblog '$(DATA_PATH)/exp197.log' \
			--nice 19 --delay 2 --colsep ',' \
			'wsls_bandit.py --env_name=BanditHardAndSparse10-v0 --num_episodes=50000 --tie_break='next' --tie_threshold=5.782e-09 --lr_R=0.00112 --save=$(DATA_PATH)/exp197_{1}.pkl --interactive=False --debug=False --master_seed={1}' ::: {1..100}

# -
# beta:
# BanditOneHigh10
exp198:
	parallel -j 40 \
			--joblog '$(DATA_PATH)/exp198.log' \
			--nice 19 --delay 2 --colsep ',' \
			'beta_bandit.py --env_name=BanditOneHigh10-v0 --num_episodes=500 --tie_break='next' --beta=0.22 --lr_R=0.18 --save=$(DATA_PATH)/exp198_{1}.pkl --interactive=False --debug=False --master_seed={1}' ::: {1..100}

# BanditTwoHigh10
exp199:
	parallel -j 40 \
			--joblog '$(DATA_PATH)/exp199.log' \
			--nice 19 --delay 2 --colsep ',' \
			'beta_bandit.py --env_name=BanditTwoHigh10-v0 --num_episodes=500 --tie_break='next' --beta=0.188 --lr_R=0.129 --save=$(DATA_PATH)/exp199_{1}.pkl --interactive=False --debug=False --master_seed={1}' ::: {1..100}

# BanditUniform121
exp200:
	parallel -j 40 \
			--joblog '$(DATA_PATH)/exp200.log' \
			--nice 19 --delay 2 --colsep ',' \
			'beta_bandit.py --env_name=BanditUniform121-v0 --num_episodes=60500 --tie_break='next' --beta=0.056 --lr_R=0.141 --save=$(DATA_PATH)/exp200_{1}.pkl --interactive=False --debug=False --master_seed={1}' ::: {1..100}

# HardAndSparse10
exp201:
	parallel -j 40 \
			--joblog '$(DATA_PATH)/exp201.log' \
			--nice 19 --delay 2 --colsep ',' \
			'beta_bandit.py --env_name=BanditHardAndSparse10-v0 --num_episodes=50000 --tie_break='next' --beta=0.217 --lr_R=0.051 --save=$(DATA_PATH)/exp201_{1}.pkl --interactive=False --debug=False --master_seed={1}' ::: {1..100}

# -
# softbeta
# BanditOneHigh10
exp202:
	parallel -j 40 \
			--joblog '$(DATA_PATH)/exp202.log' \
			--nice 19 --delay 2 --colsep ',' \
			'softbeta_bandit.py --env_name=BanditOneHigh10-v0 --num_episodes=500 --beta=0.066 --lr_R=0.13 --temp=0.13 --save=$(DATA_PATH)/exp202_{1}.pkl --interactive=False --debug=False --master_seed={1}' ::: {1..100}

# BanditTwoHigh10
exp203:
	parallel -j 40 \
			--joblog '$(DATA_PATH)/exp203.log' \
			--nice 19 --delay 2 --colsep ',' \
			'softbeta_bandit.py --env_name=BanditTwoHigh10-v0 --num_episodes=500 --beta=0.133 --lr_R=0.030 --temp=0.098 --save=$(DATA_PATH)/exp203_{1}.pkl --interactive=False --debug=False --master_seed={1}' ::: {1..100}

# BanditUniform121
exp204:
	parallel -j 40 \
			--joblog '$(DATA_PATH)/exp204.log' \
			--nice 19 --delay 2 --colsep ',' \
			'softbeta_bandit.py --env_name=BanditUniform121-v0 --num_episodes=60500 --beta=0.125 --lr_R=0.174 --temp=0.0811 --save=$(DATA_PATH)/exp204_{1}.pkl --interactive=False --debug=False --master_seed={1}' ::: {1..100}

# HardAndSparse10
exp205:
	parallel -j 40 \
			--joblog '$(DATA_PATH)/exp205.log' \
			--nice 19 --delay 2 --colsep ',' \
			'softbeta_bandit.py --env_name=BanditHardAndSparse10-v0 --num_episodes=50000 --beta=2.140 --lr_R=0.128 --temp=5.045 --save=$(DATA_PATH)/exp205_{1}.pkl --interactive=False --debug=False --master_seed={1}' ::: {1..100}

# -
# ep

# BanditOneHigh10
exp206:
	parallel -j 40 \
			--joblog '$(DATA_PATH)/exp206.log' \
			--nice 19 --delay 2 --colsep ',' \
			'epsilon_bandit.py --env_name=BanditOneHigh10-v0 --num_episodes=500 --epsilon=0.14 --lr_R=0.087 --save=$(DATA_PATH)/exp206_{1}.pkl --interactive=False --debug=False --master_seed={1}' ::: {1..100}

# BanditTwoHigh10
exp207:
	parallel -j 40 \
			--joblog '$(DATA_PATH)/exp207.log' \
			--nice 19 --delay 2 --colsep ',' \
		'epsilon_bandit.py --env_name=BanditTwoHigh10-v0 --num_episodes=500 --epsilon=0.087 --lr_R=0.08583 --save=$(DATA_PATH)/exp207_{1}.pkl --interactive=False --debug=False --master_seed={1}' ::: {1..100}

# BanditUniform121
exp208:
	parallel -j 40 \
			--joblog '$(DATA_PATH)/exp208.log' \
			--nice 19 --delay 2 --colsep ',' \
			'epsilon_bandit.py --env_name=BanditUniform121-v0 --num_episodes=60500 --epsilon=0.0117 --lr_R=0.137 --save=$(DATA_PATH)/exp208_{1}.pkl --interactive=False --debug=False --master_seed={1}' ::: {1..100}

# HardAndSparse10
exp209:
	parallel -j 40 \
			--joblog '$(DATA_PATH)/exp209.log' \
			--nice 19 --delay 2 --colsep ',' \
			'epsilon_bandit.py --env_name=BanditHardAndSparse10-v0 --num_episodes=50000 --epsilon=0.4057 --lr_R=0.000484 --save=$(DATA_PATH)/exp209_{1}.pkl --interactive=False --debug=False --master_seed={1}' ::: {1..100}

# -
# anneal-ep

# BanditOneHigh10
exp210:
	parallel -j 40 \
			--joblog '$(DATA_PATH)/exp210.log' \
			--nice 19 --delay 2 --colsep ',' \
			'epsilon_bandit.py --env_name=BanditOneHigh10-v0 --num_episodes=500 --epsilon=0.45 --epsilon_decay_tau=0.061 --lr_R=0.14 --save=$(DATA_PATH)/exp210_{1}.pkl --interactive=False --debug=False --master_seed={1}' ::: {1..100}

# BanditTwoHigh10
exp211:
	parallel -j 40 \
			--joblog '$(DATA_PATH)/exp211.log' \
			--nice 19 --delay 2 --colsep ',' \
			'epsilon_bandit.py --env_name=BanditTwoHigh10-v0 --num_episodes=500 --epsilon=0.980 --epsilon_decay_tau=0.084 --lr_R=0.194  --save=$(DATA_PATH)/exp211_{1}.pkl --interactive=False --debug=False --master_seed={1}' ::: {1..100}

# BanditUniform121
exp212:
	parallel -j 40 \
			--joblog '$(DATA_PATH)/exp212.log' \
			--nice 19 --delay 2 --colsep ',' \
			'epsilon_bandit.py --env_name=BanditUniform121-v0 --num_episodes=60500 --epsilon=0.850 --lr_R=0.173 --epsilon_decay_tau=0.00777 --save=$(DATA_PATH)/exp212_{1}.pkl --interactive=False --debug=False --master_seed={1}' ::: {1..100}

# HardAndSparse10
exp213:
	parallel -j 40 \
			--joblog '$(DATA_PATH)/exp213.log' \
			--nice 19 --delay 2 --colsep ',' \
			'epsilon_bandit.py --env_name=BanditHardAndSparse10-v0 --num_episodes=50000 --epsilon=0.5148 --epsilon_decay_tau=0.0723 --lr_R=0.000271 --save=$(DATA_PATH)/exp213_{1}.pkl --interactive=False --debug=False --master_seed={1}' ::: {1..100}
			

# ----------------------------------------------------------------------------
# 2-6-2020
# 860b5a07bf5a0232d03550edc7cd94f14ea2621f
#
# Run all agents (for the first time) on DeceptiveBanditOneHigh10.
# Just take HPs from the  OneHigh10 bandit. Why not? 
exp214:
	parallel -j 40 \
			--joblog '$(DATA_PATH)/exp214.log' \
			--nice 19 --delay 2 --colsep ',' \
			'wsls_bandit.py --env_name=DeceptiveBanditOneHigh10-v0 --num_episodes=500 --tie_break='next' --tie_threshold=0.053 --lr_R=0.34 --save=$(DATA_PATH)/exp214_{1}.pkl --interactive=False --debug=False --master_seed={1}' ::: {1..100}

exp215:
	parallel -j 40 \
			--joblog '$(DATA_PATH)/exp215.log' \
			--nice 19 --delay 2 --colsep ',' \
			'softbeta_bandit.py --env_name=DeceptiveBanditOneHigh10-v0 --num_episodes=500 --beta=0.066 --lr_R=0.13 --temp=0.13 --save=$(DATA_PATH)/exp215_{1}.pkl --interactive=False --debug=False --master_seed={1}' ::: {1..100}

exp216:
	parallel -j 40 \
			--joblog '$(DATA_PATH)/exp216.log' \
			--nice 19 --delay 2 --colsep ',' \
			'epsilon_bandit.py --env_name=DeceptiveBanditOneHigh10-v0 --num_episodes=500 --epsilon=0.14 --lr_R=0.087 --save=$(DATA_PATH)/exp216_{1}.pkl --interactive=False --debug=False --master_seed={1}' ::: {1..100}

exp217:
	parallel -j 40 \
			--joblog '$(DATA_PATH)/exp217.log' \
			--nice 19 --delay 2 --colsep ',' \
			'epsilon_bandit.py --env_name=DeceptiveBanditOneHigh10-v0 --num_episodes=500 --epsilon=0.45 --epsilon_decay_tau=0.061 --lr_R=0.14 --save=$(DATA_PATH)/exp217_{1}.pkl --interactive=False --debug=False --master_seed={1}' ::: {1..100}


# ---------------------------------------------------------------------------
# f5e76cc8765953dc39809fd14de10903a605a6e6
# 2-13-2020
#
# The first try on DeceptiveBanditOneHigh10 had too long a learning period
# (exp214-217). This hid the deception. The deception is the point. 
# Try again with. 
# This time the game ends after 20 steps. 
#
# The expected value of the best arm is 0.4. The expected value of the rest of the arms is 0.2.
# 
# A new round of HP tuning is needed because of the short duration of these experiments. To check things out let's just try some params from BanditOneHigh10. This is what I did last time as well.

# RESULTS: meta does better than the rest. Both in terms of final p_best and 
#          rewards. ep is not far off. softbeta and anneal-ep are bad. 
#          I should give them five trials at the end? 

exp218:
	parallel -j 40 \
			--joblog '$(DATA_PATH)/exp218.log' \
			--nice 19 --delay 2 --colsep ',' \
			'wsls_bandit.py --env_name=DeceptiveBanditOneHigh10-v0 --num_episodes=20 --tie_break='next' --tie_threshold=0.053 --lr_R=0.34 --save=$(DATA_PATH)/exp218_{1}.pkl --interactive=False --debug=False --master_seed={1}' ::: {1..100}

exp219:
	parallel -j 40 \
			--joblog '$(DATA_PATH)/exp219.log' \
			--nice 19 --delay 2 --colsep ',' \
			'softbeta_bandit.py --env_name=DeceptiveBanditOneHigh10-v0 --num_episodes=20 --beta=0.066 --lr_R=0.13 --temp=0.13 --save=$(DATA_PATH)/exp219_{1}.pkl --interactive=False --debug=False --master_seed={1}' ::: {1..100}

exp220:
	parallel -j 40 \
			--joblog '$(DATA_PATH)/exp220.log' \
			--nice 19 --delay 2 --colsep ',' \
			'epsilon_bandit.py --env_name=DeceptiveBanditOneHigh10-v0 --num_episodes=20 --epsilon=0.14 --lr_R=0.087 --save=$(DATA_PATH)/exp220_{1}.pkl --interactive=False --debug=False --master_seed={1}' ::: {1..100}

exp221:
	parallel -j 40 \
			--joblog '$(DATA_PATH)/exp221.log' \
			--nice 19 --delay 2 --colsep ',' \
			'epsilon_bandit.py --env_name=DeceptiveBanditOneHigh10-v0 --num_episodes=20 --epsilon=0.45 --epsilon_decay_tau=0.061 --lr_R=0.14 --save=$(DATA_PATH)/exp221_{1}.pkl --interactive=False --debug=False --master_seed={1}' ::: {1..100}


# ---------------------------------------------------------------------------
# 252bcd09d3c8beffbdff3e4a4c6fa26fbd27b87b
# 2-14-2020
#
# Another round of deception exps. Redfined deception as a single U-turn 
# bandit. As you walk away at the start reward is nagetive. Once you hit the 
# U-turn point value becomes positive. 
# - Overall the p(reward) for the U-turn arm is 0.8.  
# - All other arms have p(reward) = 0.2. 
# - If the best arm is choosen about 50 times its higher expected value 
#   should be clear. 
#
# RESULTS: Only meta/pipi makes any progress in finding the best arm in the
#          end. Even then it's a small improvement. 
#          Let's run the experiment out twice as long.
exp222:
	parallel -j 40 \
			--joblog '$(DATA_PATH)/exp222.log' \
			--nice 19 --delay 2 --colsep ',' \
			'wsls_bandit.py --env_name=DeceptiveBanditOneHigh10-v0 --num_episodes=50 --tie_break='next' --tie_threshold=0.053 --lr_R=0.34 --save=$(DATA_PATH)/exp222_{1}.pkl --interactive=False --debug=False --master_seed={1}' ::: {1..100}

exp223:
	parallel -j 40 \
			--joblog '$(DATA_PATH)/exp223.log' \
			--nice 19 --delay 2 --colsep ',' \
			'softbeta_bandit.py --env_name=DeceptiveBanditOneHigh10-v0 --num_episodes=50 --beta=0.066 --lr_R=0.13 --temp=0.13 --save=$(DATA_PATH)/exp223_{1}.pkl --interactive=False --debug=False --master_seed={1}' ::: {1..100}

exp224:
	parallel -j 40 \
			--joblog '$(DATA_PATH)/exp224.log' \
			--nice 19 --delay 2 --colsep ',' \
			'epsilon_bandit.py --env_name=DeceptiveBanditOneHigh10-v0 --num_episodes=50 --epsilon=0.14 --lr_R=0.087 --save=$(DATA_PATH)/exp224_{1}.pkl --interactive=False --debug=False --master_seed={1}' ::: {1..100}

exp225:
	parallel -j 40 \
			--joblog '$(DATA_PATH)/exp225.log' \
			--nice 19 --delay 2 --colsep ',' \
			'epsilon_bandit.py --env_name=DeceptiveBanditOneHigh10-v0 --num_episodes=50 --epsilon=0.45 --epsilon_decay_tau=0.061 --lr_R=0.14 --save=$(DATA_PATH)/exp225_{1}.pkl --interactive=False --debug=False --master_seed={1}' ::: {1..100}

# ---------------------------------------------------------------------------
# 2-14-2020
# 
# Repeat last exps222-225 but with num_episodes=100. Which is up from 50.
#
# RESULTS: meta didn't converge to best. None did. Time to try some HP tuning.
#          See what happends then. I think this task design is right.
exp226:
	parallel -j 40 \
			--joblog '$(DATA_PATH)/exp226.log' \
			--nice 19 --delay 2 --colsep ',' \
			'wsls_bandit.py --env_name=DeceptiveBanditOneHigh10-v0 --num_episodes=100 --tie_break='next' --tie_threshold=0.053 --lr_R=0.34 --save=$(DATA_PATH)/exp226_{1}.pkl --interactive=False --debug=False --master_seed={1}' ::: {1..100}

exp227:
	parallel -j 40 \
			--joblog '$(DATA_PATH)/exp227.log' \
			--nice 19 --delay 2 --colsep ',' \
			'softbeta_bandit.py --env_name=DeceptiveBanditOneHigh10-v0 --num_episodes=100 --beta=0.066 --lr_R=0.13 --temp=0.13 --save=$(DATA_PATH)/exp227_{1}.pkl --interactive=False --debug=False --master_seed={1}' ::: {1..100}

exp228:
	parallel -j 40 \
			--joblog '$(DATA_PATH)/exp228.log' \
			--nice 19 --delay 2 --colsep ',' \
			'epsilon_bandit.py --env_name=DeceptiveBanditOneHigh10-v0 --num_episodes=100 --epsilon=0.14 --lr_R=0.087 --save=$(DATA_PATH)/exp228_{1}.pkl --interactive=False --debug=False --master_seed={1}' ::: {1..100}

exp229:
	parallel -j 40 \
			--joblog '$(DATA_PATH)/exp229.log' \
			--nice 19 --delay 2 --colsep ',' \
			'epsilon_bandit.py --env_name=DeceptiveBanditOneHigh10-v0 --num_episodes=100 --epsilon=0.45 --epsilon_decay_tau=0.061 --lr_R=0.14 --save=$(DATA_PATH)/exp229_{1}.pkl --interactive=False --debug=False --master_seed={1}' ::: {1..100}

# ----------------------------------------------------------------------------
# 2-15-202
# cbc1dac28a486aa2360d9fcd5a201fbb603f9bd2
#
# Replicator HP tuning - round 1 - DeceptiveBanditOneHigh10

exp230:
	tune_bandit.py replicator $(DATA_PATH)/exp230 \
		--exp_name='wsls_bandit' \
		--env_name=DeceptiveBanditOneHigh10-v0 \
        --num_iterations=100 \
        --num_episodes=50 \
		--num_replicators=800 \
		--num_processes=40 \
		--perturbation=0.05 \
		--metric="total_R" \
		--tie_threshold='(1e-5, .1)' \
		--lr_R='(0.000001, 0.4)'

exp231:
	tune_bandit.py replicator $(DATA_PATH)/exp231 \
		--exp_name='softbeta_bandit' \
		--env_name=DeceptiveBanditOneHigh10-v0 \
        --num_iterations=100 \
        --num_episodes=50 \
		--num_replicators=800 \
		--num_processes=40 \
		--perturbation=0.05 \
		--metric="total_R" \
		--beta='(1e-3, 3)' \
		--lr_R='(1e-5, 0.2)' \
		--temp='(1e-1, 3)'

exp232:
	tune_bandit.py replicator $(DATA_PATH)/exp232 \
		--exp_name='epsilon_bandit' \
		--env_name=DeceptiveBanditOneHigh10-v0 \
		--num_iterations=100 \
        --num_episodes=50 \
		--num_replicators=800 \
		--num_processes=40 \
		--perturbation=0.05 \
		--metric="total_R" \
		--epsilon='(0.01, 0.99)' \
		--lr_R='(0.000000001, 0.2)'

exp233:
	tune_bandit.py replicator $(DATA_PATH)/exp233 \
		--exp_name='epsilon_bandit' \
		--env_name=BanditOneHigh10-v0 \
		--num_iterations=100 \
        --num_episodes=50 \
		--num_replicators=800 \
		--num_processes=40 \
		--perturbation=0.05 \
		--metric="total_R" \
		--epsilon='(0.01, 0.99)' \
		--epsilon_decay_tau='(0.000000001, 0.1)' \
		--lr_R='(0.000000001, 0.2)' 

# ----------------------------------------------------------------------------
# 2-17-2020
# 5befd1fcf1f09d13ad96e87c4fe5b2d1e458f790 
#
# RERUN after FIX 
# Replicator HP tuning - round 3 - DeceptiveBanditOneHigh10

exp234:
	tune_bandit.py replicator $(DATA_PATH)/exp234 \
		--exp_name='wsls_bandit' \
		--env_name=DeceptiveBanditOneHigh10-v0 \
        --num_iterations=100 \
        --num_episodes=50 \
		--num_replicators=800 \
		--num_processes=40 \
		--perturbation=0.05 \
		--metric="total_R" \
		--tie_threshold='(1e-5, .1)' \
		--lr_R='(0.000001, 0.4)'

exp235:
	tune_bandit.py replicator $(DATA_PATH)/exp235 \
		--exp_name='softbeta_bandit' \
		--env_name=DeceptiveBanditOneHigh10-v0 \
        --num_iterations=100 \
        --num_episodes=50 \
		--num_replicators=800 \
		--num_processes=40 \
		--perturbation=0.05 \
		--metric="total_R" \
		--beta='(1e-3, 3)' \
		--lr_R='(1e-5, 0.2)' \
		--temp='(1e-1, 3)'

exp236:
	tune_bandit.py replicator $(DATA_PATH)/exp236 \
		--exp_name='epsilon_bandit' \
		--env_name=DeceptiveBanditOneHigh10-v0 \
		--num_iterations=100 \
        --num_episodes=50 \
		--num_replicators=800 \
		--num_processes=40 \
		--perturbation=0.05 \
		--metric="total_R" \
		--epsilon='(0.01, 0.99)' \
		--lr_R='(0.000000001, 0.2)'

exp237:
	tune_bandit.py replicator $(DATA_PATH)/exp237 \
		--exp_name='epsilon_bandit' \
		--env_name=DeceptiveBanditOneHigh10-v0 \
		--num_iterations=100 \
        --num_episodes=50 \
		--num_replicators=800 \
		--num_processes=40 \
		--perturbation=0.05 \
		--metric="total_R" \
		--epsilon='(0.01, 0.99)' \
		--epsilon_decay_tau='(0.000000001, 0.1)' \
		--lr_R='(0.000000001, 0.2)' 

# ----------------------------------------------------------------------------
# 2-17-20
# 4f6d73c4b3c62f59d777061b3cb839b5cd63d2e9
#
# Redesigned slightly the memory model and how E is calculated for meta
# and the beta variations. Re-running those on the standard four bandits.
# Though the TwoHigh condition is dropped. So that is the three standard 
# bandits now. Two will be replaced in the paper by a Deception exp.

# Meta
# BanditOneHigh10
exp238:
	parallel -j 40 \
			--joblog '$(DATA_PATH)/exp238.log' \
			--nice 19 --delay 2 --colsep ',' \
			'wsls_bandit.py --env_name=BanditOneHigh10-v0 --num_episodes=500 --tie_break='next' --tie_threshold=0.053 --lr_R=0.34 --save=$(DATA_PATH)/exp238_{1}.pkl --interactive=False --debug=False --master_seed={1}' ::: {1..100}

# BanditUniform121
exp239:
	parallel -j 40 \
			--joblog '$(DATA_PATH)/exp239.log' \
			--nice 19 --delay 2 --colsep ',' \
			'wsls_bandit.py --env_name=BanditUniform121-v0 --num_episodes=60500 --tie_break='next' --tie_threshold=0.00355 --lr_R=0.147 --save=$(DATA_PATH)/exp239_{1}.pkl --interactive=False --debug=False --master_seed={1}' ::: {1..100}

# HardAndSparse10
exp240:
	parallel -j 40 \
			--joblog '$(DATA_PATH)/exp240.log' \
			--nice 19 --delay 2 --colsep ',' \
			'wsls_bandit.py --env_name=BanditHardAndSparse10-v0 --num_episodes=50000 --tie_break='next' --tie_threshold=5.782e-09 --lr_R=0.00112 --save=$(DATA_PATH)/exp240_{1}.pkl --interactive=False --debug=False --master_seed={1}' ::: {1..100}

# -
# beta:
# BanditOneHigh10
exp241:
	parallel -j 40 \
			--joblog '$(DATA_PATH)/exp241.log' \
			--nice 19 --delay 2 --colsep ',' \
			'beta_bandit.py --env_name=BanditOneHigh10-v0 --num_episodes=500 --tie_break='next' --beta=0.22 --lr_R=0.18 --save=$(DATA_PATH)/exp241_{1}.pkl --interactive=False --debug=False --master_seed={1}' ::: {1..100}

# BanditUniform121
exp242:
	parallel -j 40 \
			--joblog '$(DATA_PATH)/exp242.log' \
			--nice 19 --delay 2 --colsep ',' \
			'beta_bandit.py --env_name=BanditUniform121-v0 --num_episodes=60500 --tie_break='next' --beta=0.056 --lr_R=0.141 --save=$(DATA_PATH)/exp242_{1}.pkl --interactive=False --debug=False --master_seed={1}' ::: {1..100}

# HardAndSparse10
exp243:
	parallel -j 40 \
			--joblog '$(DATA_PATH)/exp243.log' \
			--nice 19 --delay 2 --colsep ',' \
			'beta_bandit.py --env_name=BanditHardAndSparse10-v0 --num_episodes=50000 --tie_break='next' --beta=0.217 --lr_R=0.051 --save=$(DATA_PATH)/exp243_{1}.pkl --interactive=False --debug=False --master_seed={1}' ::: {1..100}

# -
# softbeta
# BanditOneHigh10
exp244:
	parallel -j 40 \
			--joblog '$(DATA_PATH)/exp244.log' \
			--nice 19 --delay 2 --colsep ',' \
			'softbeta_bandit.py --env_name=BanditOneHigh10-v0 --num_episodes=500 --beta=0.066 --lr_R=0.13 --temp=0.13 --save=$(DATA_PATH)/exp244_{1}.pkl --interactive=False --debug=False --master_seed={1}' ::: {1..100}

# BanditUniform121
exp245:
	parallel -j 40 \
			--joblog '$(DATA_PATH)/exp245.log' \
			--nice 19 --delay 2 --colsep ',' \
			'softbeta_bandit.py --env_name=BanditUniform121-v0 --num_episodes=60500 --beta=0.125 --lr_R=0.174 --temp=0.0811 --save=$(DATA_PATH)/exp245_{1}.pkl --interactive=False --debug=False --master_seed={1}' ::: {1..100}

# HardAndSparse10
exp246:
	parallel -j 40 \
			--joblog '$(DATA_PATH)/exp246.log' \
			--nice 19 --delay 2 --colsep ',' \
			'softbeta_bandit.py --env_name=BanditHardAndSparse10-v0 --num_episodes=50000 --beta=2.140 --lr_R=0.128 --temp=5.045 --save=$(DATA_PATH)/exp246_{1}.pkl --interactive=False --debug=False --master_seed={1}' ::: {1..100}

# ----------------------------------------------------------------------------
# For the sake of having everything run from the same commit, I am also 
# re-running the ep and random models now...

# -
# ep

# BanditOneHigh10
exp247:
	parallel -j 40 \
			--joblog '$(DATA_PATH)/exp247.log' \
			--nice 19 --delay 2 --colsep ',' \
			'epsilon_bandit.py --env_name=BanditOneHigh10-v0 --num_episodes=500 --epsilon=0.14 --lr_R=0.087 --save=$(DATA_PATH)/exp247_{1}.pkl --interactive=False --debug=False --master_seed={1}' ::: {1..100}

# BanditUniform121
exp248:
	parallel -j 40 \
			--joblog '$(DATA_PATH)/exp248.log' \
			--nice 19 --delay 2 --colsep ',' \
			'epsilon_bandit.py --env_name=BanditUniform121-v0 --num_episodes=60500 --epsilon=0.0117 --lr_R=0.137 --save=$(DATA_PATH)/exp248_{1}.pkl --interactive=False --debug=False --master_seed={1}' ::: {1..100}

# HardAndSparse10
exp249:
	parallel -j 40 \
			--joblog '$(DATA_PATH)/exp249.log' \
			--nice 19 --delay 2 --colsep ',' \
			'epsilon_bandit.py --env_name=BanditHardAndSparse10-v0 --num_episodes=50000 --epsilon=0.4057 --lr_R=0.000484 --save=$(DATA_PATH)/exp249_{1}.pkl --interactive=False --debug=False --master_seed={1}' ::: {1..100}

# -
# anneal-ep

# BanditOneHigh10
exp250:
	parallel -j 40 \
			--joblog '$(DATA_PATH)/exp250.log' \
			--nice 19 --delay 2 --colsep ',' \
			'epsilon_bandit.py --env_name=BanditOneHigh10-v0 --num_episodes=500 --epsilon=0.45 --epsilon_decay_tau=0.061 --lr_R=0.14 --save=$(DATA_PATH)/exp250_{1}.pkl --interactive=False --debug=False --master_seed={1}' ::: {1..100}

# BanditUniform121
exp251:
	parallel -j 40 \
			--joblog '$(DATA_PATH)/exp251.log' \
			--nice 19 --delay 2 --colsep ',' \
			'epsilon_bandit.py --env_name=BanditUniform121-v0 --num_episodes=60500 --epsilon=0.850 --lr_R=0.173 --epsilon_decay_tau=0.00777 --save=$(DATA_PATH)/exp251_{1}.pkl --interactive=False --debug=False --master_seed={1}' ::: {1..100}

# HardAndSparse10
exp252:
	parallel -j 40 \
			--joblog '$(DATA_PATH)/exp252.log' \
			--nice 19 --delay 2 --colsep ',' \
			'epsilon_bandit.py --env_name=BanditHardAndSparse10-v0 --num_episodes=50000 --epsilon=0.5148 --epsilon_decay_tau=0.0723 --lr_R=0.000271 --save=$(DATA_PATH)/exp252_{1}.pkl --interactive=False --debug=False --master_seed={1}' ::: {1..100}
			
# -
# Random
# BanditOneHigh10
exp253:
	parallel -j 40 \
			--joblog '$(DATA_PATH)/exp253.log' \
			--nice 19 --delay 2 --colsep ',' \
			'random_bandit.py --env_name=BanditOneHigh10-v0 --num_episodes=500  --lr_R=0.1 --save=$(DATA_PATH)/exp253_{1}.pkl --interactive=False --debug=False --master_seed={1}' ::: {1..100}

# BanditUniform121
exp254:
	parallel -j 40 \
			--joblog '$(DATA_PATH)/exp254.log' \
			--nice 19 --delay 2 --colsep ',' \
			'random_bandit.py --env_name=BanditUniform121-v0 --num_episodes=60500  --lr_R=0.1 --save=$(DATA_PATH)/exp254_{1}.pkl --interactive=False --debug=False --master_seed={1}' ::: {1..100}

# BanditHardAndSparse10
exp255:
	parallel -j 40 \
			--joblog '$(DATA_PATH)/exp255.log' \
			--nice 19 --delay 2 --colsep ',' \
			'random_bandit.py --env_name=BanditHardAndSparse10-v0 --num_episodes=50000 --lr_R=0.1 --save=$(DATA_PATH)/exp255_{1}.pkl --interactive=False --debug=False --master_seed={1}' ::: {1..100}


# ---------------------------------------------------------------------------
# 2-18-2020
# 4f6d73c4b3c62f59d777061b3cb839b5cd63d2e9
#
# Random tuning for DeceptiveBanditOneHigh10 - round 1
#
# Replicator tuning is doing less well than my efforts at hand
# tuning. Given the replicator is my own experiment in HP, best
# to now abaonden it and move to something simpler. Plain old
# random search. 
# 
# If this works out I'll need to retune all the other models/agents/tasks.

# RESULT: only looked at meta, which I know from hand tuning can find the right
#         best arm. The others can't. Again, by hand. Though this seems robust
#         enough when I putter I take it seriously. 
#         Perhaps num_episodes is too small? Let's double it and try again.
exp256:
	tune_bandit.py random $(DATA_PATH)/exp256 \
		--exp_name='wsls_bandit' \
		--env_name=DeceptiveBanditOneHigh10-v0 \
        --num_samples=5000 \
        --num_episodes=50 \
		--num_processes=40 \
		--metric="total_R" \
		--tie_threshold='(1e-5, .1)' \
		--lr_R='(0.000001, 0.4)'

exp257:
	tune_bandit.py random $(DATA_PATH)/exp257 \
		--exp_name='softbeta_bandit' \
		--env_name=DeceptiveBanditOneHigh10-v0 \
        --num_samples=5000 \
        --num_episodes=50 \
		--num_processes=40 \
		--metric="total_R" \
		--beta='(1e-3, 3)' \
		--lr_R='(1e-5, 0.2)' \
		--temp='(1e-1, 3)'

exp258:
	tune_bandit.py random $(DATA_PATH)/exp258 \
		--exp_name='epsilon_bandit' \
		--env_name=DeceptiveBanditOneHigh10-v0 \
		--num_samples=5000 \
        --num_episodes=50 \
		--num_processes=40 \
		--metric="total_R" \
		--epsilon='(0.01, 0.99)' \
		--lr_R='(0.000000001, 0.2)'

exp259:
	tune_bandit.py random $(DATA_PATH)/exp259 \
		--exp_name='epsilon_bandit' \
		--env_name=DeceptiveBanditOneHigh10-v0 \
		--num_samples=5000 \
        --num_episodes=50 \
		--num_processes=40 \
		--metric="total_R" \
		--epsilon='(0.01, 0.99)' \
		--epsilon_decay_tau='(0.000000001, 0.1)' \
		--lr_R='(0.000000001, 0.2)' 

# ---------------------------------------------------------------------------
# 2-18-2020
# 4f6d73c4b3c62f59d777061b3cb839b5cd63d2e9
#
# Random tuning for DeceptiveBanditOneHigh10 - round 2
# - Increased to num_episodes=100
# - Tweak search ranges

# RESULTS: exp260 is looks promising in the low eta regime. This matches 
#          hand tuning. 
#          exp261 the critic does not learn the best arm
#          exp262 the critic does not learn the best arm
#          exp263 the critic does not learn the best arm
#          The above hold for num_episodes=100. If that is increased to 1000
#          all the agents solve the task. But this is a much easier task.
#          Deception doesn't matter really when you have noise and endless 
#          attempts. That is, for AI. But for animals? For us?
exp260:
	tune_bandit.py random $(DATA_PATH)/exp260 \
		--exp_name='wsls_bandit' \
		--env_name=DeceptiveBanditOneHigh10-v0 \
        --num_samples=5000 \
        --num_episodes=100 \
		--num_processes=40 \
		--metric="total_R" \
		--tie_threshold='(1e-6, .1)' \
		--lr_R='(0.000001, 0.4)'

exp261:
	tune_bandit.py random $(DATA_PATH)/exp261 \
		--exp_name='softbeta_bandit' \
		--env_name=DeceptiveBanditOneHigh10-v0 \
        --num_samples=5000 \
        --num_episodes=100 \
		--num_processes=40 \
		--metric="total_R" \
		--beta='(2e-3, 20)' \
		--lr_R='(1e-5, 0.2)' \
		--temp='(1e-1, 3)'

exp262:
	tune_bandit.py random $(DATA_PATH)/exp262 \
		--exp_name='epsilon_bandit' \
		--env_name=DeceptiveBanditOneHigh10-v0 \
		--num_samples=5000 \
        --num_episodes=100 \
		--num_processes=40 \
		--metric="total_R" \
		--epsilon='(0.01, 0.99)' \
		--lr_R='(0.000000001, 0.2)'

exp263:
	tune_bandit.py random $(DATA_PATH)/exp263 \
		--exp_name='epsilon_bandit' \
		--env_name=DeceptiveBanditOneHigh10-v0 \
		--num_samples=5000 \
        --num_episodes=100 \
		--num_processes=40 \
		--metric="total_R" \
		--epsilon='(0.01, 0.99)' \
		--epsilon_decay_tau='(0.000000001, 0.1)' \
		--lr_R='(0.000000001, 0.2)' 

# ----------------------------------------------------------------------------
# RERUN after FIX 
# Now that random is giving something more sensible. Let's try replicator again.
# See if there are further improvments.
#
# It won't be worth using this in the paper. Explanation burden. Unless for
# some reason it does A LOT better. This looks unlikely.
#
# RESULTS: exp264 the replicator version of 260 looks quite a bit stronger
#          for consistency (there are many models near 40 total R). exp260
#          only had 1-3 models near this best point. Peak performance between
#          the two methods is looking the same. To a first pass analysis.
#
#          In comparison exp265 (replicator) and exp261 (random) look about 
#          the same. Both in peak AND consistency. I'd need to run A LOT more
#          samples of both to claim this as a result though. Run a 100000
#          sample versions?
# 
# 		   A similiar no improvement trend held between 262/6 and 263/7.
# 		   Again need a large N run.
#          
#          A this point a large N random run seems like a safe bet. It
#          would be easier to explain and the best models are the same-ish.
exp264:
	tune_bandit.py replicator $(DATA_PATH)/exp264 \
		--exp_name='wsls_bandit' \
		--env_name=DeceptiveBanditOneHigh10-v0 \
        --num_iterations=100 \
        --num_episodes=100 \
		--num_replicators=800 \
		--num_processes=40 \
		--perturbation=0.05 \
		--metric="total_R" \
		--tie_threshold='(1e-6, .1)' \
		--lr_R='(0.000001, 0.4)'

exp265:
	tune_bandit.py replicator $(DATA_PATH)/exp265 \
		--exp_name='softbeta_bandit' \
		--env_name=DeceptiveBanditOneHigh10-v0 \
        --num_iterations=100 \
        --num_episodes=100 \
		--num_replicators=800 \
		--num_processes=40 \
		--perturbation=0.05 \
		--metric="total_R" \
		--beta='(1e-3, 30)' \
		--lr_R='(1e-5, 0.2)' \
		--temp='(1e-1, 3)'

exp266:
	tune_bandit.py replicator $(DATA_PATH)/exp266 \
		--exp_name='epsilon_bandit' \
		--env_name=DeceptiveBanditOneHigh10-v0 \
		--num_iterations=100 \
        --num_episodes=100 \
		--num_replicators=800 \
		--num_processes=40 \
		--perturbation=0.05 \
		--metric="total_R" \
		--epsilon='(0.01, 0.99)' \
		--lr_R='(0.000000001, 0.2)'

exp267:
	tune_bandit.py replicator $(DATA_PATH)/exp267 \
		--exp_name='epsilon_bandit' \
		--env_name=DeceptiveBanditOneHigh10-v0 \
		--num_iterations=100 \
        --num_episodes=100 \
		--num_replicators=800 \
		--num_processes=40 \
		--perturbation=0.05 \
		--metric="total_R" \
		--epsilon='(0.01, 0.99)' \
		--epsilon_decay_tau='(0.000000001, 0.1)' \
		--lr_R='(0.000000001, 0.2)' 

# --------------------------------------------------------------------------
# 4-1-2020
#
#
# For UCSD talk I need a set of results for Deception10. The replicator are
# the most reliable looking, so run the best of those in n=100 
#
# Copy best params from the analysis ipynb notebooks:
# exp264:
# {'tie_threshold': 0.0014841871547063938,
#  'lr_R': 0.351967767293941,
#  'total_R': 46.0}
# exp265:
# {'beta': 0.3119265393399749,
#  'lr_R': 0.1862363578566501,
#  'temp': 0.08251127278311668,
#  'total_R': 32.111111111111114}
# exp266:
# {'epsilon': 0.011348183941763144, 
#  'lr_R': 0.11581362623794718,
#  'total_R': 31.0}
# exp267:
# {'epsilon': 0.08053999525431153,
#  'epsilon_decay_tau': 0.039934523969666534,
#  'lr_R': 0.10524012976913527,
#  'total_R': 33.0}

# meta - sample results from exp264 best_params
exp269:
	parallel -j 40 \
			--joblog '$(DATA_PATH)/exp269.log' \
			--nice 19 --delay 2 --colsep ',' \
			'wsls_bandit.py --env_name=DeceptiveBanditOneHigh10-v0 --num_episodes=100 --tie_break='next' --tie_threshold=0.0014 --lr_R=0.35 --save=$(DATA_PATH)/exp269_{1}.pkl --interactive=False --debug=False --master_seed={1}' ::: {1..100}

# ep - sample results from exp266 best_params
exp270:
	parallel -j 40 \
			--joblog '$(DATA_PATH)/exp270.log' \
			--nice 19 --delay 2 --colsep ',' \
			'epsilon_bandit.py --env_name=DeceptiveBanditOneHigh10-v0 --num_episodes=100 --epsilon=0.011 --lr_R=0.115 --save=$(DATA_PATH)/exp270_{1}.pkl --interactive=False --debug=False --master_seed={1}' ::: {1..100}


# anneal - sample results from exp267 best_params
exp271:
	parallel -j 40 \
			--joblog '$(DATA_PATH)/exp271.log' \
			--nice 19 --delay 2 --colsep ',' \
			'epsilon_bandit.py --env_name=DeceptiveBanditOneHigh10-v0 --num_episodes=100 --epsilon=0.080 --epsilon_decay_tau=0.039 --lr_R=0.10 --save=$(DATA_PATH)/exp271_{1}.pkl --interactive=False --debug=False --master_seed={1}' ::: {1..100}

# softbeta - sample results from exp265 best_params
exp272:
	parallel -j 40 \
			--joblog '$(DATA_PATH)/exp272.log' \
			--nice 19 --delay 2 --colsep ',' \
			'softbeta_bandit.py --env_name=DeceptiveBanditOneHigh10-v0 --num_episodes=100 --beta=0.31 --lr_R=0.18 --temp=0.082 --save=$(DATA_PATH)/exp272_{1}.pkl --interactive=False --debug=False --master_seed={1}' ::: {1..100}

# random - just run 100. (There is nothing to tune.)
exp273:
	parallel -j 40 \
			--joblog '$(DATA_PATH)/exp273.log' \
			--nice 19 --delay 2 --colsep ',' \
			'random_bandit.py --env_name=DeceptiveBanditOneHigh10-v0 --num_episodes=100  --lr_R=0.1 --save=$(DATA_PATH)/exp273_{1}.pkl --interactive=False --debug=False --master_seed={1}' ::: {1..100}
		

# -------------------------------------------------------------------------
# 4-14-2020
# e39d0f4f3b70cf002e5609727e26bfbb54b8cb65

# Another, even larger, HP tuning set of exps. All agents and envs this time.
#
# Note: Added csv output to all tune_* methods. 
# This will let me more easily run best HP, and so 
# testing more models as a result.

#
# RESULTS: Overall it looks like 10000 samples was enough for most agent
#          on most tasks to find consistent solutions. The sampling was also
#          very wide. This should rule out undiscovered improvement as a 
#          criticism in the final results. But N=50000 would be even better.
#          So would N=100000. And so on. I should stop here, I think.

# ---------------
# BanditOneHigh10
# ---------------

# meta
exp278:
	tune_bandit.py random $(DATA_PATH)/exp278 \
		--exp_name='wsls_bandit' \
		--env_name=BanditOneHigh10-v0 \
        --num_samples=10000 \
        --num_episodes=100 \
		--num_processes=40 \
		--metric="total_R" \
		--tie_threshold='(1e-10, .1)' \
		--lr_R='(0.000001, 0.4)'

# ep
exp279:
	tune_bandit.py random $(DATA_PATH)/exp279 \
		--exp_name='epsilon_bandit' \
		--env_name=BanditOneHigh10-v0 \
		--num_samples=10000 \
        --num_episodes=100 \
		--num_processes=40 \
		--metric="total_R" \
		--epsilon='(0.01, 0.99)' \
		--lr_R='(0.000000001, 0.4)' 

# anneal-ep
exp280:
	tune_bandit.py random $(DATA_PATH)/exp280 \
		--exp_name='epsilon_bandit' \
		--env_name=BanditOneHigh10-v0 \
		--num_samples=10000 \
        --num_episodes=100 \
		--num_processes=40 \
		--metric="total_R" \
		--epsilon='(0.01, 0.99)' \
		--epsilon_decay_tau='(0.000000001, 0.1)' \
		--lr_R='(0.000000001, 0.4)' 

# beta
exp281:
	tune_bandit.py random $(DATA_PATH)/exp281 \
		--exp_name='softbeta_bandit' \
		--env_name=BanditOneHigh10-v0 \
		--num_samples=10000 \
        --num_episodes=100 \
		--num_processes=40 \
		--metric="total_R" \
		--beta='(1e-3, 10)' \
		--lr_R='(1e-5, 0.4)' \
		--temp='(1e-1, 3)'

# ---------------------
# BanditHardAndSparse10
# ---------------------

# meta
exp282:
	tune_bandit.py random $(DATA_PATH)/exp282 \
		--exp_name='wsls_bandit' \
		--env_name=BanditHardAndSparse10-v0 \
        --num_samples=10000 \
        --num_episodes=50000 \
		--num_processes=40 \
		--metric="total_R" \
		--tie_threshold='(1e-10, .1)' \
		--lr_R='(0.000001, 0.4)'

# ep
exp283:
	tune_bandit.py random $(DATA_PATH)/exp283 \
		--exp_name='epsilon_bandit' \
		--env_name=BanditHardAndSparse10-v0 \
		--num_samples=10000 \
        --num_episodes=50000 \
		--num_processes=40 \
		--metric="total_R" \
		--epsilon='(0.01, 0.99)' \
		--lr_R='(0.000000001, 0.4)' 

# anneal-ep
exp284:
	tune_bandit.py random $(DATA_PATH)/exp284 \
		--exp_name='epsilon_bandit' \
		--env_name=BanditHardAndSparse10-v0 \
		--num_samples=10000 \
        --num_episodes=50000 \
		--num_processes=40 \
		--metric="total_R" \
		--epsilon='(0.01, 0.99)' \
		--epsilon_decay_tau='(0.000000001, 0.1)' \
		--lr_R='(0.000000001, 0.4)' 

# beta
exp285:
	tune_bandit.py random $(DATA_PATH)/exp285 \
		--exp_name='softbeta_bandit' \
		--env_name=BanditHardAndSparse10-v0 \
		--num_samples=10000 \
        --num_episodes=50000 \
		--num_processes=40 \
		--metric="total_R" \
		--beta='(1e-3, 10)' \
		--lr_R='(1e-5, 0.4)' \
		--temp='(1e-1, 3)'

# ----------------
# BanditUniform121
# ----------------

# meta
exp286:
	tune_bandit.py random $(DATA_PATH)/exp286 \
		--exp_name='wsls_bandit' \
		--env_name=BanditUniform121-v0 \
        --num_samples=10000 \
        --num_episodes=60500 \
		--num_processes=40 \
		--metric="total_R" \
		--tie_threshold='(1e-10, .1)' \
		--lr_R='(0.000001, 0.4)'

# ep
exp287:
	tune_bandit.py random $(DATA_PATH)/exp287 \
		--exp_name='epsilon_bandit' \
		--env_name=BanditUniform121-v0 \
		--num_samples=10000 \
        --num_episodes=60500 \
		--num_processes=40 \
		--metric="total_R" \
		--epsilon='(0.01, 0.99)' \
		--lr_R='(0.000000001, 0.4)' 

# anneal-ep
exp288:
	tune_bandit.py random $(DATA_PATH)/exp288 \
		--exp_name='epsilon_bandit' \
		--env_name=BanditUniform121-v0 \
		--num_samples=10000 \
        --num_episodes=60500 \
		--num_processes=40 \
		--metric="total_R" \
		--epsilon='(0.01, 0.99)' \
		--epsilon_decay_tau='(0.000000001, 0.1)' \
		--lr_R='(0.000000001, 0.4)' 

# beta
exp289:
	tune_bandit.py random $(DATA_PATH)/exp289 \
		--exp_name='softbeta_bandit' \
		--env_name=BanditUniform121-v0 \
		--num_samples=10000 \
        --num_episodes=60500 \
		--num_processes=40 \
		--metric="total_R" \
		--beta='(1e-3, 10)' \
		--lr_R='(1e-5, 0.4)' \
		--temp='(1e-1, 3)'

# ---------------------
# DeceptiveBanditOneHigh10
# ---------------------

# meta
exp290:
	tune_bandit.py random $(DATA_PATH)/exp290 \
		--exp_name='wsls_bandit' \
		--env_name=DeceptiveBanditOneHigh10-v0 \
		--num_samples=10000 \
        --num_episodes=100 \
		--num_processes=40 \
		--metric="total_R" \
		--tie_threshold='(1e-10, .1)' \
		--lr_R='(0.000001, 0.4)'

exp291:
	tune_bandit.py random $(DATA_PATH)/exp291 \
		--exp_name='epsilon_bandit' \
		--env_name=DeceptiveBanditOneHigh10-v0 \
		--num_samples=10000 \
        --num_episodes=100 \
		--num_processes=40 \
		--metric="total_R" \
		--epsilon='(0.01, 0.99)' \
		--lr_R='(0.000000001, 0.2)'

exp292:
	tune_bandit.py random $(DATA_PATH)/exp292 \
		--exp_name='epsilon_bandit' \
		--env_name=DeceptiveBanditOneHigh10-v0 \
		--num_samples=10000 \
        --num_episodes=100 \
		--num_processes=40 \
		--metric="total_R" \
		--epsilon='(0.01, 0.99)' \
		--epsilon_decay_tau='(0.000000001, 0.1)' \
		--lr_R='(0.000000001, 0.2)' 

exp293:
	tune_bandit.py random $(DATA_PATH)/exp293 \
		--exp_name='softbeta_bandit' \
		--env_name=DeceptiveBanditOneHigh10-v0 \
        --num_samples=10000 \
        --num_episodes=100 \
		--num_processes=40 \
		--metric="total_R" \
		--beta='(1e-3, 10)' \
		--lr_R='(1e-5, 0.2)' \
		--temp='(1e-1, 3)'

# ---------------------------------------------------------------------------
# 4-17-2020
# 9ad1ca9fc40f818c3e60c1d6fe2040f7ebc0df6d

# In the above random exps (278-293), often things look fine. 
# I said this in the RESULTS above. 
# But! 
#
# From hand tuning, I know meta can perfectly solve hardandsparse10! 
# It is not here. 
#
# The problem was random sampling in all exps so far has been uniform, 
# but what I really need is loguniform. Doy!

# This way sampling will span the large order of magnitude ranges 
# I need to be search in all agents and envs. 
#
# ....So I added a log_space option to tune_random. 
#
# And so I repeat the 'last tune', one more time!
#
# RESULTS: meta is reliably solving the sparse task, and other agents 
#          performance *seems* to be no worse, and sometimes better. Given 
#          these take a couple days to run, and seem like a decent enough HP
#          sample (N=10000), I'm calling it. 

# ---------------
# BanditOneHigh10
# ---------------

# meta
exp294:
	tune_bandit.py random $(DATA_PATH)/exp294 \
		--exp_name='wsls_bandit' \
		--env_name=BanditOneHigh10-v0 \
        --num_samples=10000 \
        --num_episodes=100 \
		--num_processes=40 \
		--log_space=True \
		--metric="total_R" \
		--tie_threshold='(1e-10, .1)' \
		--lr_R='(0.000001, 0.4)'

# ep
exp295:
	tune_bandit.py random $(DATA_PATH)/exp295 \
		--exp_name='epsilon_bandit' \
		--env_name=BanditOneHigh10-v0 \
		--num_samples=10000 \
        --num_episodes=100 \
		--num_processes=40 \
		--log_space=True \
		--metric="total_R" \
		--epsilon='(0.01, 0.99)' \
		--lr_R='(0.000000001, 0.4)' 

# anneal-ep
exp296:
	tune_bandit.py random $(DATA_PATH)/exp296 \
		--exp_name='epsilon_bandit' \
		--env_name=BanditOneHigh10-v0 \
		--num_samples=10000 \
        --num_episodes=100 \
		--num_processes=40 \
		--log_space=True \
		--metric="total_R" \
		--epsilon='(0.01, 0.99)' \
		--epsilon_decay_tau='(0.000000001, 0.1)' \
		--lr_R='(0.000000001, 0.4)' 

# beta
exp297:
	tune_bandit.py random $(DATA_PATH)/exp297 \
		--exp_name='softbeta_bandit' \
		--env_name=BanditOneHigh10-v0 \
		--num_samples=10000 \
        --num_episodes=100 \
		--num_processes=40 \
		--log_space=True \
		--metric="total_R" \
		--beta='(1e-3, 10)' \
		--lr_R='(1e-5, 0.4)' \
		--temp='(1e-1, 3)'

# ---------------------
# BanditHardAndSparse10
# ---------------------

# meta
exp298:
	tune_bandit.py random $(DATA_PATH)/exp298 \
		--exp_name='wsls_bandit' \
		--env_name=BanditHardAndSparse10-v0 \
        --num_samples=10000 \
        --num_episodes=50000 \
		--num_processes=40 \
		--log_space=True \
		--metric="total_R" \
		--tie_threshold='(1e-10, .1)' \
		--lr_R='(0.000001, 0.4)'

# ep
exp299:
	tune_bandit.py random $(DATA_PATH)/exp299 \
		--exp_name='epsilon_bandit' \
		--env_name=BanditHardAndSparse10-v0 \
		--num_samples=10000 \
        --num_episodes=50000 \
		--num_processes=40 \
		--log_space=True \
		--metric="total_R" \
		--epsilon='(0.01, 0.99)' \
		--lr_R='(0.000000001, 0.4)' 

# anneal-ep
exp300:
	tune_bandit.py random $(DATA_PATH)/exp300 \
		--exp_name='epsilon_bandit' \
		--env_name=BanditHardAndSparse10-v0 \
		--num_samples=10000 \
        --num_episodes=50000 \
		--num_processes=40 \
		--log_space=True \
		--metric="total_R" \
		--epsilon='(0.01, 0.99)' \
		--epsilon_decay_tau='(0.000000001, 0.1)' \
		--lr_R='(0.000000001, 0.4)' 

# beta
exp301:
	tune_bandit.py random $(DATA_PATH)/exp301 \
		--exp_name='softbeta_bandit' \
		--env_name=BanditHardAndSparse10-v0 \
		--num_samples=10000 \
        --num_episodes=50000 \
		--num_processes=40 \
		--log_space=True \
		--metric="total_R" \
		--beta='(1e-3, 10)' \
		--lr_R='(1e-5, 0.4)' \
		--temp='(1e-1, 3)'

# ----------------
# BanditUniform121
# ----------------

# meta
exp302:
	tune_bandit.py random $(DATA_PATH)/exp302 \
		--exp_name='wsls_bandit' \
		--env_name=BanditUniform121-v0 \
        --num_samples=10000 \
        --num_episodes=60500 \
		--num_processes=40 \
		--log_space=True \
		--metric="total_R" \
		--tie_threshold='(1e-10, .1)' \
		--lr_R='(0.000001, 0.4)'

# ep
exp303:
	tune_bandit.py random $(DATA_PATH)/exp303 \
		--exp_name='epsilon_bandit' \
		--env_name=BanditUniform121-v0 \
		--num_samples=10000 \
        --num_episodes=60500 \
		--num_processes=40 \
		--log_space=True \
		--metric="total_R" \
		--epsilon='(0.01, 0.99)' \
		--lr_R='(0.000000001, 0.4)' 

# anneal-ep
exp304:
	tune_bandit.py random $(DATA_PATH)/exp304 \
		--exp_name='epsilon_bandit' \
		--env_name=BanditUniform121-v0 \
		--num_samples=10000 \
        --num_episodes=60500 \
		--num_processes=40 \
		--log_space=True \
		--metric="total_R" \
		--epsilon='(0.01, 0.99)' \
		--epsilon_decay_tau='(0.000000001, 0.1)' \
		--lr_R='(0.000000001, 0.4)' 

# beta
exp305:
	tune_bandit.py random $(DATA_PATH)/exp305 \
		--exp_name='softbeta_bandit' \
		--env_name=BanditUniform121-v0 \
		--num_samples=10000 \
        --num_episodes=60500 \
		--num_processes=40 \
		--metric="total_R" \
		--log_space=True \
		--beta='(1e-3, 10)' \
		--lr_R='(1e-5, 0.4)' \
		--temp='(1e-1, 3)'

# ------------------------
# DeceptiveBanditOneHigh10
# ------------------------

# meta
exp306:
	tune_bandit.py random $(DATA_PATH)/exp306 \
		--exp_name='wsls_bandit' \
		--env_name=DeceptiveBanditOneHigh10-v0 \
		--num_samples=10000 \
        --num_episodes=100 \
		--num_processes=40 \
		--log_space=True \
		--metric="total_R" \
		--tie_threshold='(1e-10, .1)' \
		--lr_R='(0.000001, 0.4)'

exp307:
	tune_bandit.py random $(DATA_PATH)/exp307 \
		--exp_name='epsilon_bandit' \
		--env_name=DeceptiveBanditOneHigh10-v0 \
		--num_samples=10000 \
        --num_episodes=100 \
		--num_processes=40 \
		--log_space=True \
		--metric="total_R" \
		--epsilon='(0.01, 0.99)' \
		--lr_R='(0.000000001, 0.2)'

exp308:
	tune_bandit.py random $(DATA_PATH)/exp308 \
		--exp_name='epsilon_bandit' \
		--env_name=DeceptiveBanditOneHigh10-v0 \
		--num_samples=10000 \
        --num_episodes=100 \
		--num_processes=40 \
		--log_space=True \
		--metric="total_R" \
		--epsilon='(0.01, 0.99)' \
		--epsilon_decay_tau='(0.000000001, 0.1)' \
		--lr_R='(0.000000001, 0.2)' 

exp309:
	tune_bandit.py random $(DATA_PATH)/exp309 \
		--exp_name='softbeta_bandit' \
		--env_name=DeceptiveBanditOneHigh10-v0 \
        --num_samples=10000 \
        --num_episodes=100 \
		--num_processes=40 \
		--log_space=True \
		--metric="total_R" \
		--beta='(1e-3, 10)' \
		--lr_R='(1e-5, 0.2)' \
		--temp='(1e-1, 3)'
		
# ------------------------------------------------------------------------
# 4-20-2020
# 09f45f65e482f0e8ab593cffacc0c58b4daefcf0
# 
# Top 10 hyper-parameters (from last random search exps abour)
# each repeated 10 time, with the different random seed.

# ---------------
# BanditOneHigh10
# ---------------

# meta - use HP from exp294
exp310:
	# Get top 10
	head -n 11 $(DATA_PATH)/exp294_sorted.csv > tmp 
	# Run them 10 times
	parallel -j 4 \
			--joblog '$(DATA_PATH)/exp310.log' \
			--nice 19 --delay 2 --colsep ',' --header : \
			'wsls_bandit.py --env_name=BanditOneHigh10-v0 --num_episodes=100 --tie_break='next' --tie_threshold={tie_threshold} --lr_R={lr_R} --save=$(DATA_PATH)/exp310_{index}_{1}.pkl --interactive=False --debug=False --master_seed={1}' ::: {0..10} :::: tmp
	# Clean up
	rm tmp

# ep - use HP from exp295
exp311:
	# Get top 10
	head -n 11 $(DATA_PATH)/exp295_sorted.csv > tmp 
	# Run them 10 times
	parallel -j 4 \
			--joblog '$(DATA_PATH)/exp311.log' \
			--nice 19 --delay 2 --colsep ',' --header : \
			'epsilon_bandit.py --env_name=BanditOneHigh10-v0 --num_episodes=100 --epsilon={epsilon} --lr_R={lr_R} --save=$(DATA_PATH)/exp311_{index}_{1}.pkl --interactive=False --debug=False --master_seed={1}' ::: {0..10} :::: tmp
	# Clean up
	rm tmp

# anneal-ep - use HP from exp296
exp312:
	# Get top 10
	head -n 11 $(DATA_PATH)/exp296_sorted.csv > tmp 
	# Run them 10 times
	parallel -j 4 \
			--joblog '$(DATA_PATH)/exp312.log' \
			--nice 19 --delay 2 --colsep ',' --header : \
			'epsilon_bandit.py --env_name=BanditOneHigh10-v0 --num_episodes=100 --epsilon={epsilon} --epsilon_decay_tau={epsilon_decay_tau} --lr_R={lr_R} --save=$(DATA_PATH)/exp312_{index}_{1}.pkl --interactive=False --debug=False --master_seed={1}' ::: {0..10} :::: tmp
	# Clean up
	rm tmp

# beta - use HP from exp297
exp313:
	# Get top 10
	head -n 11 $(DATA_PATH)/exp297_sorted.csv > tmp 
	# Run them 10 times
	parallel -j 4 \
			--joblog '$(DATA_PATH)/exp313.log' \
			--nice 19 --delay 2 --colsep ',' --header : \
			'softbeta_bandit.py --env_name=BanditOneHigh10-v0 --num_episodes=100 --beta={beta} --temp={temp} --lr_R={lr_R} --save=$(DATA_PATH)/exp313_{index}_{1}.pkl --interactive=False --debug=False --master_seed={1}' ::: {0..10} :::: tmp
	# Clean up
	rm tmp

# ---------------------
# BanditHardAndSparse10
# ---------------------

# meta - use HP from exp298
exp314:
	# Get top 10
	head -n 11 $(DATA_PATH)/exp298_sorted.csv > tmp 
	# Run them 10 times
	parallel -j 4 \
			--joblog '$(DATA_PATH)/exp314.log' \
			--nice 19 --delay 2 --colsep ',' --header : \
			'wsls_bandit.py --env_name=BanditHardAndSparse10-v0 --num_episodes=50000 --tie_break='next' --tie_threshold={tie_threshold} --lr_R={lr_R} --save=$(DATA_PATH)/exp314_{index}_{1}.pkl --interactive=False --debug=False --master_seed={1}' ::: {0..10} :::: tmp
	# Clean up
	rm tmp

# ep - use HP from exp299
exp315:
	# Get top 10
	head -n 11 $(DATA_PATH)/exp299_sorted.csv > tmp 
	# Run them 10 times
	parallel -j 4 \
			--joblog '$(DATA_PATH)/exp315.log' \
			--nice 19 --delay 2 --colsep ',' --header : \
			'epsilon_bandit.py --env_name=BanditHardAndSparse10-v0 --num_episodes=50000 --epsilon={epsilon} --lr_R={lr_R} --save=$(DATA_PATH)/exp315_{index}_{1}.pkl --interactive=False --debug=False --master_seed={1}' ::: {0..10} :::: tmp
	# Clean up
	rm tmp

# anneal-ep - use HP from exp300
exp316:
	# Get top 10
	head -n 11 $(DATA_PATH)/exp300_sorted.csv > tmp 
	# Run them 10 times
	parallel -j 4 \
			--joblog '$(DATA_PATH)/exp316.log' \
			--nice 19 --delay 2 --colsep ',' --header : \
			'epsilon_bandit.py --env_name=BanditHardAndSparse10-v0 --num_episodes=50000 --epsilon={epsilon} --epsilon_decay_tau={epsilon_decay_tau} --lr_R={lr_R} --save=$(DATA_PATH)/exp316_{index}_{1}.pkl --interactive=False --debug=False --master_seed={1}' ::: {0..10} :::: tmp
	# Clean up
	rm tmp

# beta - use HP from exp301
exp317:
	# Get top 10
	head -n 11 $(DATA_PATH)/exp301_sorted.csv > tmp 
	# Run them 10 times
	parallel -j 4 \
			--joblog '$(DATA_PATH)/exp317.log' \
			--nice 19 --delay 2 --colsep ',' --header : \
			'softbeta_bandit.py --env_name=BanditHardAndSparse10-v0 --num_episodes=50000 --beta={beta} --temp={temp} --lr_R={lr_R} --save=$(DATA_PATH)/exp317_{index}_{1}.pkl --interactive=False --debug=False --master_seed={1}' ::: {0..10} :::: tmp
	# Clean up
	rm tmp


# ----------------
# BanditUniform121
# ----------------
# meta - use HP from exp302
exp318:
	# Get top 10
	head -n 11 $(DATA_PATH)/exp302_sorted.csv > tmp 
	# Run them 10 times
	parallel -j 4 \
			--joblog '$(DATA_PATH)/exp318.log' \
			--nice 19 --delay 2 --colsep ',' --header : \
			'wsls_bandit.py --env_name=BanditUniform121-v0 --num_episodes=60500 --tie_break='next' --tie_threshold={tie_threshold} --lr_R={lr_R} --save=$(DATA_PATH)/exp318_{index}_{1}.pkl --interactive=False --debug=False --master_seed={1}' ::: {0..10} :::: tmp
	# Clean up
	rm tmp

# ep - use HP from exp303
exp319:
	# Get top 10
	head -n 11 $(DATA_PATH)/exp303_sorted.csv > tmp 
	# Run them 10 times
	parallel -j 4 \
			--joblog '$(DATA_PATH)/exp319.log' \
			--nice 19 --delay 2 --colsep ',' --header : \
			'epsilon_bandit.py --env_name=BanditUniform121-v0 --num_episodes=60500 --epsilon={epsilon} --lr_R={lr_R} --save=$(DATA_PATH)/exp319_{index}_{1}.pkl --interactive=False --debug=False --master_seed={1}' ::: {0..10} :::: tmp
	# Clean up
	rm tmp

# ep-anneal - use HP from exp304
exp320:
	# Get top 10
	head -n 11 $(DATA_PATH)/exp304_sorted.csv > tmp 
	# Run them 10 times
	parallel -j 4 \
			--joblog '$(DATA_PATH)/exp320.log' \
			--nice 19 --delay 2 --colsep ',' --header : \
			'epsilon_bandit.py --env_name=BanditUniform121-v0 --num_episodes=60500 --epsilon={epsilon} --epsilon_decay_tau={epsilon_decay_tau} --lr_R={lr_R} --save=$(DATA_PATH)/exp320_{index}_{1}.pkl --interactive=False --debug=False --master_seed={1}' ::: {0..10} :::: tmp
	# Clean up
	rm tmp

# beta - use HP from exp305
exp321:
	# Get top 10
	head -n 11 $(DATA_PATH)/exp305_sorted.csv > tmp 
	# Run them 10 times
	parallel -j 4 \
			--joblog '$(DATA_PATH)/exp321.log' \
			--nice 19 --delay 2 --colsep ',' --header : \
			'softbeta_bandit.py --env_name=BanditUniform121-v0 --num_episodes=60500 --temp={temp} --beta={beta} --lr_R={lr_R} --save=$(DATA_PATH)/exp321_{index}_{1}.pkl --interactive=False --debug=False --master_seed={1}' ::: {0..10} :::: tmp
	# Clean up
	rm tmp

# ------------------------
# DeceptiveBanditOneHigh10
# ------------------------

# meta - use HP from exp306
exp322:
	# Get top 10
	head -n 11 $(DATA_PATH)/exp306_sorted.csv > tmp 
	# Run them 10 times
	parallel -j 4 \
			--joblog '$(DATA_PATH)/exp322.log' \
			--nice 19 --delay 2 --colsep ',' --header : \
			'wsls_bandit.py --env_name=DeceptiveBanditOneHigh10-v0 --num_episodes=100 --tie_break='next' --tie_threshold={tie_threshold} --lr_R={lr_R} --save=$(DATA_PATH)/exp322_{index}_{1}.pkl --interactive=False --debug=False --master_seed={1}' ::: {0..10} :::: tmp
	# Clean up
	rm tmp

# ep - use HP from exp307
exp323:
	# Get top 10
	head -n 11 $(DATA_PATH)/exp307_sorted.csv > tmp 
	# Run them 10 times
	parallel -j 4 \
			--joblog '$(DATA_PATH)/exp323.log' \
			--nice 19 --delay 2 --colsep ',' --header : \
			'epsilon_bandit.py --env_name=DeceptiveBanditOneHigh10-v0 --num_episodes=100 --epsilon={epsilon} --lr_R={lr_R} --save=$(DATA_PATH)/exp323_{index}_{1}.pkl --interactive=False --debug=False --master_seed={1}' ::: {0..10} :::: tmp
	# Clean up
	rm tmp

# anneal-ep - use HP from exp308
exp324:
	# Get top 10
	head -n 11 $(DATA_PATH)/exp308_sorted.csv > tmp 
	# Run them 10 times
	parallel -j 4 \
			--joblog '$(DATA_PATH)/exp324.log' \
			--nice 19 --delay 2 --colsep ',' --header : \
			'epsilon_bandit.py --env_name=DeceptiveBanditOneHigh10-v0 --num_episodes=100 --epsilon={epsilon} --epsilon_decay_tau={epsilon_decay_tau} --lr_R={lr_R} --save=$(DATA_PATH)/exp324_{index}_{1}.pkl --interactive=False --debug=False --master_seed={1}' ::: {0..10} :::: tmp
	# Clean up
	rm tmp

# beta - use HP from exp309
exp325:
	# Get top 10
	head -n 11 $(DATA_PATH)/exp309_sorted.csv > tmp 
	# Run them 10 times
	parallel -j 4 \
			--joblog '$(DATA_PATH)/exp325.log' \
			--nice 19 --delay 2 --colsep ',' --header : \
			'softbeta_bandit.py --env_name=DeceptiveBanditOneHigh10-v0 --num_episodes=100 --temp={temp} --beta={beta} --lr_R={lr_R} --save=$(DATA_PATH)/exp325_{index}_{1}.pkl --interactive=False --debug=False --master_seed={1}' ::: {0..10} :::: tmp
	# Clean up
	rm tmp

# ----------------------------------------
# c1315c993d65103f5e62e09f3d73f287a6d897a7
# 7-3-2020
# Curiosity bandits - testing and examples
#
# Deterministic mode
exp326:
	curiosity_bandit.py \
		--env_name='InfoBlueYellow4b-v0' \
        --num_episodes=320 \
		--lr_E=1 \
        --tie_break='next' \
        --tie_threshold=1e-4 \
        --beta=None \
        --master_seed=42 \
        --reward_mode=False \
        --log_dir=$(DATA_PATH)/exp326

# Sampling mode
exp327:
	curiosity_bandit.py \
		--env_name='InfoBlueYellow4b-v0' \
        --num_episodes=320 \
		--lr_E=1 \
        --tie_break='next' \
        --tie_threshold=1e-4 \
        --beta=1000 \
        --master_seed=42 \
        --reward_mode=False \
        --log_dir=$(DATA_PATH)/exp327


# ----------------------------------------
# c1315c993d65103f5e62e09f3d73f287a6d897a7
# 7-3-2020
# Same agent, different random seeds

# seed=124
exp328:
	curiosity_bandit.py \
		--env_name='InfoBlueYellow4b-v0' \
        --num_episodes=40 \
        --tie_break='next' \
        --tie_threshold=1e-5 \
        --beta=None \
        --master_seed=127 \
        --reward_mode=False \
        --log_dir=$(DATA_PATH)/exp328/run1/
	curiosity_bandit.py \
		--env_name='InfoBlueYellow4b-v0' \
        --num_episodes=40 \
        --tie_break='next' \
        --tie_threshold=1e-5 \
        --beta=None \
        --master_seed=23 \
        --reward_mode=False \
        --log_dir=$(DATA_PATH)/exp328/run2/
	curiosity_bandit.py \
		--env_name='InfoBlueYellow4b-v0' \
        --num_episodes=40 \
        --tie_break='next' \
        --tie_threshold=1e-5 \
        --beta=None \
        --master_seed=802 \
        --reward_mode=False \
        --log_dir=$(DATA_PATH)/exp328/run3/
	curiosity_bandit.py \
		--env_name='InfoBlueYellow4b-v0' \
        --num_episodes=40 \
        --tie_break='next' \
        --tie_threshold=1e-5 \
        --beta=None \
        --master_seed=42 \
        --reward_mode=False \
        --log_dir=$(DATA_PATH)/exp328/run4/
	curiosity_bandit.py \
		--env_name='InfoBlueYellow4b-v0' \
        --num_episodes=40 \
        --tie_break='next' \
        --tie_threshold=1e-5 \
        --beta=None \
        --master_seed=673 \
        --reward_mode=False \
        --log_dir=$(DATA_PATH)/exp328/run5/

# ----------------------------------------
# f9c61b4a5722e3d52512cbd079ef00816feac164
# 7-9-2020
# Curiosity bandits - testing and examples
# N = 100 runs

# Deterministic mode
exp329:
	parallel -j 39 \
			--joblog '$(DATA_PATH)/exp329.log' \
			--nice 19 --delay 0 \
		'curiosity_bandit.py --env_name='InfoBlueYellow4b-v0' --num_episodes=320 --lr_E=1 --tie_break='next' --tie_threshold=1e-4 --beta=None --master_seed={1} --reward_mode=False --log_dir=$(DATA_PATH)/exp329/run{1}' ::: {1..100}

# Softmx mode
exp330:
	parallel -j 39 \
			--joblog '$(DATA_PATH)/exp330.log' \
			--nice 19 --delay 0 \
		'curiosity_bandit.py --env_name='InfoBlueYellow4b-v0' --num_episodes=320 --lr_E=1 --tie_break='next' --tie_threshold=1e-4 --beta=1000 --master_seed={1} --reward_mode=False --log_dir=$(DATA_PATH)/exp330/run{1}' ::: {1..100}


# ----------------------------------------
# f9c61b4a5722e3d52512cbd079ef00816feac164
# 7-9-2020
# Curiosity bandits - testing and examples
# N = 100 runs
# One good arm: InfoBlueYellow4a

# Deterministic mode
exp331:
	parallel -j 39 \
			--joblog '$(DATA_PATH)/exp331.log' \
			--nice 19 --delay 0 \
		'curiosity_bandit.py --env_name='InfoBlueYellow4a-v0' --num_episodes=80 --lr_E=1 --tie_break='next' --tie_threshold=1e-5 --beta=None --master_seed={1} --reward_mode=False --log_dir=$(DATA_PATH)/exp331/run{1}' ::: {1..100}

# Softmx mode
exp332:
	parallel -j 39 \
			--joblog '$(DATA_PATH)/exp332.log' \
			--nice 19 --delay 0 \
		'curiosity_bandit.py --env_name='InfoBlueYellow4a-v0' --num_episodes=80 --lr_E=1 --tie_break='next' --tie_threshold=1e-5 --beta=1000 --master_seed={1} --reward_mode=False --log_dir=$(DATA_PATH)/exp332/run{1}' ::: {1..100}

# ----------------------------------------
# d44dd50c538994f2f8d62d5ec90e50903874884f
# 7-9-2020
# Curiosity bandits - testing and examples
# N = 100 runs
# Mixture of arms: InfoBlueYellow4c

# Deterministic mode
exp333:
	parallel -j 39 \
			--joblog '$(DATA_PATH)/exp333.log' \
			--nice 19 --delay 0 \
		'curiosity_bandit.py --env_name='InfoBlueYellow4c-v0' --num_episodes=320 --lr_E=1 --tie_break='next' --tie_threshold=1e-5 --beta=None --master_seed={1} --reward_mode=False --log_dir=$(DATA_PATH)/exp333/run{1}' ::: {1..100}

# Softmx mode
exp334:
	parallel -j 39 \
			--joblog '$(DATA_PATH)/exp334.log' \
			--nice 19 --delay 0 \
		'curiosity_bandit.py --env_name='InfoBlueYellow4c-v0' --num_episodes=320 --lr_E=1 --tie_break='next' --tie_threshold=1e-5 --beta=1000 --master_seed={1} --reward_mode=False --log_dir=$(DATA_PATH)/exp334/run{1}' ::: {1..100}


# TODO: the important of boredom
# TODO: eta sensitivity for reward bandits. -> Supp.
# TODO: info/reward bandits. Dense, dense. Sparse/Dense S+R vectors.
# TODO: port all models to SummaryWriter


# -----------------------------------------------------------------------
# BREAKING CHANGE TO `curiosity_bandit` API. 
# -----------------------------------------------------------------------
# 7-10-2020
# 691fce1
# I needed to introduce even more actor classes. So....
# You must now specify the actor, and its kwargs which go at the end.
#
# Here's a simple example of the new api, and the new actors:
exp335:
	curiosity_bandit.py \
		--env_name='InfoBlueYellow4b-v0' \
		--actor='DeterministicActor' \
		--num_episodes=320 \
		--lr_E=1 \
		--master_seed=42 \
		--reward_mode=False \
		--log_dir=$(DATA_PATH)/exp335/DeterministicActor \
		--tie_break='next' \
		--tie_threshold=1e-4 
	curiosity_bandit.py \
		--env_name='InfoBlueYellow4b-v0' \
		--actor='SoftmaxActor' \
		--num_episodes=320 \
		--lr_E=1 \
		--master_seed=42 \
		--reward_mode=False \
		--log_dir=$(DATA_PATH)/exp335/SoftmaxActor \
		--beta=500 \
		--tie_threshold=1e-4 
	curiosity_bandit.py \
		--env_name='InfoBlueYellow4b-v0' \
		--actor='ThresholdActor' \
		--num_episodes=320 \
		--lr_E=1 \
		--master_seed=42 \
		--reward_mode=False \
		--log_dir=$(DATA_PATH)/exp335/ThresholdActor \
		--tie_threshold=1e-4 
	curiosity_bandit.py \
		--env_name='InfoBlueYellow4b-v0' \
		--actor='RandomActor' \
		--num_episodes=320 \
		--lr_E=1 \
		--master_seed=42 \
		--reward_mode=False \
		--log_dir=$(DATA_PATH)/exp335/RandomActor 

# Repeat 335 
exp336:
	curiosity_bandit.py \
		--env_name='InfoBlueYellow4b-v0' \
		--actor='DeterministicActor' \
		--num_episodes=320 \
		--lr_E=1 \
		--master_seed=693 \
		--reward_mode=False \
		--log_dir=$(DATA_PATH)/exp336/DeterministicActor \
		--tie_break='next' \
		--tie_threshold=1e-4 
	curiosity_bandit.py \
		--env_name='InfoBlueYellow4b-v0' \
		--actor='SoftmaxActor' \
		--num_episodes=320 \
		--lr_E=1 \
		--master_seed=693 \
		--reward_mode=False \
		--log_dir=$(DATA_PATH)/exp336/SoftmaxActor \
		--beta=500 \
		--tie_threshold=1e-4 
	curiosity_bandit.py \
		--env_name='InfoBlueYellow4b-v0' \
		--actor='ThresholdActor' \
		--num_episodes=320 \
		--lr_E=1 \
		--master_seed=693 \
		--reward_mode=False \
		--log_dir=$(DATA_PATH)/exp336/ThresholdActor \
		--tie_threshold=1e-4 
	curiosity_bandit.py \
		--env_name='InfoBlueYellow4b-v0' \
		--actor='RandomActor' \
		--num_episodes=320 \
		--lr_E=1 \
		--master_seed=693 \
		--reward_mode=False \
		--log_dir=$(DATA_PATH)/exp336/RandomActor 

# Repeat 335 again
exp337:
	curiosity_bandit.py \
		--env_name='InfoBlueYellow4b-v0' \
		--actor='DeterministicActor' \
		--num_episodes=320 \
		--lr_E=1 \
		--master_seed=12 \
		--reward_mode=False \
		--log_dir=$(DATA_PATH)/exp337/DeterministicActor \
		--tie_break='next' \
		--tie_threshold=1e-4 
	curiosity_bandit.py \
		--env_name='InfoBlueYellow4b-v0' \
		--actor='SoftmaxActor' \
		--num_episodes=320 \
		--lr_E=1 \
		--master_seed=12 \
		--reward_mode=False \
		--log_dir=$(DATA_PATH)/exp337/SoftmaxActor \
		--beta=500 \
		--tie_threshold=1e-4 
	curiosity_bandit.py \
		--env_name='InfoBlueYellow4b-v0' \
		--actor='ThresholdActor' \
		--num_episodes=320 \
		--lr_E=1 \
		--master_seed=12 \
		--reward_mode=False \
		--log_dir=$(DATA_PATH)/exp337/ThresholdActor \
		--tie_threshold=1e-4 
	curiosity_bandit.py \
		--env_name='InfoBlueYellow4b-v0' \
		--actor='RandomActor' \
		--num_episodes=320 \
		--lr_E=1 \
		--master_seed=12 \
		--reward_mode=False \
		--log_dir=$(DATA_PATH)/exp337/RandomActor 

# Try InfoBlueYellow4a now
exp338:
	curiosity_bandit.py \
		--env_name='InfoBlueYellow4a-v0' \
		--actor='DeterministicActor' \
		--num_episodes=80 \
		--lr_E=1 \
		--master_seed=42 \
		--reward_mode=False \
		--log_dir=$(DATA_PATH)/exp338/DeterministicActor \
		--tie_break='next' \
		--tie_threshold=1e-3 
	curiosity_bandit.py \
		--env_name='InfoBlueYellow4a-v0' \
		--actor='SoftmaxActor' \
		--num_episodes=80 \
		--lr_E=1 \
		--master_seed=42 \
		--reward_mode=False \
		--log_dir=$(DATA_PATH)/exp338/SoftmaxActor \
		--beta=500 \
		--tie_threshold=1e-3 
	curiosity_bandit.py \
		--env_name='InfoBlueYellow4a-v0' \
		--actor='ThresholdActor' \
		--num_episodes=80 \
		--lr_E=1 \
		--master_seed=42 \
		--reward_mode=False \
		--log_dir=$(DATA_PATH)/exp338/ThresholdActor \
		--tie_threshold=1e-3 
	curiosity_bandit.py \
		--env_name='InfoBlueYellow4a-v0' \
		--actor='RandomActor' \
		--num_episodes=80 \
		--lr_E=1 \
		--master_seed=42 \
		--reward_mode=False \
		--log_dir=$(DATA_PATH)/exp338/RandomActor 

# ---------------------------------------------------------------------------
# 7-13-2020
# 5eccfc011c9d528c2483305c27562b2c85180671
#
# All actors can now stop early. Let's test this on InfoBlueYellow4a first.
exp339:
	curiosity_bandit.py \
		--env_name='InfoBlueYellow4a-v0' \
		--actor='DeterministicActor' \
		--num_episodes=80 \
		--lr_E=1 \
		--initial_bins='[1,2]' \
		--initial_count=1 \
		--master_seed=42 \
		--reward_mode=False \
		--log_dir=$(DATA_PATH)/exp339/DeterministicActor \
		--tie_break='next' \
		--tie_threshold=1e-3 
	curiosity_bandit.py \
		--env_name='InfoBlueYellow4a-v0' \
		--actor='SoftmaxActor' \
		--num_episodes=80 \
		--lr_E=1 \
		--initial_bins='[1,2]' \
		--initial_count=1 \
		--master_seed=42 \
		--reward_mode=False \
		--log_dir=$(DATA_PATH)/exp339/SoftmaxActor \
		--beta=500 \
		--tie_threshold=1e-3 
	curiosity_bandit.py \
		--env_name='InfoBlueYellow4a-v0' \
		--actor='ThresholdActor' \
		--num_episodes=80 \
		--lr_E=1 \
		--initial_bins='[1,2]' \
		--initial_count=1 \
		--master_seed=42 \
		--reward_mode=False \
		--log_dir=$(DATA_PATH)/exp339/ThresholdActor \
		--tie_threshold=1e-3 
	curiosity_bandit.py \
		--env_name='InfoBlueYellow4a-v0' \
		--actor='RandomActor' \
		--num_episodes=80 \
		--lr_E=1 \
		--initial_bins='[1,2]' \
		--initial_count=1 \
		--master_seed=42 \
		--reward_mode=False \
		--log_dir=$(DATA_PATH)/exp339/RandomActor \
		--tie_threshold=1e-3 

# Repeat 339, with decreased boredom
exp340:
	curiosity_bandit.py \
		--env_name='InfoBlueYellow4a-v0' \
		--actor='DeterministicActor' \
		--num_episodes=320 \
		--lr_E=1 \
		--initial_bins='[1,2]' \
		--initial_count=1 \
		--master_seed=42 \
		--reward_mode=False \
		--log_dir=$(DATA_PATH)/exp340/DeterministicActor \
		--tie_break='next' \
		--tie_threshold=1e-4 
	curiosity_bandit.py \
		--env_name='InfoBlueYellow4a-v0' \
		--actor='SoftmaxActor' \
		--num_episodes=320 \
		--lr_E=1 \
		--initial_bins='[1,2]' \
		--initial_count=1 \
		--master_seed=42 \
		--reward_mode=False \
		--log_dir=$(DATA_PATH)/exp340/SoftmaxActor \
		--beta=500 \
		--tie_threshold=1e-4 
	curiosity_bandit.py \
		--env_name='InfoBlueYellow4a-v0' \
		--actor='ThresholdActor' \
		--num_episodes=320 \
		--lr_E=1 \
		--initial_bins='[1,2]' \
		--initial_count=1 \
		--master_seed=42 \
		--reward_mode=False \
		--log_dir=$(DATA_PATH)/exp340/ThresholdActor \
		--tie_threshold=1e-4 
	curiosity_bandit.py \
		--env_name='InfoBlueYellow4a-v0' \
		--actor='RandomActor' \
		--num_episodes=320 \
		--lr_E=1 \
		--initial_bins='[1,2]' \
		--initial_count=1 \
		--master_seed=42 \
		--reward_mode=False \
		--log_dir=$(DATA_PATH)/exp340/RandomActor \
		--tie_threshold=1e-4 


# Repeat 340, InfoBlueYellow4b (all max ent arms)
exp341:
	curiosity_bandit.py \
		--env_name='InfoBlueYellow4b-v0' \
		--actor='DeterministicActor' \
		--num_episodes=320 \
		--lr_E=1 \
		--initial_bins='[1,2]' \
		--initial_count=1 \
		--master_seed=42 \
		--reward_mode=False \
		--log_dir=$(DATA_PATH)/exp341/DeterministicActor \
		--tie_break='next' \
		--tie_threshold=1e-4 
	curiosity_bandit.py \
		--env_name='InfoBlueYellow4b-v0' \
		--actor='SoftmaxActor' \
		--num_episodes=320 \
		--lr_E=1 \
		--initial_bins='[1,2]' \
		--initial_count=1 \
		--master_seed=42 \
		--reward_mode=False \
		--log_dir=$(DATA_PATH)/exp341/SoftmaxActor \
		--beta=500 \
		--tie_threshold=1e-4 
	curiosity_bandit.py \
		--env_name='InfoBlueYellow4b-v0' \
		--actor='ThresholdActor' \
		--num_episodes=320 \
		--lr_E=1 \
		--initial_bins='[1,2]' \
		--initial_count=1 \
		--master_seed=42 \
		--reward_mode=False \
		--log_dir=$(DATA_PATH)/exp341/ThresholdActor \
		--tie_threshold=1e-4 
	curiosity_bandit.py \
		--env_name='InfoBlueYellow4b-v0' \
		--actor='RandomActor' \
		--num_episodes=320 \
		--lr_E=1 \
		--initial_bins='[1,2]' \
		--initial_count=1 \
		--master_seed=42 \
		--reward_mode=False \
		--log_dir=$(DATA_PATH)/exp341/RandomActor \
		--tie_threshold=1e-4 

# Repeat 340, but TURN off initial_count
# We offer the memory no intial knowledge.
exp342:
	curiosity_bandit.py \
		--env_name='InfoBlueYellow4a-v0' \
		--actor='DeterministicActor' \
		--num_episodes=320 \
		--lr_E=1 \
		--initial_count=0 \
		--master_seed=42 \
		--reward_mode=False \
		--log_dir=$(DATA_PATH)/exp342/DeterministicActor \
		--tie_break='next' \
		--tie_threshold=1e-4 
	curiosity_bandit.py \
		--env_name='InfoBlueYellow4a-v0' \
		--actor='SoftmaxActor' \
		--num_episodes=320 \
		--lr_E=1 \
		--initial_count=0 \
		--master_seed=42 \
		--reward_mode=False \
		--log_dir=$(DATA_PATH)/exp342/SoftmaxActor \
		--beta=500 \
		--tie_threshold=1e-4 
	curiosity_bandit.py \
		--env_name='InfoBlueYellow4a-v0' \
		--actor='ThresholdActor' \
		--num_episodes=320 \
		--lr_E=1 \
		--initial_count=0 \
		--master_seed=42 \
		--reward_mode=False \
		--log_dir=$(DATA_PATH)/exp342/ThresholdActor \
		--tie_threshold=1e-4 
	curiosity_bandit.py \
		--env_name='InfoBlueYellow4a-v0' \
		--actor='RandomActor' \
		--num_episodes=320 \
		--lr_E=1 \
		--initial_count=0 \
		--master_seed=42 \
		--reward_mode=False \
		--log_dir=$(DATA_PATH)/exp342/RandomActor \
		--tie_threshold=1e-4 

# --------------------------------------------------------------------------
# 7-13-2020
# f36a82e
#
# - Compare exploration when no initial information is available, 
#   compared to when it is. 
# - Try 100 different seeds
# - Use InfoBlueYellow4b-v0 (all arms are max ent)

# --- EXPERIMENT ---
exp343: exp343a exp343b exp343c exp343d exp343e exp343f exp343g exp343h

# --- DeterministicActor ---
exp343a:
	parallel -j 39 \
			--joblog '$(DATA_PATH)/exp343a.log' \
			--nice 19 --delay 0 \
		'curiosity_bandit.py --env_name='InfoBlueYellow4b-v0' --actor='DeterministicActor' --num_episodes=320 --lr_E=1 --initial_count=1 --initial_bins="[1,2]" --master_seed={1} --reward_mode=False --log_dir=$(DATA_PATH)/exp343/DeterministicActor/with/run{1} --tie_break='next' --tie_threshold=1e-4' ::: {1..100}

exp343b:
	parallel -j 39 \
			--joblog '$(DATA_PATH)/exp343a.log' \
			--nice 19 --delay 0 \
		'curiosity_bandit.py --env_name='InfoBlueYellow4b-v0' --actor='DeterministicActor' --num_episodes=320 --lr_E=1 --initial_count=0 --master_seed={1} --reward_mode=False --log_dir=$(DATA_PATH)/exp343/DeterministicActor/without/run{1} --tie_break='next' --tie_threshold=1e-4' ::: {1..100}

# --- SoftmaxActor ---
exp343c:
	parallel -j 39 \
			--joblog '$(DATA_PATH)/exp343c.log' \
			--nice 19 --delay 0 \
		'curiosity_bandit.py --env_name='InfoBlueYellow4b-v0' --actor='SoftmaxActor' --num_episodes=320 --lr_E=1 --initial_count=1 --initial_bins="[1,2]" --master_seed={1} --reward_mode=False --log_dir=$(DATA_PATH)/exp343/SoftmaxActor/with/run{1} --beta=500 --tie_threshold=1e-4' ::: {1..100}

exp343d:
	parallel -j 39 \
			--joblog '$(DATA_PATH)/exp343d.log' \
			--nice 19 --delay 0 \
		'curiosity_bandit.py --env_name='InfoBlueYellow4b-v0' --actor='SoftmaxActor' --num_episodes=320 --lr_E=1 --initial_count=0 --master_seed={1} --reward_mode=False --log_dir=$(DATA_PATH)/exp343/SoftmaxActor/without/run{1} --beta=500 --tie_threshold=1e-4' ::: {1..100}
		
# --- ThresholdActor ---
exp343e:
	parallel -j 39 \
			--joblog '$(DATA_PATH)/exp343e.log' \
			--nice 19 --delay 0 \
		'curiosity_bandit.py --env_name='InfoBlueYellow4b-v0' --actor='ThresholdActor' --num_episodes=320 --lr_E=1 --initial_count=1 --initial_bins="[1,2]" --master_seed={1} --reward_mode=False --log_dir=$(DATA_PATH)/exp343/ThresholdActor/with/run{1} --tie_threshold=1e-4' ::: {1..100}

exp343f:
	parallel -j 39 \
			--joblog '$(DATA_PATH)/exp343f.log' \
			--nice 19 --delay 0 \
		'curiosity_bandit.py --env_name='InfoBlueYellow4b-v0' --actor='ThresholdActor' --num_episodes=320 --lr_E=1 --initial_count=0 --master_seed={1} --reward_mode=False --log_dir=$(DATA_PATH)/exp343/ThresholdActor/without/run{1} --tie_threshold=1e-4' ::: {1..100}

# --- RandomActor ---
exp343g:
	parallel -j 39 \
			--joblog '$(DATA_PATH)/exp343f.log' \
			--nice 19 --delay 0 \
		'curiosity_bandit.py --env_name='InfoBlueYellow4b-v0' --actor='RandomActor' --num_episodes=320 --lr_E=1 --initial_count=1 --initial_bins="[1,2]" --master_seed={1} --reward_mode=False --log_dir=$(DATA_PATH)/exp343/RandomActor/with/run{1} --tie_threshold=1e-4' ::: {1..100}

exp343h:
	parallel -j 39  \
			--joblog '$(DATA_PATH)/exp343g.log' \
			--nice 19 --delay 0 \
		'curiosity_bandit.py --env_name='InfoBlueYellow4b-v0' --actor='RandomActor' --num_episodes=320 --lr_E=1 --initial_count=0 --master_seed={1} --reward_mode=False --log_dir=$(DATA_PATH)/exp343/RandomActor/without/run{1} --tie_threshold=1e-4' ::: {1..100}

# --------------------------------------------------------------------------
# 7-14-2020
# f36a82e
#
# - Compare exploration when no initial information is available, 
#   compared to when it is. 
# - Try 100 different seeds
# - Use InfoBlueYellow4a-v0 (arm 2 is max ent, rest are nearly deterministic)

# --- EXPERIMENT ---
exp344: exp344a exp344b exp344c exp344d exp344e exp344f exp344g exp344h

# --- DeterministicActor ---
exp344a:
	parallel -j 39 \
			--joblog '$(DATA_PATH)/exp344a.log' \
			--nice 19 --delay 0 \
		'curiosity_bandit.py --env_name='InfoBlueYellow4a-v0' --actor='DeterministicActor' --num_episodes=320 --lr_E=1 --initial_count=1 --initial_bins="[1,2]" --master_seed={1} --reward_mode=False --log_dir=$(DATA_PATH)/exp344/DeterministicActor/with/run{1} --tie_break='next' --tie_threshold=1e-4' ::: {1..100}

exp344b:
	parallel -j 39 \
			--joblog '$(DATA_PATH)/exp344a.log' \
			--nice 19 --delay 0 \
		'curiosity_bandit.py --env_name='InfoBlueYellow4a-v0' --actor='DeterministicActor' --num_episodes=320 --lr_E=1 --initial_count=0 --master_seed={1} --reward_mode=False --log_dir=$(DATA_PATH)/exp344/DeterministicActor/without/run{1} --tie_break='next' --tie_threshold=1e-4' ::: {1..100}

# --- SoftmaxActor ---
exp344c:
	parallel -j 39 \
			--joblog '$(DATA_PATH)/exp344c.log' \
			--nice 19 --delay 0 \
		'curiosity_bandit.py --env_name='InfoBlueYellow4a-v0' --actor='SoftmaxActor' --num_episodes=320 --lr_E=1 --initial_count=1 --initial_bins="[1,2]" --master_seed={1} --reward_mode=False --log_dir=$(DATA_PATH)/exp344/SoftmaxActor/with/run{1} --beta=500 --tie_threshold=1e-4' ::: {1..100}

exp344d:
	parallel -j 39 \
			--joblog '$(DATA_PATH)/exp344d.log' \
			--nice 19 --delay 0 \
		'curiosity_bandit.py --env_name='InfoBlueYellow4a-v0' --actor='SoftmaxActor' --num_episodes=320 --lr_E=1 --initial_count=0 --master_seed={1} --reward_mode=False --log_dir=$(DATA_PATH)/exp344/SoftmaxActor/without/run{1} --beta=500 --tie_threshold=1e-4' ::: {1..100}
		
# --- ThresholdActor ---
exp344e:
	parallel -j 39 \
			--joblog '$(DATA_PATH)/exp344e.log' \
			--nice 19 --delay 0 \
		'curiosity_bandit.py --env_name='InfoBlueYellow4a-v0' --actor='ThresholdActor' --num_episodes=320 --lr_E=1 --initial_count=1 --initial_bins="[1,2]" --master_seed={1} --reward_mode=False --log_dir=$(DATA_PATH)/exp344/ThresholdActor/with/run{1} --tie_threshold=1e-4' ::: {1..100}

exp344f:
	parallel -j 39 \
			--joblog '$(DATA_PATH)/exp344f.log' \
			--nice 19 --delay 0 \
		'curiosity_bandit.py --env_name='InfoBlueYellow4a-v0' --actor='ThresholdActor' --num_episodes=320 --lr_E=1 --initial_count=0 --master_seed={1} --reward_mode=False --log_dir=$(DATA_PATH)/exp344/ThresholdActor/without/run{1} --tie_threshold=1e-4' ::: {1..100}

# --- RandomActor ---
exp344g:
	parallel -j 39 \
			--joblog '$(DATA_PATH)/exp344f.log' \
			--nice 19 --delay 0 \
		'curiosity_bandit.py --env_name='InfoBlueYellow4a-v0' --actor='RandomActor' --num_episodes=320 --lr_E=1 --initial_count=1 --initial_bins="[1,2]" --master_seed={1} --reward_mode=False --log_dir=$(DATA_PATH)/exp344/RandomActor/with/run{1} --tie_threshold=1e-4' ::: {1..100}

exp344h:
	parallel -j 39  \
			--joblog '$(DATA_PATH)/exp344g.log' \
			--nice 19 --delay 0 \
		'curiosity_bandit.py --env_name='InfoBlueYellow4a-v0' --actor='RandomActor' --num_episodes=320 --lr_E=1 --initial_count=0 --master_seed={1} --reward_mode=False --log_dir=$(DATA_PATH)/exp344/RandomActor/without/run{1} --tie_threshold=1e-4' ::: {1..100}

# --------------------------------------------------------------------------
# 7-14-2020
# f36a82e
#
# - Compare exploration when no initial information is available, 
#   compared to when it is. 
# - Try 100 different seeds
# - Use InfoBlueYellow4c-v0 (each arm has its own ent, 2 is max ent)

# --- EXPERIMENT ---
exp345: exp345a exp345b exp345c exp345d exp345e exp345f exp345g exp345h

# --- DeterministicActor ---
exp345a:
	parallel -j 39 \
			--joblog '$(DATA_PATH)/exp345a.log' \
			--nice 19 --delay 0 \
		'curiosity_bandit.py --env_name='InfoBlueYellow4c-v0' --actor='DeterministicActor' --num_episodes=320 --lr_E=1 --initial_count=1 --initial_bins="[1,2]" --master_seed={1} --reward_mode=False --log_dir=$(DATA_PATH)/exp345/DeterministicActor/with/run{1} --tie_break='next' --tie_threshold=1e-4' ::: {1..100}

exp345b:
	parallel -j 39 \
			--joblog '$(DATA_PATH)/exp345a.log' \
			--nice 19 --delay 0 \
		'curiosity_bandit.py --env_name='InfoBlueYellow4c-v0' --actor='DeterministicActor' --num_episodes=320 --lr_E=1 --initial_count=0 --master_seed={1} --reward_mode=False --log_dir=$(DATA_PATH)/exp345/DeterministicActor/without/run{1} --tie_break='next' --tie_threshold=1e-4' ::: {1..100}

# --- SoftmaxActor ---
exp345c:
	parallel -j 39 \
			--joblog '$(DATA_PATH)/exp345c.log' \
			--nice 19 --delay 0 \
		'curiosity_bandit.py --env_name='InfoBlueYellow4c-v0' --actor='SoftmaxActor' --num_episodes=320 --lr_E=1 --initial_count=1 --initial_bins="[1,2]" --master_seed={1} --reward_mode=False --log_dir=$(DATA_PATH)/exp345/SoftmaxActor/with/run{1} --beta=500 --tie_threshold=1e-4' ::: {1..100}

exp345d:
	parallel -j 39 \
			--joblog '$(DATA_PATH)/exp345d.log' \
			--nice 19 --delay 0 \
		'curiosity_bandit.py --env_name='InfoBlueYellow4c-v0' --actor='SoftmaxActor' --num_episodes=320 --lr_E=1 --initial_count=0 --master_seed={1} --reward_mode=False --log_dir=$(DATA_PATH)/exp345/SoftmaxActor/without/run{1} --beta=500 --tie_threshold=1e-4' ::: {1..100}

# --- ThresholdActor ---
exp345e:
	parallel -j 39 \
			--joblog '$(DATA_PATH)/exp345e.log' \
			--nice 19 --delay 0 \
		'curiosity_bandit.py --env_name='InfoBlueYellow4c-v0' --actor='ThresholdActor' --num_episodes=320 --lr_E=1 --initial_count=1 --initial_bins="[1,2]" --master_seed={1} --reward_mode=False --log_dir=$(DATA_PATH)/exp345/ThresholdActor/with/run{1} --tie_threshold=1e-4' ::: {1..100}

exp345f:
	parallel -j 39 \
			--joblog '$(DATA_PATH)/exp345f.log' \
			--nice 19 --delay 0 \
		'curiosity_bandit.py --env_name='InfoBlueYellow4c-v0' --actor='ThresholdActor' --num_episodes=320 --lr_E=1 --initial_count=0 --master_seed={1} --reward_mode=False --log_dir=$(DATA_PATH)/exp345/ThresholdActor/without/run{1} --tie_threshold=1e-4' ::: {1..100}

# --- RandomActor ---
exp345g:
	parallel -j 39 \
			--joblog '$(DATA_PATH)/exp345f.log' \
			--nice 19 --delay 0 \
		'curiosity_bandit.py --env_name='InfoBlueYellow4c-v0' --actor='RandomActor' --num_episodes=320 --lr_E=1 --initial_count=1 --initial_bins="[1,2]" --master_seed={1} --reward_mode=False --log_dir=$(DATA_PATH)/exp345/RandomActor/with/run{1} --tie_threshold=1e-4' ::: {1..100}

exp345h:
	parallel -j 39  \
			--joblog '$(DATA_PATH)/exp345g.log' \
			--nice 19 --delay 0 \
		'curiosity_bandit.py --env_name='InfoBlueYellow4c-v0' --actor='RandomActor' --num_episodes=320 --lr_E=1 --initial_count=0 --master_seed={1} --reward_mode=False --log_dir=$(DATA_PATH)/exp345/RandomActor/without/run{1} --tie_threshold=1e-4' ::: {1..100}


# --------------------------------------------------------------------------
# Figure data - det versus sto.
# 7-14-20
# 
# Repeat 340, with increase beta (need to tune this to a 'fair' value).
exp346:
	curiosity_bandit.py \
		--env_name='InfoBlueYellow4a-v0' \
		--actor='DeterministicActor' \
		--num_episodes=320 \
		--lr_E=1 \
		--initial_bins='[1,2]' \
		--initial_count=1 \
		--master_seed=42 \
		--reward_mode=False \
		--log_dir=$(DATA_PATH)/exp346/DeterministicActor \
		--tie_break='next' \
		--tie_threshold=1e-4 
	curiosity_bandit.py \
		--env_name='InfoBlueYellow4a-v0' \
		--actor='SoftmaxActor' \
		--num_episodes=320 \
		--lr_E=1 \
		--initial_bins='[1,2]' \
		--initial_count=1 \
		--master_seed=42 \
		--reward_mode=False \
		--log_dir=$(DATA_PATH)/exp346/SoftmaxActor \
		--beta=10000 \
		--tie_threshold=1e-4 
	curiosity_bandit.py \
		--env_name='InfoBlueYellow4a-v0' \
		--actor='RandomActor' \
		--num_episodes=320 \
		--lr_E=1 \
		--initial_bins='[1,2]' \
		--initial_count=1 \
		--master_seed=42 \
		--reward_mode=False \
		--log_dir=$(DATA_PATH)/exp346/RandomActor \
		--tie_threshold=1e-4 

# ----------------------------------------
# 7-14-2020
# df69237

# Figure data - Det C in random world
exp347:
	curiosity_bandit.py \
		--env_name='InfoBlueYellow4b-v0' \
		--actor='DeterministicActor' \
		--num_episodes=4000 \
		--lr_E=1 \
		--initial_bins='[1,2]' \
		--initial_count=1 \
		--master_seed=127 \
		--reward_mode=False \
		--log_dir=$(DATA_PATH)/exp347/run1/ \
		--tie_break='next' \
		--tie_threshold=1e-4
	curiosity_bandit.py \
		--env_name='InfoBlueYellow4b-v0' \
		--actor='DeterministicActor' \
		--num_episodes=4000 \
		--lr_E=1 \
		--initial_bins='[1,2]' \
		--initial_count=1 \
		--master_seed=23 \
		--reward_mode=False \
		--log_dir=$(DATA_PATH)/exp347/run2/ \
		--tie_break='next' \
		--tie_threshold=1e-4
	curiosity_bandit.py \
		--env_name='InfoBlueYellow4b-v0' \
		--actor='DeterministicActor' \
		--num_episodes=4000 \
		--lr_E=1 \
		--initial_bins='[1,2]' \
		--initial_count=1 \
		--master_seed=802 \
		--reward_mode=False \
		--log_dir=$(DATA_PATH)/exp347/run3/ \
		--tie_break='next' \
		--tie_threshold=1e-4
	curiosity_bandit.py \
		--env_name='InfoBlueYellow4b-v0' \
		--actor='DeterministicActor' \
		--num_episodes=4000 \
		--lr_E=1 \
		--initial_bins='[1,2]' \
		--initial_count=1 \
		--master_seed=42 \
		--reward_mode=False \
		--log_dir=$(DATA_PATH)/exp347/run4/ \
		--tie_break='next' \
		--tie_threshold=1e-4
	curiosity_bandit.py \
		--env_name='InfoBlueYellow4b-v0' \
		--actor='DeterministicActor' \
		--num_episodes=4000 \
		--lr_E=1 \
		--initial_bins='[1,2]' \
		--initial_count=1 \
		--master_seed=673 \
		--reward_mode=False \
		--log_dir=$(DATA_PATH)/exp347/run5/ \
		--tie_break='next' \
		--tie_threshold=1e-4
	curiosity_bandit.py \
		--env_name='InfoBlueYellow4b-v0' \
		--actor='DeterministicActor' \
		--num_episodes=4000 \
		--lr_E=1 \
		--initial_bins='[1,2]' \
		--initial_count=1 \
		--master_seed=592\
		--reward_mode=False \
		--log_dir=$(DATA_PATH)/exp347/run6/ \
		--tie_break='next' \
		--tie_threshold=1e-4


# ----------------------------------------
# 7-14-2020
# df69237

# Figure data - 100 experiments, sto v det
exp348: exp348a exp348b exp348c

exp348a:
	parallel -j 39 \
			--joblog '$(DATA_PATH)/exp348a.log' \
			--nice 19 --delay 0 \
		'curiosity_bandit.py --env_name='InfoBlueYellow4a-v0' --actor='DeterministicActor' --num_episodes=320 --lr_E=1 --initial_count=1 --initial_bins="[1,2]" --master_seed={1} --reward_mode=False --log_dir=$(DATA_PATH)/exp348/DeterministicActor/run{1} --tie_break='next' --tie_threshold=1e-4' ::: {1..100}

exp348b:
	parallel -j 39 \
			--joblog '$(DATA_PATH)/exp348b.log' \
			--nice 19 --delay 0 \
		'curiosity_bandit.py --env_name='InfoBlueYellow4a-v0' --actor='SoftmaxActor' --num_episodes=320 --lr_E=1 --initial_count=1 --initial_bins="[1,2]" --master_seed={1} --reward_mode=False --log_dir=$(DATA_PATH)/exp348/SoftmaxActor/run{1} --beta=10000 --tie_threshold=1e-4' ::: {1..100}

exp348c:
	parallel -j 39 \
			--joblog '$(DATA_PATH)/exp348c.log' \
			--nice 19 --delay 0 \
		'curiosity_bandit.py --env_name='InfoBlueYellow4a-v0' --actor='RandomActor' --num_episodes=320 --lr_E=1 --initial_count=1 --initial_bins="[1,2]" --master_seed={1} --reward_mode=False --log_dir=$(DATA_PATH)/exp348/RandomActor/run{1} --tie_threshold=1e-4' ::: {1..100}

# --------------------------------------------------------------------------
# 7-14-2020
# f1ef2e9
#
# Figure data - explore boredom, over the same task
exp349: exp349a exp349b exp349c

exp349a:
	parallel -j 39 \
			--joblog '$(DATA_PATH)/exp349a.log' \
			--nice 19 --delay 0 \
		'curiosity_bandit.py --env_name='InfoBlueYellow4b-v0' --actor='DeterministicActor' --num_episodes=320 --lr_E=1 --initial_count=1 --initial_bins="[1,2]" --master_seed={2} --reward_mode=False --log_dir=$(DATA_PATH)/exp349/DeterministicActor/threshold{1}/run{2} --tie_break='next' --tie_threshold={1}' ::: 1e-1 1e-2 1e-3 1e-4 1e-5 ::: {1..100}

exp349b:
	parallel -j 39 \
			--joblog '$(DATA_PATH)/exp349b.log' \
			--nice 19 --delay 0 \
		'curiosity_bandit.py --env_name='InfoBlueYellow4b-v0' --actor='SoftmaxActor' --num_episodes=320 --lr_E=1 --initial_count=1 --initial_bins="[1,2]" --master_seed={2} --reward_mode=False --log_dir=$(DATA_PATH)/exp349/SoftmaxActor/threshold{1}/run{2} --beta=500 --tie_threshold={1}' ::: 1e-1 1e-2 1e-3 1e-4 1e-5 ::: {1..100}

exp349c:
	parallel -j 39 \
			--joblog '$(DATA_PATH)/exp349c.log' \
			--nice 19 --delay 0 \
		'curiosity_bandit.py --env_name='InfoBlueYellow4b-v0' --actor='RandomActor' --num_episodes=320 --lr_E=1 --initial_count=1 --initial_bins="[1,2]" --master_seed={2} --reward_mode=False --log_dir=$(DATA_PATH)/exp349/RandomActor/threshold{1}/run{2} --tie_threshold={1}' ::: 1e-1 1e-2 1e-3 1e-4 1e-5 ::: {1..100}


# ----------------------------------------
# 7-14-2020
# df69237

# Figure data - Sto C in random world...

# Tim asked what the softmax version of exp347 looked like,
# which is a good Q
exp350:
	curiosity_bandit.py \
		--env_name='InfoBlueYellow4b-v0' \
		--actor='SoftmaxActor' \
		--num_episodes=4000 \
		--lr_E=1 \
		--initial_bins='[1,2]' \
		--initial_count=1 \
		--master_seed=127 \
		--reward_mode=False \
		--log_dir=$(DATA_PATH)/exp350/run1/ \
		--beta=10000 \
		--tie_threshold=1e-4
	curiosity_bandit.py \
		--env_name='InfoBlueYellow4b-v0' \
		--actor='SoftmaxActor' \
		--num_episodes=4000 \
		--lr_E=1 \
		--initial_bins='[1,2]' \
		--initial_count=1 \
		--master_seed=23 \
		--reward_mode=False \
		--log_dir=$(DATA_PATH)/exp350/run2/ \
		--beta=10000 \
		--tie_threshold=1e-4
	curiosity_bandit.py \
		--env_name='InfoBlueYellow4b-v0' \
		--actor='SoftmaxActor' \
		--num_episodes=4000 \
		--lr_E=1 \
		--initial_bins='[1,2]' \
		--initial_count=1 \
		--master_seed=802 \
		--reward_mode=False \
		--log_dir=$(DATA_PATH)/exp350/run3/ \
		--beta=10000 \
		--tie_threshold=1e-4
	curiosity_bandit.py \
		--env_name='InfoBlueYellow4b-v0' \
		--actor='SoftmaxActor' \
		--num_episodes=4000 \
		--lr_E=1 \
		--initial_bins='[1,2]' \
		--initial_count=1 \
		--master_seed=42 \
		--reward_mode=False \
		--log_dir=$(DATA_PATH)/exp350/run4/ \
		--beta=10000 \
		--tie_threshold=1e-4
	curiosity_bandit.py \
		--env_name='InfoBlueYellow4b-v0' \
		--actor='SoftmaxActor' \
		--num_episodes=4000 \
		--lr_E=1 \
		--initial_bins='[1,2]' \
		--initial_count=1 \
		--master_seed=673 \
		--reward_mode=False \
		--log_dir=$(DATA_PATH)/exp350/run5/ \
		--beta=10000 \
		--tie_threshold=1e-4
	curiosity_bandit.py \
		--env_name='InfoBlueYellow4b-v0' \
		--actor='SoftmaxActor' \
		--num_episodes=4000 \
		--lr_E=1 \
		--initial_bins='[1,2]' \
		--initial_count=1 \
		--master_seed=592\
		--reward_mode=False \
		--log_dir=$(DATA_PATH)/exp350/run6/ \
		--beta=10000 \
		--tie_threshold=1e-4

# exp350, but decrease beta to 500
exp351:
	curiosity_bandit.py \
		--env_name='InfoBlueYellow4b-v0' \
		--actor='SoftmaxActor' \
		--num_episodes=4000 \
		--lr_E=1 \
		--initial_bins='[1,2]' \
		--initial_count=1 \
		--master_seed=127 \
		--reward_mode=False \
		--log_dir=$(DATA_PATH)/exp351/run1/ \
		--beta=500 \
		--tie_threshold=1e-4
	curiosity_bandit.py \
		--env_name='InfoBlueYellow4b-v0' \
		--actor='SoftmaxActor' \
		--num_episodes=4000 \
		--lr_E=1 \
		--initial_bins='[1,2]' \
		--initial_count=1 \
		--master_seed=23 \
		--reward_mode=False \
		--log_dir=$(DATA_PATH)/exp351/run2/ \
		--beta=500 \
		--tie_threshold=1e-4
	curiosity_bandit.py \
		--env_name='InfoBlueYellow4b-v0' \
		--actor='SoftmaxActor' \
		--num_episodes=4000 \
		--lr_E=1 \
		--initial_bins='[1,2]' \
		--initial_count=1 \
		--master_seed=802 \
		--reward_mode=False \
		--log_dir=$(DATA_PATH)/exp351/run3/ \
		--beta=500 \
		--tie_threshold=1e-4
	curiosity_bandit.py \
		--env_name='InfoBlueYellow4b-v0' \
		--actor='SoftmaxActor' \
		--num_episodes=4000 \
		--lr_E=1 \
		--initial_bins='[1,2]' \
		--initial_count=1 \
		--master_seed=42 \
		--reward_mode=False \
		--log_dir=$(DATA_PATH)/exp351/run4/ \
		--beta=500 \
		--tie_threshold=1e-4
	curiosity_bandit.py \
		--env_name='InfoBlueYellow4b-v0' \
		--actor='SoftmaxActor' \
		--num_episodes=4000 \
		--lr_E=1 \
		--initial_bins='[1,2]' \
		--initial_count=1 \
		--master_seed=673 \
		--reward_mode=False \
		--log_dir=$(DATA_PATH)/exp351/run5/ \
		--beta=500 \
		--tie_threshold=1e-4
	curiosity_bandit.py \
		--env_name='InfoBlueYellow4b-v0' \
		--actor='SoftmaxActor' \
		--num_episodes=4000 \
		--lr_E=1 \
		--initial_bins='[1,2]' \
		--initial_count=1 \
		--master_seed=592\
		--reward_mode=False \
		--log_dir=$(DATA_PATH)/exp351/run6/ \
		--beta=500 \
		--tie_threshold=1e-4

# --------------------------------------------------------------------------
# 7-16-2020
# b70bfef
#
# Animals as parameters, v1
#
# Figure data - 100 experiments with the same world but with 100 different
#               parameters.

exp352: exp352a exp352b

exp352a:
	paramsearch.py loguniform $(DATA_PATH)/exp352a.csv --master_seed=42 --num_sample=100 --tie_threshold='(1e-5, 1e-2)'
	parallel -j 40 \
			--verbose \
			--joblog '$(DATA_PATH)/exp352a.log' \
			--nice 19 \
			--delay 0 \
			--skip-first-line \
			--colsep ',' \
		"curiosity_bandit.py --env_name='InfoBlueYellow4a-v0' --actor='DeterministicActor' --num_episodes=320 --lr_E=1 --initial_count=1 --initial_bins='[1,2]' --master_seed=502 --reward_mode=False --log_dir=$(DATA_PATH)/exp352/DeterministicActor/run{1} --tie_break='next' --tie_threshold={2}" :::: $(DATA_PATH)/exp352a.csv

exp352b:
	paramsearch.py loguniform $(DATA_PATH)/exp352b.csv --master_seed=42 --num_sample=100 --tie_threshold='(1e-5, 1e-2)' --beta='(500, 50000)'
	parallel -j 40 \
			--verbose \
			--joblog '$(DATA_PATH)/exp352b.log' \
			--nice 19 \
			--delay 0 \
			--skip-first-line \
			--colsep ',' \
		'curiosity_bandit.py --env_name='InfoBlueYellow4a-v0' --actor='SoftmaxActor' --num_episodes=320 --lr_E=1 --initial_count=1 --initial_bins="[1,2]" --master_seed=502 --reward_mode=False --log_dir=$(DATA_PATH)/exp352/SoftmaxActor/run{1} --beta={2} --tie_threshold={3}' :::: $(DATA_PATH)/exp352b.csv

# --------------------------------------------------------------------------
# 7-17-2020
# 3e37b16
#
# Animals as parameters, v2. 
#
# Figure data - 100 experiments with the same world but with 100 different
#               parameters.
#             - in v2 add a little noise to E_0 (--default_noise=0.05)

exp353: exp353a exp353b

exp353a:
	paramsearch.py loguniform $(DATA_PATH)/exp353a.csv --master_seed=42 --num_sample=100 --tie_threshold='(1e-5, 1e-2)'
	parallel -j 40 \
			--verbose \
			--joblog '$(DATA_PATH)/exp353a.log' \
			--nice 19 \
			--delay 0 \
			--skip-first-line \
			--colsep ',' \
		"curiosity_bandit.py --env_name='InfoBlueYellow4a-v0' --actor='DeterministicActor' --num_episodes=320 --lr_E=1 --initial_count=1 --initial_bins='[1,2]' --initial_noise=0.05 --master_seed=502 --reward_mode=False --log_dir=$(DATA_PATH)/exp353/DeterministicActor/run{1} --tie_break='next' --tie_threshold={2}" :::: $(DATA_PATH)/exp353a.csv

exp353b:
	paramsearch.py loguniform $(DATA_PATH)/exp353b.csv --master_seed=42 --num_sample=100 --tie_threshold='(1e-5, 1e-2)' --beta='(500, 50000)'
	parallel -j 40 \
			--verbose \
			--joblog '$(DATA_PATH)/exp353b.log' \
			--nice 19 \
			--delay 0 \
			--skip-first-line \
			--colsep ',' \
		'curiosity_bandit.py --env_name='InfoBlueYellow4a-v0' --actor='SoftmaxActor' --num_episodes=320 --lr_E=1 --initial_count=1 --initial_bins="[1,2]" --initial_noise=0.05 --master_seed=502 --reward_mode=False --log_dir=$(DATA_PATH)/exp353/SoftmaxActor/run{1} --beta={2} --tie_threshold={3}' :::: $(DATA_PATH)/exp353b.csv


# --------------------------------------------------------------------------
# 7-17-2020
# 6b8fe07 - # !! BIG but backwards compat API CHANGE !!
#
# Animals as parameters, v3. 
#
# Figure data - 100 experiments with the same world but with 100 different
#               parameters.
#             - in v3 add a ONLY little noise to E_0 (--default_noise=0.1)
#             - all other params are fixed
exp354: exp354a exp354b 

exp354a:
	parallel -j 39 \
			--joblog '$(DATA_PATH)/exp354a.log' \
			--nice 19 --delay 0 \
		'curiosity_bandit.py --env_name='InfoBlueYellow4a-v0' --actor='DeterministicActor' --num_episodes=320 --lr_E=1 --initial_count=1 --initial_bins="[1,2]" --initial_noise=0.1 --master_seed=None --env_seed=502 --critic_seed={1} --actor_seed={1} --reward_mode=False --log_dir=$(DATA_PATH)/exp354/DeterministicActor/run{1} --tie_break='next' --tie_threshold=1e-4' ::: {1..100}

exp354b:
	parallel -j 39 \
			--joblog '$(DATA_PATH)/exp354b.log' \
			--nice 19 --delay 0 \
		'curiosity_bandit.py --env_name='InfoBlueYellow4a-v0' --actor='SoftmaxActor' --num_episodes=320 --lr_E=1 --initial_count=1 --initial_bins="[1,2]" --initial_noise=0.1 --env_seed=502 --master_seed=None --critic_seed={1} --actor_seed={1} --reward_mode=False --log_dir=$(DATA_PATH)/exp354/SoftmaxActor/run{1} --beta=10000 --tie_threshold=1e-4' ::: {1..100}


# --------------------------------------------------------------------------
# 7-17-2020
# ddba159
# A test of E0 delta. Not sure things are working right....
# Seeding was setup wong. Fixed. Rerun exp354.
exp355:
	parallel -j 1 \
			--joblog '$(DATA_PATH)/exp355.log' \
			--nice 19 --delay 0 \
		'curiosity_bandit.py --env_name='InfoBlueYellow4a-v0' --actor='DeterministicActor' --num_episodes=320 --lr_E=1 --initial_count=1 --initial_bins="[1,2]" --initial_noise=0.1 --master_seed=None --env_seed=502 --critic_seed={1} --actor_seed={1} --reward_mode=False --log_dir=$(DATA_PATH)/exp355/DeterministicActor/run{1} --tie_break='next' --tie_threshold=1e-4' ::: {1..5}


# --------------------------------------------------------------------------
# 7/20/2020
# fdf65d1
# For consistenct and simpicity all models are getting ported to save
# data w/ SummaryWriter.
#
# Test beta
exp356:
	beta_bandit.py \
		--env_name='BanditOneHigh10-v0' \
		--num_episodes=1000 \
		--lr_R=.1 \
		--master_seed=802 \
		--log_dir=$(DATA_PATH)/exp356/ \
		--beta=1.0 \
		--tie_threshold=1e-4 

# Test meta
exp357:
	wsls_bandit.py \
		--env_name='BanditOneHigh10-v0' \
		--num_episodes=1000 \
		--lr_R=.1 \
		--master_seed=802 \
		--log_dir=$(DATA_PATH)/exp357/ \
		--tie_threshold=1e-4 

# Test ep
exp358:
	epsilon_bandit.py \
		--env_name='BanditOneHigh10-v0' \
		--num_episodes=1000 \
		--lr_R=.1 \
		--epsilon=0.1 \
		--epsilon_decay_tau=0.0001 \
		--master_seed=802 \
		--log_dir=$(DATA_PATH)/exp358/ 

# Test random
exp359:
	random_bandit.py \
		--env_name='BanditOneHigh10-v0' \
		--num_episodes=1000 \
		--lr_R=.1 \
		--master_seed=802 \
		--log_dir=$(DATA_PATH)/exp359/ 


# Test beta
exp360:
	softbeta_bandit.py \
		--env_name='BanditOneHigh10-v0' \
		--num_episodes=1000 \
		--lr_R=.1 \
		--master_seed=802 \
		--log_dir=$(DATA_PATH)/exp360/ \
		--beta=1.0 \
		--temp=100 

# --------------------------------------------------------------------------
# 7/20/2020
# fdf65d1
# Rerun top 10 models, for all bandits. Converting data to SW format.
# ------------------------------------------------------------------------
# 4-20-2020
# 09f45f65e482f0e8ab593cffacc0c58b4daefcf0
# 
# Top 10 hyper-parameters (from last random search exps abour)
# each repeated 10 time, with the different random seed.

# ---------------
# BanditOneHigh10
# ---------------
 
 # meta - use HP from exp294
exp361:
	# Get top 10
	head -n 11 $(DATA_PATH)/exp294_sorted.csv > tmp 
	# Run them 10 times
	parallel -j 4 \
			--joblog '$(DATA_PATH)/exp361.log' \
			--nice 19 --delay 0 --colsep ',' --header : \
			'wsls_bandit.py --env_name=BanditOneHigh10-v0 --num_episodes=100 --tie_break='next' --tie_threshold={tie_threshold} --lr_R={lr_R} --log_dir=$(DATA_PATH)/exp361/param{index}/run{1} --master_seed={1}' ::: {0..10} :::: tmp
	# Clean up
	rm tmp

# ep - use HP from exp295
exp362:
	# Get top 10
	head -n 11 $(DATA_PATH)/exp377_sorted.csv > tmp 
	# Run them 10 times
	parallel -j 4 \
			--joblog '$(DATA_PATH)/exp362.log' \
			--nice 19 --delay 0 --colsep ',' --header : \
			'epsilon_bandit.py --env_name=BanditOneHigh10-v0 --num_episodes=100 --epsilon={epsilon} --lr_R={lr_R} --log_dir=$(DATA_PATH)/exp362/param{index}/run{1} --master_seed={1}' ::: {0..10} :::: tmp
	# Clean up
	rm tmp

# anneal-ep - use HP from exp296
exp363:
	# Get top 10
	head -n 11 $(DATA_PATH)/exp296_sorted.csv > tmp 
	# Run them 10 times
	parallel -j 4 \
			--joblog '$(DATA_PATH)/exp363.log' \
			--nice 19 --delay 0 --colsep ',' --header : \
			'epsilon_bandit.py --env_name=BanditOneHigh10-v0 --num_episodes=100 --epsilon={epsilon} --epsilon_decay_tau={epsilon_decay_tau} --lr_R={lr_R} --log_dir=$(DATA_PATH)/exp363/param{index}/run{1} --master_seed={1}' ::: {0..10} :::: tmp
	# Clean up
	rm tmp

# beta - use HP from exp297
exp364:
	# Get top 10
	head -n 11 $(DATA_PATH)/exp297_sorted.csv > tmp 
	# Run them 10 times
	parallel -j 4 \
			--joblog '$(DATA_PATH)/exp364.log' \
			--nice 19 --delay 0 --colsep ',' --header : \
			'softbeta_bandit.py --env_name=BanditOneHigh10-v0 --num_episodes=100 --beta={beta} --temp={temp} --lr_R={lr_R} --log_dir=$(DATA_PATH)/exp364/param{index}/run{1} --master_seed={1}' ::: {0..10} :::: tmp
	# Clean up
	rm tmp

# ---------------------
# BanditHardAndSparse10
# ---------------------

# meta - use HP from exp298
exp365:
	# Get top 10
	head -n 11 $(DATA_PATH)/exp298_sorted.csv > tmp 
	# Run them 10 times
	parallel -j 4 \
			--joblog '$(DATA_PATH)/exp365.log' \
			--nice 19 --delay 0 --colsep ',' --header : \
			'wsls_bandit.py --env_name=BanditHardAndSparse10-v0 --num_episodes=50000 --tie_break='next' --tie_threshold={tie_threshold} --lr_R={lr_R} --log_dir=$(DATA_PATH)/exp365/param{index}/run{1} --master_seed={1}' ::: {0..10} :::: tmp
	# Clean up
	rm tmp

# ep - use HP from exp299
exp366:
	# Get top 10
	head -n 11 $(DATA_PATH)/exp299_sorted.csv > tmp 
	# Run them 10 times
	parallel -j 4 \
			--joblog '$(DATA_PATH)/exp366.log' \
			--nice 19 --delay 0 --colsep ',' --header : \
			'epsilon_bandit.py --env_name=BanditHardAndSparse10-v0 --num_episodes=50000 --epsilon={epsilon} --lr_R={lr_R} --log_dir=$(DATA_PATH)/exp366/param{index}/run{1} --master_seed={1}' ::: {0..10} :::: tmp
	# Clean up
	rm tmp

# anneal-ep - use HP from exp300
exp367:
	# Get top 10
	head -n 11 $(DATA_PATH)/exp300_sorted.csv > tmp 
	# Run them 10 times
	parallel -j 4 \
			--joblog '$(DATA_PATH)/exp367.log' \
			--nice 19 --delay 0 --colsep ',' --header : \
			'epsilon_bandit.py --env_name=BanditHardAndSparse10-v0 --num_episodes=50000 --epsilon={epsilon} --epsilon_decay_tau={epsilon_decay_tau} --lr_R={lr_R} --log_dir=$(DATA_PATH)/exp367/patam{index}/run{1} --master_seed={1}' ::: {0..10} :::: tmp
	# Clean up
	rm tmp

# beta - use HP from exp301
exp368:
	# Get top 10
	head -n 11 $(DATA_PATH)/exp301_sorted.csv > tmp 
	# Run them 10 times
	parallel -j 4 \
			--joblog '$(DATA_PATH)/exp368.log' \
			--nice 19 --delay 0 --colsep ',' --header : \
			'softbeta_bandit.py --env_name=BanditHardAndSparse10-v0 --num_episodes=50000 --beta={beta} --temp={temp} --lr_R={lr_R} --log_dir=$(DATA_PATH)/exp368/param{index}/run{1} --master_seed={1}' ::: {0..10} :::: tmp
	# Clean up
	rm tmp


# ----------------
# BanditUniform121
# ----------------
# meta - use HP from exp302
exp369:
	# Get top 10
	head -n 11 $(DATA_PATH)/exp302_sorted.csv > tmp 
	# Run them 10 times
	parallel -j 4 \
			--joblog '$(DATA_PATH)/exp369.log' \
			--nice 19 --delay 0 --colsep ',' --header : \
			'wsls_bandit.py --env_name=BanditUniform121-v0 --num_episodes=60500 --tie_break='next' --tie_threshold={tie_threshold} --lr_R={lr_R} --log_dir=$(DATA_PATH)/exp369/param{index}/run{1} --master_seed={1}' ::: {0..10} :::: tmp
	# Clean up
	rm tmp

# ep - use HP from exp303
exp370:
	# Get top 10
	head -n 11 $(DATA_PATH)/exp303_sorted.csv > tmp 
	# Run them 10 times
	parallel -j 4 \
			--joblog '$(DATA_PATH)/exp370.log' \
			--nice 19 --delay 0 --colsep ',' --header : \
			'epsilon_bandit.py --env_name=BanditUniform121-v0 --num_episodes=60500 --epsilon={epsilon} --lr_R={lr_R} --log_dir=$(DATA_PATH)/exp370/param{index}/run{1} --master_seed={1}' ::: {0..10} :::: tmp
	# Clean up
	rm tmp

# ep-anneal - use HP from exp304
exp371:
	# Get top 10
	head -n 11 $(DATA_PATH)/exp304_sorted.csv > tmp 
	# Run them 10 times
	parallel -j 4 \
			--joblog '$(DATA_PATH)/exp371.log' \
			--nice 19 --delay 0 --colsep ',' --header : \
			'epsilon_bandit.py --env_name=BanditUniform121-v0 --num_episodes=60500 --epsilon={epsilon} --epsilon_decay_tau={epsilon_decay_tau} --lr_R={lr_R} --log_dir=$(DATA_PATH)/exp371/param{index}/run{1} --master_seed={1}' ::: {0..10} :::: tmp
	# Clean up
	rm tmp

# beta - use HP from exp305
exp372:
	# Get top 10
	head -n 11 $(DATA_PATH)/exp305_sorted.csv > tmp 
	# Run them 10 times
	parallel -j 4 \
			--joblog '$(DATA_PATH)/exp372.log' \
			--nice 19 --delay 0 --colsep ',' --header : \
			'softbeta_bandit.py --env_name=BanditUniform121-v0 --num_episodes=60500 --temp={temp} --beta={beta} --lr_R={lr_R} --log_dir=$(DATA_PATH)/exp372/param{index}/run{1} --master_seed={1}' ::: {0..10} :::: tmp
	# Clean up
	rm tmp

# ------------------------
# DeceptiveBanditOneHigh10
# ------------------------

# meta - use HP from exp306
exp373:
	# Get top 10
	head -n 11 $(DATA_PATH)/exp306_sorted.csv > tmp 
	# Run them 10 times
	parallel -j 4 \
			--joblog '$(DATA_PATH)/exp373.log' \
			--nice 19 --delay 0 --colsep ',' --header : \
			'wsls_bandit.py --env_name=DeceptiveBanditOneHigh10-v0 --num_episodes=100 --tie_break='next' --tie_threshold={tie_threshold} --lr_R={lr_R} --log_dir=$(DATA_PATH)/exp373/param{index}/run{1} --master_seed={1}' ::: {0..10} :::: tmp
	# Clean up
	rm tmp

# ep - use HP from exp307
exp374:
	# Get top 10
	head -n 11 $(DATA_PATH)/exp307_sorted.csv > tmp 
	# Run them 10 times
	parallel -j 4 \
			--joblog '$(DATA_PATH)/exp374.log' \
			--nice 19 --delay 0 --colsep ',' --header : \
			'epsilon_bandit.py --env_name=DeceptiveBanditOneHigh10-v0 --num_episodes=100 --epsilon={epsilon} --lr_R={lr_R} --log_dir=$(DATA_PATH)/exp374/param{index}/run{1} --master_seed={1}' ::: {0..10} :::: tmp
	# Clean up
	rm tmp

# anneal-ep - use HP from exp308
exp375:
	# Get top 10
	head -n 11 $(DATA_PATH)/exp308_sorted.csv > tmp 
	# Run them 10 times
	parallel -j 4 \
			--joblog '$(DATA_PATH)/exp375.log' \
			--nice 19 --delay 0 --colsep ',' --header : \
			'epsilon_bandit.py --env_name=DeceptiveBanditOneHigh10-v0 --num_episodes=100 --epsilon={epsilon} --epsilon_decay_tau={epsilon_decay_tau} --lr_R={lr_R} --log_dir=$(DATA_PATH)/exp375/param{index}/run{1} --master_seed={1}' ::: {0..10} :::: tmp
	# Clean up
	rm tmp

# beta - use HP from exp309
exp376:
	# Get top 10
	head -n 11 $(DATA_PATH)/exp309_sorted.csv > tmp 
	# Run them 10 times
	parallel -j 4 \
			--joblog '$(DATA_PATH)/exp376.log' \
			--nice 19 --delay 0 --colsep ',' --header : \
			'softbeta_bandit.py --env_name=DeceptiveBanditOneHigh10-v0 --num_episodes=100 --temp={temp} --beta={beta} --lr_R={lr_R} --log_dir=$(DATA_PATH)/exp376/param{index}/run{1} --master_seed={1}' ::: {0..10} :::: tmp
	# Clean up
	rm tmp

# ---------------------------------------------------------------------------
# 7-21-2020 
# e3dcf4f -- !! API CHANGE !!
#
# Fix bandit tuning. Several sub-experiments.
#
# - The tune_bandit.py was running 1 repeat / parameter. This lead to 
# significant random seed effects in the final results. 
# - All bandit results need to be re-tuned. 
# - Below is a first test for BanditOneHigh10 env

# --- BanditOneHigh10 ---
# ep
exp377:
	tune_bandit.py random $(DATA_PATH)/exp377 \
		--exp_name='epsilon_bandit' \
		--env_name=BanditOneHigh10-v0 \
		--num_samples=1000 \
		--num_episodes=100 \
		--num_repeats=50 \
		--num_processes=39 \
		--log_space=True \
		--metric="total_R" \
		--epsilon='(0.01, 0.99)' \
		--lr_R='(0.001, 0.5)' 

# ep-decay
exp378:
	tune_bandit.py random $(DATA_PATH)/exp378 \
		--exp_name='epsilon_bandit' \
		--env_name=BanditOneHigh10-v0 \
		--num_samples=1000 \
		--num_episodes=100 \
		--num_repeats=50 \
		--num_processes=39 \
		--log_space=True \
		--metric="total_R" \
		--epsilon='(0.01, 0.99)' \
		--epsilon_decay_tau='(0.0001, 0.1)' \
		--lr_R='(0.001, 0.5)' 

# eta/dual value
exp379:
	tune_bandit.py random $(DATA_PATH)/exp379 \
		--exp_name='wsls_bandit' \
		--env_name=BanditOneHigh10-v0 \
		--num_samples=1000 \
		--num_episodes=100 \
		--num_repeats=50 \
		--num_processes=39 \
		--log_space=True \
		--metric="total_R" \
		--tie_threshold='(1e-9, 1e-2)' \
		--lr_R='(0.001, 0.5)' 

# beta
exp380:
	tune_bandit.py random $(DATA_PATH)/exp380 \
		--exp_name='softbeta_bandit' \
		--env_name=BanditOneHigh10-v0 \
		--num_samples=1000 \
		--num_episodes=100 \
		--num_repeats=50 \
		--num_processes=39 \
		--log_space=True \
		--metric="total_R" \
		--beta='(0.001, 10)' \
		--lr_R='(0.001, 0.5)' \
		--temp='(0.1, 3)'

# --- BanditOneHigh121 ---
# ep
exp381:
	tune_bandit.py random $(DATA_PATH)/exp381 \
		--exp_name='epsilon_bandit' \
		--env_name=BanditOneHigh121-v0 \
		--num_samples=1000 \
		--num_episodes=12100 \
		--num_repeats=50 \
		--num_processes=39 \
		--log_space=True \
		--metric="total_R" \
		--epsilon='(0.01, 0.99)' \
		--lr_R='(0.001, 0.5)' 

# ep-decay
exp382:
	tune_bandit.py random $(DATA_PATH)/exp382 \
		--exp_name='epsilon_bandit' \
		--env_name=BanditOneHigh121-v0 \
		--num_samples=1000 \
		--num_episodes=12100 \
		--num_repeats=50 \
		--num_processes=39 \
		--log_space=True \
		--metric="total_R" \
		--epsilon='(0.01, 0.99)' \
		--epsilon_decay_tau='(0.0001, 0.1)' \
		--lr_R='(0.001, 0.5)' 

# eta/dual value
exp383:
	tune_bandit.py random $(DATA_PATH)/exp383 \
		--exp_name='wsls_bandit' \
		--env_name=BanditOneHigh121-v0 \
		--num_samples=1000 \
		--num_episodes=12100 \
		--num_repeats=50 \
		--num_processes=39 \
		--log_space=True \
		--metric="total_R" \
		--tie_threshold='(1e-9, 1e-2)' \
		--lr_R='(0.001, 0.5)' 

# beta
exp384:
	tune_bandit.py random $(DATA_PATH)/exp384 \
		--exp_name='softbeta_bandit' \
		--env_name=BanditOneHigh121-v0 \
		--num_samples=1000 \
		--num_episodes=12100 \
		--num_repeats=50 \
		--num_processes=39 \
		--log_space=True \
		--metric="total_R" \
		--beta='(0.001, 10)' \
		--lr_R='(0.001, 0.5)' \
		--temp='(0.0001, 1000)'


# --- BanditHardAndSparse10 ---
# ep
exp385:
	tune_bandit.py random $(DATA_PATH)/exp385 \
		--exp_name='epsilon_bandit' \
		--env_name=BanditHardAndSparse10-v0 \
		--num_samples=1000 \
		--num_episodes=10000 \
		--num_repeats=50 \
		--num_processes=39 \
		--log_space=True \
		--metric="total_R" \
		--epsilon='(0.01, 0.99)' \
		--lr_R='(0.001, 0.5)' 

# ep-decay
exp386:
	tune_bandit.py random $(DATA_PATH)/exp386 \
		--exp_name='epsilon_bandit' \
		--env_name=BanditHardAndSparse10-v0 \
		--num_samples=1000 \
		--num_episodes=10000 \
		--num_repeats=50 \
		--num_processes=39 \
		--log_space=True \
		--metric="total_R" \
		--epsilon='(0.01, 0.99)' \
		--epsilon_decay_tau='(0.0001, 0.1)' \
		--lr_R='(0.001, 0.5)' 

# eta/dual value
exp387:
	tune_bandit.py random $(DATA_PATH)/exp387 \
		--exp_name='wsls_bandit' \
		--env_name=BanditHardAndSparse10-v0 \
		--num_samples=1000 \
		--num_episodes=10000 \
		--num_repeats=50 \
		--num_processes=39 \
		--log_space=True \
		--metric="total_R" \
		--tie_threshold='(1e-9, 1e-2)' \
		--lr_R='(0.001, 0.5)' 

# beta
exp388:
	tune_bandit.py random $(DATA_PATH)/exp388 \
		--exp_name='softbeta_bandit' \
		--env_name=BanditHardAndSparse10-v0 \
		--num_samples=1000 \
		--num_episodes=10000 \
		--num_repeats=50 \
		--num_processes=39 \
		--log_space=True \
		--metric="total_R" \
		--beta='(0.001, 10)' \
		--lr_R='(0.001, 0.5)' \
		--temp='(0.0001, 1000)'

# --DeceptiveBanditOneHigh10 --
# ep
exp389:
	tune_bandit.py random $(DATA_PATH)/exp389 \
		--exp_name='epsilon_bandit' \
		--env_name=DeceptiveBanditOneHigh10-v0 \
		--num_samples=1000 \
		--num_episodes=100 \
		--num_repeats=50 \
		--num_processes=39 \
		--log_space=True \
		--metric="total_R" \
		--epsilon='(0.01, 0.99)' \
		--lr_R='(0.001, 0.5)' 

# ep-decay
exp390:
	tune_bandit.py random $(DATA_PATH)/exp390 \
		--exp_name='epsilon_bandit' \
		--env_name=DeceptiveBanditOneHigh10-v0 \
		--num_samples=1000 \
		--num_episodes=100 \
		--num_repeats=50 \
		--num_processes=39 \
		--log_space=True \
		--metric="total_R" \
		--epsilon='(0.01, 0.99)' \
		--epsilon_decay_tau='(0.0001, 0.1)' \
		--lr_R='(0.001, 0.5)' 

# eta/dual value
exp391:
	tune_bandit.py random $(DATA_PATH)/exp391 \
		--exp_name='wsls_bandit' \
		--env_name=DeceptiveBanditOneHigh10-v0 \
		--num_samples=1000 \
		--num_episodes=100 \
		--num_repeats=50 \
		--num_processes=39 \
		--log_space=True \
		--metric="total_R" \
		--tie_threshold='(1e-9, 1e-2)' \
		--lr_R='(0.001, 0.5)' 

# beta
exp392:
	tune_bandit.py random $(DATA_PATH)/exp392 \
		--exp_name='softbeta_bandit' \
		--env_name=DeceptiveBanditOneHigh10-v0 \
		--num_samples=1000 \
		--num_episodes=100 \
		--num_repeats=50 \
		--num_processes=39 \
		--log_space=True \
		--metric="total_R" \
		--beta='(0.001, 10)' \
		--lr_R='(0.001, 0.5)' \
		--temp='(0.0001, 1000)'


# --------------------------------------------------------------------------
# 7-23-2020
# c87210d

# Test runs for tune exp377-90. 
exp393_exp397: exp393 exp394 exp395 exp396 exp397 

# ---------------
# BanditOneHigh10
# ---------------
 
 # meta - use exp379_sorted
exp393:
	# Get top 10
	head -n 11 $(DATA_PATH)/exp379_sorted.csv > tmp 
	# Run them 10 times
	parallel -j 39 \
			--joblog '$(DATA_PATH)/exp393.log' \
			--nice 19 --delay 0 --bar --colsep ',' --header : \
			'wsls_bandit.py --env_name=BanditOneHigh10-v0 --num_episodes=200 --tie_break='next' --tie_threshold={tie_threshold} --lr_R={lr_R} --log_dir=$(DATA_PATH)/exp393/param{index}/run{1} --master_seed={1}' ::: {0..10} :::: tmp
	# Clean up
	rm tmp

# ep - use exp377_sorted
exp394:
	# Get top 10
	head -n 11 $(DATA_PATH)/exp377_sorted.csv > tmp 
	# Run them 10 times
	parallel -j 39 \
			--joblog '$(DATA_PATH)/exp394.log' \
			--nice 19 --delay 0 --bar --colsep ',' --header : \
			'epsilon_bandit.py --env_name=BanditOneHigh10-v0 --num_episodes=200 --epsilon={epsilon} --lr_R={lr_R} --log_dir=$(DATA_PATH)/exp394/param{index}/run{1} --master_seed={1}' ::: {0..10} :::: tmp
	# Clean up
	rm tmp

# anneal-ep - use exp378_sorted
exp395:
	# Get top 10
	head -n 11 $(DATA_PATH)/exp378_sorted.csv > tmp 
	# Run them 10 times
	parallel -j 39 \
			--joblog '$(DATA_PATH)/exp395.log' \
			--nice 19 --delay 0 --bar --colsep ',' --header : \
			'epsilon_bandit.py --env_name=BanditOneHigh10-v0 --num_episodes=200 --epsilon={epsilon} --epsilon_decay_tau={epsilon_decay_tau} --lr_R={lr_R} --log_dir=$(DATA_PATH)/exp395/param{index}/run{1} --master_seed={1}' ::: {0..10} :::: tmp
	# Clean up
	rm tmp

# beta - use exp380_sorted
exp396:
	# Get top 10
	head -n 11 $(DATA_PATH)/exp380_sorted.csv > tmp 
	# Run them 10 times
	parallel -j 39 \
			--joblog '$(DATA_PATH)/exp396.log' \
			--nice 19 --delay 0 --bar --colsep ',' --header : \
			'softbeta_bandit.py --env_name=BanditOneHigh10-v0 --num_episodes=200 --beta={beta} --temp={temp} --lr_R={lr_R} --log_dir=$(DATA_PATH)/exp396/param{index}/run{1} --master_seed={1}' ::: {0..10} :::: tmp
	# Clean up
	rm tmp

# random
exp397:
	parallel -j 39 \
			--joblog '$(DATA_PATH)/exp397.log' \
			--nice 19 --delay 0 --bar --colsep ',' \
			'random_bandit.py --env_name=BanditOneHigh10-v0 --num_episodes=200  --lr_R=0.1 --log_dir=$(DATA_PATH)/exp397/param0/run{1} --master_seed={1}' ::: {1..100}

# ---------------------------------------------------------------------------
# 7-23-2020
# 
# BanditOneHigh4 -- for a cartoon/example figure. 
# - It is the same size as the InfoBandit.
# - Recycle the HPs for BanditOneHigh10; close enough.

exp398_402: exp398 exp399 exp400 exp401 exp402 

exp398_402_clean:
	-rm -rf $(DATA_PATH)/exp398/*
	-rm -rf $(DATA_PATH)/exp399/*
	-rm -rf $(DATA_PATH)/exp400/*
	-rm -rf $(DATA_PATH)/exp401/*
	-rm -rf $(DATA_PATH)/exp402/*
	
# meta - use exp379_sorted
exp398:
	# Get top 10
	head -n 11 $(DATA_PATH)/exp379_sorted.csv > tmp 
	# Run them 10 times
	parallel -j 39 \
			--joblog '$(DATA_PATH)/exp398.log' \
			--nice 19 --delay 0 --bar --colsep ',' --header : \
			'wsls_bandit.py --env_name=BanditOneHigh4-v0 --num_episodes=320 --tie_break='next' --tie_threshold={tie_threshold} --lr_R={lr_R} --log_dir=$(DATA_PATH)/exp398/param{index}/run{1} --master_seed={1}' ::: {0..10} :::: tmp
	# Clean up
	rm tmp

# ep - use exp377_sorted
exp399:
	# Get top 10
	head -n 11 $(DATA_PATH)/exp377_sorted.csv > tmp 
	# Run them 10 times
	parallel -j 39 \
			--joblog '$(DATA_PATH)/exp399.log' \
			--nice 19 --delay 0 --bar --colsep ',' --header : \
			'epsilon_bandit.py --env_name=BanditOneHigh4-v0 --num_episodes=320 --epsilon={epsilon} --lr_R={lr_R} --log_dir=$(DATA_PATH)/exp399/param{index}/run{1} --master_seed={1}' ::: {0..10} :::: tmp
	# Clean up
	rm tmp

# anneal-ep - use exp378_sorted
exp400:
	# Get top 10
	head -n 11 $(DATA_PATH)/exp378_sorted.csv > tmp 
	# Run them 10 times
	parallel -j 39 \
			--joblog '$(DATA_PATH)/exp400.log' \
			--nice 19 --delay 0 --bar --colsep ',' --header : \
			'epsilon_bandit.py --env_name=BanditOneHigh4-v0 --num_episodes=320 --epsilon={epsilon} --epsilon_decay_tau={epsilon_decay_tau} --lr_R={lr_R} --log_dir=$(DATA_PATH)/exp400/param{index}/run{1} --master_seed={1}' ::: {0..10} :::: tmp
	# Clean up
	rm tmp

# beta - use exp380_sorted
exp401:
	# Get top 10
	head -n 11 $(DATA_PATH)/exp380_sorted.csv > tmp 
	# Run them 10 times
	parallel -j 39 \
			--joblog '$(DATA_PATH)/exp401.log' \
			--nice 19 --delay 0 --bar --colsep ',' --header : \
			'softbeta_bandit.py --env_name=BanditOneHigh4-v0 --num_episodes=320 --beta={beta} --temp={temp} --lr_R={lr_R} --log_dir=$(DATA_PATH)/exp401/param{index}/run{1} --master_seed={1}' ::: {0..10} :::: tmp
	# Clean up
	rm tmp

# random
exp402:
	parallel -j 39 \
			--joblog '$(DATA_PATH)/exp402.log' \
			--nice 19 --delay 0 --bar --colsep ',' \
			'random_bandit.py --env_name=BanditOneHigh4-v0 --num_episodes=320  --lr_R=0.1 --log_dir=$(DATA_PATH)/exp402/param0/run{1} --master_seed={1}' ::: {1..100}


# ---------------------------------------------------------------------------
# 7-23-2020
# feaf715
#
# BanditOneHigh121 -High-d exploration, made simple.

exp403_407: exp403 exp404 exp405 exp405 exp406 exp407

exp403_407_clean:
	-rm -rf $(DATA_PATH)/exp403/*
	-rm -rf $(DATA_PATH)/exp404/*
	-rm -rf $(DATA_PATH)/exp405/*
	-rm -rf $(DATA_PATH)/exp406/*
	-rm -rf $(DATA_PATH)/exp407/*
	
# meta - use exp379_sorted
exp403:
	# Get top 10
	head -n 11 $(DATA_PATH)/exp383_sorted.csv > tmp 
	# Run them 10 times
	parallel -j 39 \
			--joblog '$(DATA_PATH)/exp403.log' \
			--nice 19 --delay 0 --bar --colsep ',' --header : \
			'wsls_bandit.py --env_name=BanditOneHigh121-v0 --num_episodes=60500 --tie_break='next' --tie_threshold={tie_threshold} --lr_R={lr_R} --log_dir=$(DATA_PATH)/exp403/param{index}/run{1} --master_seed={1}' ::: {0..10} :::: tmp
	# Clean up
	rm tmp

# ep - use exp377_sorted
exp404:
	# Get top 10
	head -n 11 $(DATA_PATH)/exp381_sorted.csv > tmp 
	# Run them 10 times
	parallel -j 39 \
			--joblog '$(DATA_PATH)/exp404.log' \
			--nice 19 --delay 0 --bar --colsep ',' --header : \
			'epsilon_bandit.py --env_name=BanditOneHigh121-v0 --num_episodes=60500 --epsilon={epsilon} --lr_R={lr_R} --log_dir=$(DATA_PATH)/exp404/param{index}/run{1} --master_seed={1}' ::: {0..10} :::: tmp
	# Clean up
	rm tmp

# anneal-ep - use exp378_sorted
exp405:
	# Get top 10
	head -n 11 $(DATA_PATH)/exp382_sorted.csv > tmp 
	# Run them 10 times
	parallel -j 39 \
			--joblog '$(DATA_PATH)/exp405.log' \
			--nice 19 --delay 0 --bar --colsep ',' --header : \
			'epsilon_bandit.py --env_name=BanditOneHigh121-v0 --num_episodes=60500 --epsilon={epsilon} --epsilon_decay_tau={epsilon_decay_tau} --lr_R={lr_R} --log_dir=$(DATA_PATH)/exp405/param{index}/run{1} --master_seed={1}' ::: {0..10} :::: tmp
	# Clean up
	rm tmp

# beta - use exp380_sorted
exp406:
	# Get top 10
	head -n 11 $(DATA_PATH)/exp384_sorted.csv > tmp 
	# Run them 10 times
	parallel -j 39 \
			--joblog '$(DATA_PATH)/exp406.log' \
			--nice 19 --delay 0 --bar --colsep ',' --header : \
			'softbeta_bandit.py --env_name=BanditOneHigh121-v0 --num_episodes=60500 --beta={beta} --temp={temp} --lr_R={lr_R} --log_dir=$(DATA_PATH)/exp406/param{index}/run{1} --master_seed={1}' ::: {0..10} :::: tmp
	# Clean up
	rm tmp

# random
exp407:
	parallel -j 39 \
			--joblog '$(DATA_PATH)/exp407.log' \
			--nice 19 --delay 0 --bar --colsep ',' \
			'random_bandit.py --env_name=BanditOneHigh121-v0 --num_episodes=60500  --lr_R=0.1 --log_dir=$(DATA_PATH)/exp407/param0/run{1} --master_seed={1}' ::: {1..100}


# ---------------------------------------------------------------------------
# 7-23-2020
# feaf715
#
# BanditHardAndSparse10 - sparse rewards

exp408_412: exp408 exp409 exp410 exp411 exp412

exp408_412_clean:
	-rm -rf $(DATA_PATH)/exp408/*
	-rm -rf $(DATA_PATH)/exp409/*
	-rm -rf $(DATA_PATH)/exp410/*
	-rm -rf $(DATA_PATH)/exp411/*
	-rm -rf $(DATA_PATH)/exp412/*
	
# meta 
exp408:
	# Get top 10
	head -n 11 $(DATA_PATH)/exp387_sorted.csv > tmp 
	# Run them 10 times
	parallel -j 39 \
			--joblog '$(DATA_PATH)/exp408.log' \
			--nice 19 --delay 0 --bar --colsep ',' --header : \
			'wsls_bandit.py --env_name=BanditHardAndSparse10-v0 --num_episodes=50000 --tie_break='next' --tie_threshold={tie_threshold} --lr_R={lr_R} --log_dir=$(DATA_PATH)/exp408/param{index}/run{1} --master_seed={1}' ::: {0..10} :::: tmp
	# Clean up
	rm tmp

# ep 
exp409:
	# Get top 10
	head -n 11 $(DATA_PATH)/exp385_sorted.csv > tmp 
	# Run them 10 times
	parallel -j 39 \
			--joblog '$(DATA_PATH)/exp409.log' \
			--nice 19 --delay 0 --bar --colsep ',' --header : \
			'epsilon_bandit.py --env_name=BanditHardAndSparse10-v0 --num_episodes=50000 --epsilon={epsilon} --lr_R={lr_R} --log_dir=$(DATA_PATH)/exp409/param{index}/run{1} --master_seed={1}' ::: {0..10} :::: tmp
	# Clean up
	rm tmp

# anneal-ep 
exp410:
	# Get top 10
	head -n 11 $(DATA_PATH)/exp386_sorted.csv > tmp 
	# Run them 10 times
	parallel -j 39 \
			--joblog '$(DATA_PATH)/exp410.log' \
			--nice 19 --delay 0 --bar --colsep ',' --header : \
			'epsilon_bandit.py --env_name=BanditHardAndSparse10-v0 --num_episodes=50000 --epsilon={epsilon} --epsilon_decay_tau={epsilon_decay_tau} --lr_R={lr_R} --log_dir=$(DATA_PATH)/exp410/param{index}/run{1} --master_seed={1}' ::: {0..10} :::: tmp
	# Clean up
	rm tmp

# beta 
exp411:
	# Get top 10
	head -n 11 $(DATA_PATH)/exp388_sorted.csv > tmp 
	# Run them 10 times
	parallel -j 39 \
			--joblog '$(DATA_PATH)/exp411.log' \
			--nice 19 --delay 0 --bar --colsep ',' --header : \
			'softbeta_bandit.py --env_name=BanditHardAndSparse10-v0 --num_episodes=50000 --beta={beta} --temp={temp} --lr_R={lr_R} --log_dir=$(DATA_PATH)/exp411/param{index}/run{1} --master_seed={1}' ::: {0..10} :::: tmp
	# Clean up
	rm tmp

# random
exp412:
	parallel -j 39 \
			--joblog '$(DATA_PATH)/exp412.log' \
			--nice 19 --delay 0 --bar --colsep ',' \
			'random_bandit.py --env_name=BanditHardAndSparse10-v0 --num_episodes=50000  --lr_R=0.1 --log_dir=$(DATA_PATH)/exp412/param0/run{1} --master_seed={1}' ::: {1..100}



# ---------------------------------------------------------------------------
# 7-23-2020
# feaf715
#
# DeceptiveBanditOneHigh10 - deceptive rewards

exp413_417: exp413 exp414 exp415 exp416 exp417

exp413_417_clean:
	-rm -rf $(DATA_PATH)/exp413/*
	-rm -rf $(DATA_PATH)/exp414/*
	-rm -rf $(DATA_PATH)/exp415/*
	-rm -rf $(DATA_PATH)/exp416/*
	-rm -rf $(DATA_PATH)/exp417/*
	
# meta 
exp413:
	# Get top 10
	head -n 11 $(DATA_PATH)/exp391_sorted.csv > tmp 
	# Run them 10 times
	parallel -j 39 \
			--joblog '$(DATA_PATH)/exp413.log' \
			--nice 19 --delay 0 --bar --colsep ',' --header : \
			'wsls_bandit.py --env_name=DeceptiveBanditOneHigh10-v0 --num_episodes=100 --tie_break='next' --tie_threshold={tie_threshold} --lr_R={lr_R} --log_dir=$(DATA_PATH)/exp413/param{index}/run{1} --master_seed={1}' ::: {0..10} :::: tmp
	# Clean up
	rm tmp

# ep 
exp414:
	# Get top 10
	head -n 11 $(DATA_PATH)/exp389_sorted.csv > tmp 
	# Run them 10 times
	parallel -j 39 \
			--joblog '$(DATA_PATH)/exp414.log' \
			--nice 19 --delay 0 --bar --colsep ',' --header : \
			'epsilon_bandit.py --env_name=DeceptiveBanditOneHigh10-v0 --num_episodes=100 --epsilon={epsilon} --lr_R={lr_R} --log_dir=$(DATA_PATH)/exp414/param{index}/run{1} --master_seed={1}' ::: {0..10} :::: tmp
	# Clean up
	rm tmp

# anneal-ep 
exp415:
	# Get top 10
	head -n 11 $(DATA_PATH)/exp390_sorted.csv > tmp 
	# Run them 10 times
	parallel -j 39 \
			--joblog '$(DATA_PATH)/exp415.log' \
			--nice 19 --delay 0 --bar --colsep ',' --header : \
			'epsilon_bandit.py --env_name=DeceptiveBanditOneHigh10-v0 --num_episodes=100 --epsilon={epsilon} --epsilon_decay_tau={epsilon_decay_tau} --lr_R={lr_R} --log_dir=$(DATA_PATH)/exp415/param{index}/run{1} --master_seed={1}' ::: {0..10} :::: tmp
	# Clean up
	rm tmp

# beta 
exp416:
	# Get top 10
	head -n 11 $(DATA_PATH)/exp392_sorted.csv > tmp 
	# Run them 10 times
	parallel -j 39 \
			--joblog '$(DATA_PATH)/exp416.log' \
			--nice 19 --delay 0 --bar --colsep ',' --header : \
			'softbeta_bandit.py --env_name=DeceptiveBanditOneHigh10-v0 --num_episodes=100 --beta={beta} --temp={temp} --lr_R={lr_R} --log_dir=$(DATA_PATH)/exp416/param{index}/run{1} --master_seed={1}' ::: {0..10} :::: tmp
	# Clean up
	rm tmp

# random
exp417:
	parallel -j 39 \
			--joblog '$(DATA_PATH)/exp417.log' \
			--nice 19 --delay 0 --bar --colsep ',' \
			'random_bandit.py --env_name=DeceptiveBanditOneHigh10-v0 --num_episodes=100  --lr_R=0.1 --log_dir=$(DATA_PATH)/exp417/param0/run{1} --master_seed={1}' ::: {1..100}


# ---------------------------------------------------------------------------
# 7-28-2020
# 
# Tune for DeceptiveBanditOneHigh10, then run test for the top10
# 
# The previous tuning run (exp389-exp392) had far too many trials making the 
# the task easy. The env should have prevented that, but the safety
# was commented out. 
#
# RESULT: - turns out 30 is **unlearnable**. 
#         - ....Which is no doubt why its was commented out.

exp418_exp421: exp418 exp419 exp420 exp421

exp418:
	tune_bandit.py random $(DATA_PATH)/exp418 \
		--exp_name='epsilon_bandit' \
		--env_name=DeceptiveBanditOneHigh10-v0 \
		--num_samples=1000 \
		--num_episodes=30 \
		--num_repeats=50 \
		--num_processes=39 \
		--log_space=True \
		--metric="total_R" \
		--epsilon='(0.01, 0.99)' \
		--lr_R='(0.001, 0.5)' 

# ep-decay
exp419:
	tune_bandit.py random $(DATA_PATH)/exp419 \
		--exp_name='epsilon_bandit' \
		--env_name=DeceptiveBanditOneHigh10-v0 \
		--num_samples=1000 \
		--num_episodes=30 \
		--num_repeats=50 \
		--num_processes=39 \
		--log_space=True \
		--metric="total_R" \
		--epsilon='(0.01, 0.99)' \
		--epsilon_decay_tau='(0.0001, 0.1)' \
		--lr_R='(0.001, 0.5)' 

# eta/dual value
exp420:
	tune_bandit.py random $(DATA_PATH)/exp420 \
		--exp_name='wsls_bandit' \
		--env_name=DeceptiveBanditOneHigh10-v0 \
		--num_samples=1000 \
		--num_episodes=30 \
		--num_repeats=50 \
		--num_processes=39 \
		--log_space=True \
		--metric="total_R" \
		--tie_threshold='(1e-9, 1e-2)' \
		--lr_R='(0.001, 0.5)' 

# beta
exp421:
	tune_bandit.py random $(DATA_PATH)/exp421 \
		--exp_name='softbeta_bandit' \
		--env_name=DeceptiveBanditOneHigh10-v0 \
		--num_samples=1000 \
		--num_episodes=30 \
		--num_repeats=50 \
		--num_processes=39 \
		--log_space=True \
		--metric="total_R" \
		--beta='(0.001, 10)' \
		--lr_R='(0.001, 0.5)' \
		--temp='(0.1, 3)'

 
# --- Run tuned top10 for DeceptiveBanditOneHigh10 ---
exp422_exp426: exp422 exp423 exp424 exp425 exp426

exp422_exp426_clean:
	-rm -rf $(DATA_PATH)/exp422/*
	-rm -rf $(DATA_PATH)/exp423/*
	-rm -rf $(DATA_PATH)/exp424/*
	-rm -rf $(DATA_PATH)/exp425/*
	-rm -rf $(DATA_PATH)/exp426/*
	
# meta 
exp422:
	# Get top 10
	head -n 11 $(DATA_PATH)/exp420_sorted.csv > tmp 
	# Run them 10 times
	parallel -j 39 \
			--joblog '$(DATA_PATH)/exp422.log' \
			--nice 19 --delay 0 --bar --colsep ',' --header : \
			'wsls_bandit.py --env_name=DeceptiveBanditOneHigh10-v0 --num_episodes=30 --tie_break='next' --tie_threshold={tie_threshold} --lr_R={lr_R} --log_dir=$(DATA_PATH)/exp422/param{index}/run{1} --master_seed={1}' ::: {0..10} :::: tmp
	# Clean up
	rm tmp

# ep 
exp423:
	# Get top 10
	head -n 11 $(DATA_PATH)/exp418_sorted.csv > tmp 
	# Run them 10 times
	parallel -j 39 \
			--joblog '$(DATA_PATH)/exp423.log' \
			--nice 19 --delay 0 --bar --colsep ',' --header : \
			'epsilon_bandit.py --env_name=DeceptiveBanditOneHigh10-v0 --num_episodes=30 --epsilon={epsilon} --lr_R={lr_R} --log_dir=$(DATA_PATH)/exp423/param{index}/run{1} --master_seed={1}' ::: {0..10} :::: tmp
	# Clean up
	rm tmp

# anneal-ep 
exp424:
	# Get top 10
	head -n 11 $(DATA_PATH)/exp419_sorted.csv > tmp 
	# Run them 10 times
	parallel -j 39 \
			--joblog '$(DATA_PATH)/exp424.log' \
			--nice 19 --delay 0 --bar --colsep ',' --header : \
			'epsilon_bandit.py --env_name=DeceptiveBanditOneHigh10-v0 --num_episodes=30 --epsilon={epsilon} --epsilon_decay_tau={epsilon_decay_tau} --lr_R={lr_R} --log_dir=$(DATA_PATH)/exp424/param{index}/run{1} --master_seed={1}' ::: {0..10} :::: tmp
	# Clean up
	rm tmp

# beta 
exp425:
	# Get top 10
	head -n 11 $(DATA_PATH)/exp421_sorted.csv > tmp 
	# Run them 10 times
	parallel -j 39 \
			--joblog '$(DATA_PATH)/exp425.log' \
			--nice 19 --delay 0 --bar --colsep ',' --header : \
			'softbeta_bandit.py --env_name=DeceptiveBanditOneHigh10-v0 --num_episodes=30 --beta={beta} --temp={temp} --lr_R={lr_R} --log_dir=$(DATA_PATH)/exp425/param{index}/run{1} --master_seed={1}' ::: {0..10} :::: tmp
	# Clean up
	rm tmp

# random
exp426:
	parallel -j 39 \
			--joblog '$(DATA_PATH)/exp426.log' \
			--nice 19 --delay 0 --bar --colsep ',' \
			'random_bandit.py --env_name=DeceptiveBanditOneHigh10-v0 --num_episodes=30  --lr_R=0.1 --log_dir=$(DATA_PATH)/exp426/param0/run{1} --master_seed={1}' ::: {1..100}

# --------------------------------------------------------------------------
# 7-28-2020
# 49118c4
#
# tie_threshold for meta was perhaps at the bottom of the range in 
# BanditHardAndSparse10. 
# - Rerun that tune and test, expanding the range
# eta/dual value
#
# RESULT: Comparing tables between exp427 and exp387 with a smaller therhsold 
#         range show identical performance. exp387 may even be better. No
#         futher analysis of this line is needed.

exp427:
	tune_bandit.py random $(DATA_PATH)/exp427 \
		--exp_name='wsls_bandit' \
		--env_name=BanditHardAndSparse10-v0 \
		--num_samples=1000 \
		--num_episodes=10000 \
		--num_repeats=50 \
		--num_processes=39 \
		--log_space=True \
		--metric="total_R" \
		--tie_threshold='(1e-11, 1e-2)' \
		--lr_R='(0.001, 0.5)' 

# meta 
exp428:
	# Get top 10
	head -n 11 $(DATA_PATH)/exp427_sorted.csv > tmp 
	# Run them 10 times
	parallel -j 39 \
			--joblog '$(DATA_PATH)/exp428.log' \
			--nice 19 --delay 0 --bar --colsep ',' --header : \
			'wsls_bandit.py --env_name=BanditHardAndSparse10-v0 --num_episodes=20000 --tie_break='next' --tie_threshold={tie_threshold} --lr_R={lr_R} --log_dir=$(DATA_PATH)/exp428/param{index}/run{1} --master_seed={1}' ::: {0..10} :::: tmp
	# Clean up
	rm tmp


# ----------------------------------------------------------------------------
# 7-29-2020
# d736a50
#
# The advatage dual value had over other algs was diminished some when I moved
# to what should be robust HP tuning. As part of the new HP tuning I adjusted
# down the number of episodes to make room for more repeats. Could the change 
# performance I see at test be due to undersampling. As time grows so to does
# the advatage of my model? 
#
# - Rerun exp408_412 (BanditHardAndSparse10) with more epsiodes
# - Increase num_episodes=40000 from num_episodes=20000
#
# RESULT: - eta does do better as num_episodes increases. This hold marginally
#         for ep-decay which should can a very similiar fixed exploitation 
#         policy very late in learning.
#         - Not sure whay is best and fair to show readers.
#         - Try increasing episodes for all the others (sans Deception)

exp429_433: exp429 exp430 exp431 exp432 exp433
	
# meta 
exp429:
	# Get top 10
	head -n 11 $(DATA_PATH)/exp387_sorted.csv > tmp 
	# Run them 10 times
	parallel -j 39 \
			--joblog '$(DATA_PATH)/exp429.log' \
			--nice 19 --delay 0 --bar --colsep ',' --header : \
			'wsls_bandit.py --env_name=BanditHardAndSparse10-v0 --num_episodes=40000 --tie_break='next' --tie_threshold={tie_threshold} --lr_R={lr_R} --log_dir=$(DATA_PATH)/exp429/param{index}/run{1} --master_seed={1}' ::: {0..10} :::: tmp
	# Clean up
	rm tmp

# ep 
exp430:
	# Get top 10
	head -n 11 $(DATA_PATH)/exp385_sorted.csv > tmp 
	# Run them 10 times
	parallel -j 39 \
			--joblog '$(DATA_PATH)/exp430.log' \
			--nice 19 --delay 0 --bar --colsep ',' --header : \
			'epsilon_bandit.py --env_name=BanditHardAndSparse10-v0 --num_episodes=40000 --epsilon={epsilon} --lr_R={lr_R} --log_dir=$(DATA_PATH)/exp430/param{index}/run{1} --master_seed={1}' ::: {0..10} :::: tmp
	# Clean up
	rm tmp

# anneal-ep 
exp431:
	# Get top 10
	head -n 11 $(DATA_PATH)/exp386_sorted.csv > tmp 
	# Run them 10 times
	parallel -j 39 \
			--joblog '$(DATA_PATH)/exp431.log' \
			--nice 19 --delay 0 --bar --colsep ',' --header : \
			'epsilon_bandit.py --env_name=BanditHardAndSparse10-v0 --num_episodes=40000 --epsilon={epsilon} --epsilon_decay_tau={epsilon_decay_tau} --lr_R={lr_R} --log_dir=$(DATA_PATH)/exp431/param{index}/run{1} --master_seed={1}' ::: {0..10} :::: tmp
	# Clean up
	rm tmp

# beta 
exp432:
	# Get top 10
	head -n 11 $(DATA_PATH)/exp388_sorted.csv > tmp 
	# Run them 10 times
	parallel -j 39 \
			--joblog '$(DATA_PATH)/exp432.log' \
			--nice 19 --delay 0 --bar --colsep ',' --header : \
			'softbeta_bandit.py --env_name=BanditHardAndSparse10-v0 --num_episodes=40000 --beta={beta} --temp={temp} --lr_R={lr_R} --log_dir=$(DATA_PATH)/exp432/param{index}/run{1} --master_seed={1}' ::: {0..10} :::: tmp
	# Clean up
	rm tmp

# random
exp433:
	parallel -j 39 \
			--joblog '$(DATA_PATH)/exp433.log' \
			--nice 19 --delay 0 --bar --colsep ',' \
			'random_bandit.py --env_name=BanditHardAndSparse10-v0 --num_episodes=40000  --lr_R=0.1 --log_dir=$(DATA_PATH)/exp433/param0/run{1} --master_seed={1}' ::: {1..100}


# --------------------------------------------------------------------------
# 7-29-2020
# d736a50
#
# - Rerun exp393_397 (BanditOneHigh10) with more epsiodes
# - Increase num_episodes=400 from num_episodes=200
#
exp434_438: exp434 exp435 exp436 exp437 exp438

# meta - use exp379_sorted
exp434:
	# Get top 10
	head -n 11 $(DATA_PATH)/exp379_sorted.csv > tmp 
	# Run them 10 times
	parallel -j 39 \
			--joblog '$(DATA_PATH)/exp434.log' \
			--nice 19 --delay 0 --bar --colsep ',' --header : \
			'wsls_bandit.py --env_name=BanditOneHigh10-v0 --num_episodes=400 --tie_break='next' --tie_threshold={tie_threshold} --lr_R={lr_R} --log_dir=$(DATA_PATH)/exp434/param{index}/run{1} --master_seed={1}' ::: {0..10} :::: tmp
	# Clean up
	rm tmp

# ep - use exp377_sorted
exp435:
	# Get top 10
	head -n 11 $(DATA_PATH)/exp377_sorted.csv > tmp 
	# Run them 10 times
	parallel -j 39 \
			--joblog '$(DATA_PATH)/exp435.log' \
			--nice 19 --delay 0 --bar --colsep ',' --header : \
			'epsilon_bandit.py --env_name=BanditOneHigh10-v0 --num_episodes=400 --epsilon={epsilon} --lr_R={lr_R} --log_dir=$(DATA_PATH)/exp435/param{index}/run{1} --master_seed={1}' ::: {0..10} :::: tmp
	# Clean up
	rm tmp

# anneal-ep - use exp378_sorted
exp436:
	# Get top 10
	head -n 11 $(DATA_PATH)/exp378_sorted.csv > tmp 
	# Run them 10 times
	parallel -j 39 \
			--joblog '$(DATA_PATH)/exp436.log' \
			--nice 19 --delay 0 --bar --colsep ',' --header : \
			'epsilon_bandit.py --env_name=BanditOneHigh10-v0 --num_episodes=400 --epsilon={epsilon} --epsilon_decay_tau={epsilon_decay_tau} --lr_R={lr_R} --log_dir=$(DATA_PATH)/exp436/param{index}/run{1} --master_seed={1}' ::: {0..10} :::: tmp
	# Clean up
	rm tmp

# beta - use exp380_sorted
exp437:
	# Get top 10
	head -n 11 $(DATA_PATH)/exp380_sorted.csv > tmp 
	# Run them 10 times
	parallel -j 39 \
			--joblog '$(DATA_PATH)/exp437.log' \
			--nice 19 --delay 0 --bar --colsep ',' --header : \
			'softbeta_bandit.py --env_name=BanditOneHigh10-v0 --num_episodes=400 --beta={beta} --temp={temp} --lr_R={lr_R} --log_dir=$(DATA_PATH)/exp437/param{index}/run{1} --master_seed={1}' ::: {0..10} :::: tmp
	# Clean up
	rm tmp

# random
exp438:
	parallel -j 39 \
			--joblog '$(DATA_PATH)/exp438.log' \
			--nice 19 --delay 0 --bar --colsep ',' \
			'random_bandit.py --env_name=BanditOneHigh10-v0 --num_episodes=400  --lr_R=0.1 --log_dir=$(DATA_PATH)/exp438/param0/run{1} --master_seed={1}' ::: {1..100}

# --------------------------------------------------------------------------
# 7-29-2020
# d736a50
#
# - Rerun exp403_407 (BanditOneHigh121) with more epsiodes
# - Increase num_episodes=400 from num_episodes=200

exp439_443: exp439 exp440 exp441 exp442 exp443 
	
# meta - use exp379_sorted
exp439:
	# Get top 10
	head -n 11 $(DATA_PATH)/exp383_sorted.csv > tmp 
	# Run them 10 times
	parallel -j 39 \
			--joblog '$(DATA_PATH)/exp439.log' \
			--nice 19 --delay 0 --bar --colsep ',' --header : \
			'wsls_bandit.py --env_name=BanditOneHigh121-v0 --num_episodes=60500 --tie_break='next' --tie_threshold={tie_threshold} --lr_R={lr_R} --log_dir=$(DATA_PATH)/exp439/param{index}/run{1} --master_seed={1}' ::: {0..10} :::: tmp
	# Clean up
	rm tmp

# ep - use exp377_sorted
exp440:
	# Get top 10
	head -n 11 $(DATA_PATH)/exp381_sorted.csv > tmp 
	# Run them 10 times
	parallel -j 39 \
			--joblog '$(DATA_PATH)/exp440.log' \
			--nice 19 --delay 0 --bar --colsep ',' --header : \
			'epsilon_bandit.py --env_name=BanditOneHigh121-v0 --num_episodes=48400 --epsilon={epsilon} --lr_R={lr_R} --log_dir=$(DATA_PATH)/exp440/param{index}/run{1} --master_seed={1}' ::: {0..10} :::: tmp
	# Clean up
	rm tmp

# anneal-ep - use exp378_sorted
exp441:
	# Get top 10
	head -n 11 $(DATA_PATH)/exp382_sorted.csv > tmp 
	# Run them 10 times
	parallel -j 39 \
			--joblog '$(DATA_PATH)/exp441.log' \
			--nice 19 --delay 0 --bar --colsep ',' --header : \
			'epsilon_bandit.py --env_name=BanditOneHigh121-v0 --num_episodes=48400 --epsilon={epsilon} --epsilon_decay_tau={epsilon_decay_tau} --lr_R={lr_R} --log_dir=$(DATA_PATH)/exp441/param{index}/run{1} --master_seed={1}' ::: {0..10} :::: tmp
	# Clean up
	rm tmp

# beta - use exp380_sorted
exp442:
	# Get top 10
	head -n 11 $(DATA_PATH)/exp384_sorted.csv > tmp 
	# Run them 10 times
	parallel -j 39 \
			--joblog '$(DATA_PATH)/exp442.log' \
			--nice 19 --delay 0 --bar --colsep ',' --header : \
			'softbeta_bandit.py --env_name=BanditOneHigh121-v0 --num_episodes=48400 --beta={beta} --temp={temp} --lr_R={lr_R} --log_dir=$(DATA_PATH)/exp442/param{index}/run{1} --master_seed={1}' ::: {0..10} :::: tmp
	# Clean up
	rm tmp

# random
exp443:
	parallel -j 39 \
			--joblog '$(DATA_PATH)/exp443.log' \
			--nice 19 --delay 0 --bar --colsep ',' \
			'random_bandit.py --env_name=BanditOneHigh121-v0 --num_episodes=48400  --lr_R=0.1 --log_dir=$(DATA_PATH)/exp443/param0/run{1} --master_seed={1}' ::: {1..100}


# ---------------------------------------------------------------------------
# 7-29-2020
# 6ab8e01
# 
# More DeceptiveBanditOneHigh10 tests:
# - Run deception bandits tuned for 100 episodes, on 50 episodes. 
# - Does our advantage lessen or widen. 
#
# RESULT: - Only ours is finding the best arm, but low trial numbers make 
#         reward the lowest as well. Showing the progression here is tricky.
#         - Try some average timecourse plots. Also....
#         - Run it out a bit past 100. Check that out. 

exp444_448: exp444 exp445 exp446 exp447 exp448
	
# meta 
exp444:
	# Get top 10
	head -n 11 $(DATA_PATH)/exp391_sorted.csv > tmp 
	# Run them 10 times
	parallel -j 39 \
			--joblog '$(DATA_PATH)/exp444.log' \
			--nice 19 --delay 0 --bar --colsep ',' --header : \
			'wsls_bandit.py --env_name=DeceptiveBanditOneHigh10-v0 --num_episodes=50 --tie_break='next' --tie_threshold={tie_threshold} --lr_R={lr_R} --log_dir=$(DATA_PATH)/exp444/param{index}/run{1} --master_seed={1}' ::: {0..10} :::: tmp
	# Clean up
	rm tmp

# ep 
exp445:
	# Get top 10
	head -n 11 $(DATA_PATH)/exp389_sorted.csv > tmp 
	# Run them 10 times
	parallel -j 39 \
			--joblog '$(DATA_PATH)/exp445.log' \
			--nice 19 --delay 0 --bar --colsep ',' --header : \
			'epsilon_bandit.py --env_name=DeceptiveBanditOneHigh10-v0 --num_episodes=50 --epsilon={epsilon} --lr_R={lr_R} --log_dir=$(DATA_PATH)/exp445/param{index}/run{1} --master_seed={1}' ::: {0..10} :::: tmp
	# Clean up
	rm tmp

# anneal-ep 
exp446:
	# Get top 10
	head -n 11 $(DATA_PATH)/exp390_sorted.csv > tmp 
	# Run them 10 times
	parallel -j 39 \
			--joblog '$(DATA_PATH)/exp446.log' \
			--nice 19 --delay 0 --bar --colsep ',' --header : \
			'epsilon_bandit.py --env_name=DeceptiveBanditOneHigh10-v0 --num_episodes=50 --epsilon={epsilon} --epsilon_decay_tau={epsilon_decay_tau} --lr_R={lr_R} --log_dir=$(DATA_PATH)/exp446/param{index}/run{1} --master_seed={1}' ::: {0..10} :::: tmp
	# Clean up
	rm tmp

# beta 
exp447:
	# Get top 10
	head -n 11 $(DATA_PATH)/exp392_sorted.csv > tmp 
	# Run them 10 times
	parallel -j 39 \
			--joblog '$(DATA_PATH)/exp447.log' \
			--nice 19 --delay 0 --bar --colsep ',' --header : \
			'softbeta_bandit.py --env_name=DeceptiveBanditOneHigh10-v0 --num_episodes=50 --beta={beta} --temp={temp} --lr_R={lr_R} --log_dir=$(DATA_PATH)/exp447/param{index}/run{1} --master_seed={1}' ::: {0..10} :::: tmp
	# Clean up
	rm tmp

# random
exp448:
	parallel -j 39 \
			--joblog '$(DATA_PATH)/exp448.log' \
			--nice 19 --delay 0 --bar --colsep ',' \
			'random_bandit.py --env_name=DeceptiveBanditOneHigh10-v0 --num_episodes=50  --lr_R=0.1 --log_dir=$(DATA_PATH)/exp448/param0/run{1} --master_seed={1}' ::: {1..100}

# ---------------------------------------------------------------------------
# 7-29-2020
# 6ab8e01
# 
# More DeceptiveBanditOneHigh10 tests:
# - Run deception bandits tuned for 100 episodes, on 200 episodes. 
# - Does our advantage lessen or widen. 
#
# RESULT: 200 looks like a fair num

exp449_453: exp449 exp450 exp451 exp452 exp453

# meta 
exp449:
	# Get top 10
	head -n 11 $(DATA_PATH)/exp391_sorted.csv > tmp 
	# Run them 10 times
	parallel -j 39 \
			--joblog '$(DATA_PATH)/exp449.log' \
			--nice 19 --delay 0 --bar --colsep ',' --header : \
			'wsls_bandit.py --env_name=DeceptiveBanditOneHigh10-v0 --num_episodes=200 --tie_break='next' --tie_threshold={tie_threshold} --lr_R={lr_R} --log_dir=$(DATA_PATH)/exp449/param{index}/run{1} --master_seed={1}' ::: {0..10} :::: tmp
	# Clean up
	rm tmp

# ep 
exp450:
	# Get top 10
	head -n 11 $(DATA_PATH)/exp389_sorted.csv > tmp 
	# Run them 10 times
	parallel -j 39 \
			--joblog '$(DATA_PATH)/exp450.log' \
			--nice 19 --delay 0 --bar --colsep ',' --header : \
			'epsilon_bandit.py --env_name=DeceptiveBanditOneHigh10-v0 --num_episodes=200 --epsilon={epsilon} --lr_R={lr_R} --log_dir=$(DATA_PATH)/exp450/param{index}/run{1} --master_seed={1}' ::: {0..10} :::: tmp
	# Clean up
	rm tmp

# anneal-ep 
exp451:
	# Get top 10
	head -n 11 $(DATA_PATH)/exp390_sorted.csv > tmp 
	# Run them 10 times
	parallel -j 39 \
			--joblog '$(DATA_PATH)/exp451.log' \
			--nice 19 --delay 0 --bar --colsep ',' --header : \
			'epsilon_bandit.py --env_name=DeceptiveBanditOneHigh10-v0 --num_episodes=200 --epsilon={epsilon} --epsilon_decay_tau={epsilon_decay_tau} --lr_R={lr_R} --log_dir=$(DATA_PATH)/exp451/param{index}/run{1} --master_seed={1}' ::: {0..10} :::: tmp
	# Clean up
	rm tmp

# beta 
exp452:
	# Get top 10
	head -n 11 $(DATA_PATH)/exp392_sorted.csv > tmp 
	# Run them 10 times
	parallel -j 39 \
			--joblog '$(DATA_PATH)/exp452.log' \
			--nice 19 --delay 0 --bar --colsep ',' --header : \
			'softbeta_bandit.py --env_name=DeceptiveBanditOneHigh10-v0 --num_episodes=200 --beta={beta} --temp={temp} --lr_R={lr_R} --log_dir=$(DATA_PATH)/exp452/param{index}/run{1} --master_seed={1}' ::: {0..10} :::: tmp
	# Clean up
	rm tmp

# random
exp453:
	parallel -j 39 \
			--joblog '$(DATA_PATH)/exp453.log' \
			--nice 19 --delay 0 --bar --colsep ',' \
			'random_bandit.py --env_name=DeceptiveBanditOneHigh10-v0 --num_episodes=200  --lr_R=0.1 --log_dir=$(DATA_PATH)/exp453/param0/run{1} --master_seed={1}' ::: {1..100}


# ----------------------------------------------------------------------------
# 7-31-2020
# b976f10
#
# Tune **softmeta** (first time) and run one some tests.
# Also tune meta, for comparison. Past runs with
# - BanditOneHigh4
# used params stolen from
# - BanditOneHigh10
# which should work but might not be ideal.

softmeta_test:
	-rm -rf $(DATA_PATH)/test # clean up
	softwsls_bandit.py --env_name=BanditOneHigh4-v0 --num_episodes=80 --tie_threshold=1e-3 --temp=1 --lr_R=0.1 --log_dir=$(DATA_PATH)/test/ --master_seed=42

exp454_exp457: exp454 exp455 exp456 exp457 

exp454_exp457_clean: 
	-rm -rf $(DATA_PATH)/exp454 
	-rm -rf $(DATA_PATH)/exp455
	-rm -rf $(DATA_PATH)/exp456
	-rm -rf $(DATA_PATH)/exp457 

# -- Tune ---
exp454:
	tune_bandit.py random $(DATA_PATH)/exp454 \
		--exp_name='softwsls_bandit' \
		--env_name=BanditOneHigh4-v0 \
		--num_samples=1000 \
		--num_episodes=40 \
		--num_repeats=50 \
		--num_processes=4 \
		--log_space=True \
		--metric="total_R" \
		--tie_threshold='(1e-9, 1e-2)' \
		--temp='(0.00001, 10000)' \
		--lr_R='(0.001, 0.5)' 

exp455:
	tune_bandit.py random $(DATA_PATH)/exp455 \
		--exp_name='wsls_bandit' \
		--env_name=BanditOneHigh4-v0 \
		--num_samples=1000 \
		--num_episodes=40 \
		--num_repeats=50 \
		--num_processes=4 \
		--log_space=True \
		--metric="total_R" \
		--tie_threshold='(1e-9, 1e-2)' \
		--lr_R='(0.001, 0.5)' 

# --- Test ---
exp456:
	# Get top 10
	head -n 11 $(DATA_PATH)/exp454_sorted.csv > tmp 
	# Run them 10 times
	parallel -j 4 \
			--joblog '$(DATA_PATH)/exp456.log' \
			--nice 19 --delay 0 --bar --colsep ',' --header : \
			'softwsls_bandit.py --env_name=BanditOneHigh4-v0 --num_episodes=2000 --temp={temp} --tie_threshold={tie_threshold} --lr_R={lr_R} --log_dir=$(DATA_PATH)/exp456/param{index}/run{1} --master_seed={1}' ::: {0..10} :::: tmp
	# Clean up
	rm tmp

exp457:
	# Get top 10
	head -n 11 $(DATA_PATH)/exp455_sorted.csv > tmp 
	# Run them 10 times
	parallel -j 4 \
			--joblog '$(DATA_PATH)/exp457.log' \
			--nice 19 --delay 0 --bar --colsep ',' --header : \
			'wsls_bandit.py --env_name=BanditOneHigh4-v0 --num_episodes=2000 --tie_break='next' --tie_threshold={tie_threshold} --lr_R={lr_R} --log_dir=$(DATA_PATH)/exp457/param{index}/run{1} --master_seed={1}' ::: {0..10} :::: tmp
	# Clean up
	rm tmp

# --------------------------------------------------------------------------
# 7-31-2020
# Test runs for basic function -- DistractionBanditOneHigh10
# 
# I defined DistractionBanditOneHigh10 which is the same as BanditOneHigh10, except
# it also returns states/stim that match the InfoBandits. That state/stim
# in no way (statistically) predict rewards. They are a distraction. 
#
# - This recipe is just for debugging. I had to change ALL the agent_bandit
#   implementations to handle this kinds of task

test_distraction:
	-rm -rf $(DATA_PATH)/test
	wsls_bandit.py --env_name=DistractionBanditOneHigh10-v0 --num_episodes=100 --tie_break='next' --tie_threshold=0.001 --lr_R=0.01 --log_dir=$(DATA_PATH)/test/wsls_bandit
	softwsls_bandit.py --env_name=DistractionBanditOneHigh10-v0 --num_episodes=100 --temp=.1 --tie_threshold=0.001 --lr_R=0.01 --log_dir=$(DATA_PATH)/test/softwsls_bandit
	softbeta_bandit.py --env_name=DistractionBanditOneHigh10-v0 --num_episodes=100 --beta=1 --temp=1 --lr_R=0.01 --log_dir=$(DATA_PATH)/test/beta_bandit
	epsilon_bandit.py --env_name=DistractionBanditOneHigh10-v0 --num_episodes=100 --epsilon=0.5 --epsilon_decay_tau=000.1 --lr_R=0.01 --log_dir=$(DATA_PATH)/test/epsilon_bandit
	random_bandit.py --env_name=DistractionBanditOneHigh10-v0 --num_episodes=100 --lr_R=0.01 --log_dir=$(DATA_PATH)/test/random_bandit

# --------------------------------------------------------------------------
# 7-31-202
# df0338f
#
# --- Tune **DistractionBanditOneHigh10** (first attempt) ---
#
# RESULTS: - To my frank surprise both meta and beta can be tuned to out
#          perform the reward only models (ep) even in the face of a 
#          high/max entropy distraction signal.
#          - That is really neat!
#          - What happens in I use the top 10 from a BanditOneHot tuning?

exp458_exp461: exp458 exp459 exp460 exp461 

# ep
exp458:
	tune_bandit.py random $(DATA_PATH)/exp458 \
		--exp_name='epsilon_bandit' \
		--env_name=DistractionBanditOneHigh10-v0 \
		--num_samples=1000 \
		--num_episodes=100 \
		--num_repeats=50 \
		--num_processes=39 \
		--log_space=True \
		--metric="total_R" \
		--epsilon='(0.01, 0.99)' \
		--lr_R='(0.001, 0.5)' 

# ep-decay
exp459:
	tune_bandit.py random $(DATA_PATH)/exp459 \
		--exp_name='epsilon_bandit' \
		--env_name=DistractionBanditOneHigh10-v0 \
		--num_samples=1000 \
		--num_episodes=100 \
		--num_repeats=50 \
		--num_processes=39 \
		--log_space=True \
		--metric="total_R" \
		--epsilon='(0.01, 0.99)' \
		--epsilon_decay_tau='(0.0001, 0.1)' \
		--lr_R='(0.001, 0.5)' 

# eta/dual value
exp460:
	tune_bandit.py random $(DATA_PATH)/exp460 \
		--exp_name='wsls_bandit' \
		--env_name=DistractionBanditOneHigh10-v0 \
		--num_samples=1000 \
		--num_episodes=100 \
		--num_repeats=50 \
		--num_processes=39 \
		--log_space=True \
		--metric="total_R" \
		--tie_threshold='(1e-9, 1e-2)' \
		--lr_R='(0.001, 0.5)' 

# beta
exp461:
	tune_bandit.py random $(DATA_PATH)/exp461 \
		--exp_name='softbeta_bandit' \
		--env_name=DistractionBanditOneHigh10-v0 \
		--num_samples=1000 \
		--num_episodes=100 \
		--num_repeats=50 \
		--num_processes=39 \
		--log_space=True \
		--metric="total_R" \
		--beta='(0.001, 10)' \
		--lr_R='(0.001, 0.5)' \
		--temp='(0.0001, 1000)'

## --- test top 10 ---
exp462_466: exp462 exp463 exp464 exp465 exp466

exp462_466_clean:
	-rm -rf $(DATA_PATH)/exp462
	-rm -rf $(DATA_PATH)/exp463
	-rm -rf $(DATA_PATH)/exp464
	-rm -rf $(DATA_PATH)/exp465
	-rm -rf $(DATA_PATH)/exp466

# meta 
exp462:
	# Get top 10
	head -n 11 $(DATA_PATH)/exp460_sorted.csv > tmp 
	# Run them 10 times
	parallel -j 39 \
			--joblog '$(DATA_PATH)/exp462.log' \
			--nice 19 --delay 0 --bar --colsep ',' --header : \
			'wsls_bandit.py --env_name=DistractionBanditOneHigh10-v0 --num_episodes=5000 --tie_break='next' --tie_threshold={tie_threshold} --lr_R={lr_R} --log_dir=$(DATA_PATH)/exp462/param{index}/run{1} --master_seed={1}' ::: {0..10} :::: tmp
	# Clean up
	rm tmp

# ep 
exp463:
	# Get top 10
	head -n 11 $(DATA_PATH)/exp458_sorted.csv > tmp 
	# Run them 10 times
	parallel -j 39 \
			--joblog '$(DATA_PATH)/exp463.log' \
			--nice 19 --delay 0 --bar --colsep ',' --header : \
			'epsilon_bandit.py --env_name=DistractionBanditOneHigh10-v0 --num_episodes=5000 --epsilon={epsilon} --lr_R={lr_R} --log_dir=$(DATA_PATH)/exp463/param{index}/run{1} --master_seed={1}' ::: {0..10} :::: tmp
	# Clean up
	rm tmp

# anneal-ep 
exp464:
	# Get top 10
	head -n 11 $(DATA_PATH)/exp459_sorted.csv > tmp 
	# Run them 10 times
	parallel -j 39 \
			--joblog '$(DATA_PATH)/exp464.log' \
			--nice 19 --delay 0 --bar --colsep ',' --header : \
			'epsilon_bandit.py --env_name=DistractionBanditOneHigh10-v0 --num_episodes=5000 --epsilon={epsilon} --epsilon_decay_tau={epsilon_decay_tau} --lr_R={lr_R} --log_dir=$(DATA_PATH)/exp464/param{index}/run{1} --master_seed={1}' ::: {0..10} :::: tmp
	# Clean up
	rm tmp

# beta 
exp465:
	# Get top 10
	head -n 11 $(DATA_PATH)/exp461_sorted.csv > tmp 
	# Run them 10 times
	parallel -j 39 \
			--joblog '$(DATA_PATH)/exp465.log' \
			--nice 19 --delay 0 --bar --colsep ',' --header : \
			'softbeta_bandit.py --env_name=DistractionBanditOneHigh10-v0 --num_episodes=5000 --beta={beta} --temp={temp} --lr_R={lr_R} --log_dir=$(DATA_PATH)/exp465/param{index}/run{1} --master_seed={1}' ::: {0..10} :::: tmp
	# Clean up
	rm tmp

# random
exp466:
	parallel -j 39 \
			--joblog '$(DATA_PATH)/exp466.log' \
			--nice 19 --delay 0 --bar --colsep ',' \
			'random_bandit.py --env_name=DistractionBanditOneHigh10-v0 --num_episodes=200  --lr_R=0.1 --log_dir=$(DATA_PATH)/exp466/param0/run{1} --master_seed={1}' ::: {1..100}

# ---------------------------------------------------------------------------
# 7-31-2020
# ef819ca
#
# *** CONTROL EXP ***
# - How robust are the model to distraction when thet were tuned for no
# distraction?
#
# Use some 'wrong' parameters from a top10 tune set.
# - they were tuned for BanditOneHigh10 
# - but use them on DistractionBanditOneHigh10 
# 
# RESULT: - with not the ideal parameters, meta DOES degrade for total reward
#           and best choice. 
#         - when you don't know ditraction is coming, curiosity can be a   
#           liability	
#         - It is still in the mix for best model, so here it is not that
#           terrible a liablity.
#         - WHAT WOULD IT TAKE TO BREAK E-explore? 

exp467_471: exp467 exp468 exp469 exp470 exp471

# meta 
exp467:
	# Get top 10
	head -n 11 $(DATA_PATH)/exp379_sorted.csv > tmp 
	# Run them 10 times
	parallel -j 39 \
			--joblog '$(DATA_PATH)/exp467.log' \
			--nice 19 --delay 0 --bar --colsep ',' --header : \
			'wsls_bandit.py --env_name=DistractionBanditOneHigh10-v0 --num_episodes=200 --tie_break='next' --tie_threshold={tie_threshold} --lr_R={lr_R} --log_dir=$(DATA_PATH)/exp467/param{index}/run{1} --master_seed={1}' ::: {0..10} :::: tmp
	# Clean up
	rm tmp

# ep 
exp468:
	# Get top 10
	head -n 11 $(DATA_PATH)/exp377_sorted.csv > tmp 
	# Run them 10 times
	parallel -j 39 \
			--joblog '$(DATA_PATH)/exp468.log' \
			--nice 19 --delay 0 --bar --colsep ',' --header : \
			'epsilon_bandit.py --env_name=DistractionBanditOneHigh10-v0 --num_episodes=200 --epsilon={epsilon} --lr_R={lr_R} --log_dir=$(DATA_PATH)/exp468/param{index}/run{1} --master_seed={1}' ::: {0..10} :::: tmp
	# Clean up
	rm tmp

# anneal-ep 
exp469:
	# Get top 10
	head -n 11 $(DATA_PATH)/exp378_sorted.csv > tmp 
	# Run them 10 times
	parallel -j 39 \
			--joblog '$(DATA_PATH)/exp469.log' \
			--nice 19 --delay 0 --bar --colsep ',' --header : \
			'epsilon_bandit.py --env_name=DistractionBanditOneHigh10-v0 --num_episodes=200 --epsilon={epsilon} --epsilon_decay_tau={epsilon_decay_tau} --lr_R={lr_R} --log_dir=$(DATA_PATH)/exp469/param{index}/run{1} --master_seed={1}' ::: {0..10} :::: tmp
	# Clean up
	rm tmp

# beta 
exp470:
	# Get top 10
	head -n 11 $(DATA_PATH)/exp380_sorted.csv > tmp 
	# Run them 10 times
	parallel -j 39 \
			--joblog '$(DATA_PATH)/exp470.log' \
			--nice 19 --delay 0 --bar --colsep ',' --header : \
			'softbeta_bandit.py --env_name=DistractionBanditOneHigh10-v0 --num_episodes=200 --beta={beta} --temp={temp} --lr_R={lr_R} --log_dir=$(DATA_PATH)/exp470/param{index}/run{1} --master_seed={1}' ::: {0..10} :::: tmp
	# Clean up
	rm tmp

# random
exp471:
	parallel -j 39 \
			--joblog '$(DATA_PATH)/exp471.log' \
			--nice 19 --delay 0 --bar --colsep ',' \
			'random_bandit.py --env_name=DistractionBanditOneHigh10-v0 --num_episodes=200  --lr_R=0.1 --log_dir=$(DATA_PATH)/exp471/param0/run{1} --master_seed={1}' ::: {1..100}


# ------------------------------------------------------------------------
# 8-4-2020
# 869ee49
#
# Test several new agents.
#
# RESULT: - Output looks sane. Bonuses behave as expected. Move on to 
#         - Move onto tune

test_agents1:
	-rm -rf $(DATA_PATH)/test
	# curiosity
	curiosity_bandit.py --env_name=InfoBlueYellow4b-v0 --num_episodes=80 --actor='DeterministicActor' --lr_E=1 --log_dir=$(DATA_PATH)/test/ --tie_threshold=0.001
	# random
	random_bandit.py --env_name=BanditOneHigh4-v0 --num_episodes=80 --lr_R=0.01 --log_dir=$(DATA_PATH)/test/test_random
	# meta
	wsls_bandit.py --env_name=BanditOneHigh4-v0 --num_episodes=80 --tie_threshold=0.001 --lr_R=0.01 --log_dir=$(DATA_PATH)/test/test_meta
	# softmeta
	softwsls_bandit.py --env_name=BanditOneHigh4-v0 --num_episodes=80 --tie_threshold=0.001 --temp=1.0 --lr_R=0.01 --log_dir=$(DATA_PATH)/test/test_softmeta
	# softbeta
	softbeta_bandit.py --env_name=BanditOneHigh4-v0 --num_episodes=80 --beta=1 --temp=1 --bonus=0 --lr_R=0.01 --log_dir=$(DATA_PATH)/test/test_softbeta
	# ep
	epsilon_bandit.py --env_name=BanditOneHigh4-v0 --num_episodes=80 --epsilon=0.1  --lr_R=0.01 --log_dir=$(DATA_PATH)/test/test_epsilon
	# ep-decay
	epsilon_bandit.py --env_name=BanditOneHigh4-v0 --num_episodes=80 --epsilon=0.1 --epsilon_decay_tau=0.0001 --lr_R=0.01 --log_dir=$(DATA_PATH)/test/test_decay
	# Novelty
	softbeta_bandit.py --env_name=BanditOneHigh4-v0 --num_episodes=80 --beta=0 --temp=1 --bonus=1 --lr_R=0.01 --log_dir=$(DATA_PATH)/test/test_novelty
	# Extrinsic only 
	softbeta_bandit.py --env_name=BanditOneHigh4-v0 --num_episodes=80 --beta=0 --temp=1 --bonus=0 --lr_R=0.01 --log_dir=$(DATA_PATH)/test/test_extrinsic
	# Count
	softcount_bandit.py --env_name=BanditOneHigh4-v0 --num_episodes=80 --beta=.1 --temp=1 --lr_R=0.01 --log_dir=$(DATA_PATH)/test/test_count
	# Entropy
	softentropy_bandit.py --env_name=BanditOneHigh4-v0 --num_episodes=80 --beta=1 --temp=10 --lr_R=0.01 --log_dir=$(DATA_PATH)/test/test_entropy



# ------------------------------------------------------------------------
# 8-4-2020
# 869ee49
#
# Tune params for:
# - softbeta_bandit in novelty mode
# - softbeta_bandit in extrinsic reward only mode
# - softcount_bandit 
# - entropy bandit

# -
# BanditOneHigh4
exp472_exp475: exp472 exp473 exp474 exp475 

# novelty
exp472:
	tune_bandit.py random $(DATA_PATH)/exp472 \
		--exp_name='softbeta_bandit' \
		--env_name=BanditOneHigh4-v0 \
		--num_samples=1000 \
		--num_episodes=40 \
		--num_repeats=50 \
		--num_processes=39 \
		--log_space=True \
		--metric="total_R" \
		--beta=0 \
		--bonus='(1, 100)' \
		--temp='(0.001, 1000)' \
		--lr_R='(0.001, 0.5)' 

# extrinsic
exp473:
	tune_bandit.py random $(DATA_PATH)/exp473 \
		--exp_name='softbeta_bandit' \
		--env_name=BanditOneHigh4-v0 \
		--num_samples=1000 \
		--num_episodes=40 \
		--num_repeats=50 \
		--num_processes=39 \
		--log_space=True \
		--metric="total_R" \
		--bonus=0 \
		--beta=0 \
		--temp='(0.001, 1000)' \
		--lr_R='(0.001, 0.5)' 

# count
exp474:
	tune_bandit.py random $(DATA_PATH)/exp474 \
		--exp_name='softcount_bandit' \
		--env_name=BanditOneHigh4-v0 \
		--num_samples=1000 \
		--num_episodes=40 \
		--num_repeats=50 \
		--num_processes=39 \
		--log_space=True \
		--metric="total_R" \
		--beta='(0.001, 10)' \
		--temp='(0.001, 1000)' \
		--lr_R='(0.001, 0.5)' 

# entropy
exp475:
	tune_bandit.py random $(DATA_PATH)/exp475 \
		--exp_name='softentropy_bandit' \
		--env_name=BanditOneHigh4-v0 \
		--num_samples=1000 \
		--num_episodes=40 \
		--num_repeats=50 \
		--num_processes=39 \
		--log_space=True \
		--metric="total_R" \
		--beta='(0.001, 10)' \
		--temp='(0.001, 1000)' \
		--lr_R='(0.001, 0.5)' 


# -
# BanditOneHigh10
exp476_exp479: exp476 exp477 exp478 exp479 

# novelty
exp476:
	tune_bandit.py random $(DATA_PATH)/exp476 \
		--exp_name='softbeta_bandit' \
		--env_name=BanditOneHigh10-v0 \
		--num_samples=1000 \
		--num_episodes=100 \
		--num_repeats=50 \
		--num_processes=39 \
		--log_space=True \
		--metric="total_R" \
		--beta=0 \
		--bonus='(1, 100)' \
		--temp='(0.001, 1000)' \
		--lr_R='(0.001, 0.5)' 

# extrinsic
exp477:
	tune_bandit.py random $(DATA_PATH)/exp477 \
		--exp_name='softbeta_bandit' \
		--env_name=BanditOneHigh10-v0 \
		--num_samples=1000 \
		--num_episodes=100 \
		--num_repeats=50 \
		--num_processes=39 \
		--log_space=True \
		--metric="total_R" \
		--bonus=0 \
		--beta=0 \
		--temp='(0.001, 1000)' \
		--lr_R='(0.001, 0.5)' 

# count
exp478:
	tune_bandit.py random $(DATA_PATH)/exp478 \
		--exp_name='softcount_bandit' \
		--env_name=BanditOneHigh10-v0 \
		--num_samples=1000 \
		--num_episodes=100 \
		--num_repeats=50 \
		--num_processes=39 \
		--log_space=True \
		--metric="total_R" \
		--beta='(0.001, 10)' \
		--temp='(0.001, 1000)' \
		--lr_R='(0.001, 0.5)' 

# entropy
exp479:
	tune_bandit.py random $(DATA_PATH)/exp479 \
		--exp_name='softentropy_bandit' \
		--env_name=BanditOneHigh10-v0 \
		--num_samples=1000 \
		--num_episodes=100 \
		--num_repeats=50 \
		--num_processes=39 \
		--log_space=True \
		--metric="total_R" \
		--beta='(0.001, 10)' \
		--temp='(0.001, 1000)' \
		--lr_R='(0.001, 0.5)' 


# -
# BanditOneHot121 
exp480_exp483: exp480 exp481 exp482 exp483


# novelty
exp480:
	tune_bandit.py random $(DATA_PATH)/exp480 \
		--exp_name='softbeta_bandit' \
		--env_name=BanditOneHigh121-v0 \
		--num_samples=1000 \
		--num_episodes=12100 \
		--num_repeats=50 \
		--num_processes=39 \
		--log_space=True \
		--metric="total_R" \
		--beta=0 \
		--bonus='(1, 100)' \
		--temp='(0.001, 1000)' \
		--lr_R='(0.001, 0.5)' 

# extrinsic
exp481:
	tune_bandit.py random $(DATA_PATH)/exp481 \
		--exp_name='softbeta_bandit' \
		--env_name=BanditOneHigh121-v0 \
		--num_samples=1000 \
		--num_episodes=12100 \
		--num_repeats=50 \
		--num_processes=39 \
		--log_space=True \
		--metric="total_R" \
		--bonus=0 \
		--beta=0 \
		--temp='(0.001, 1000)' \
		--lr_R='(0.001, 0.5)' 

# count
exp482:
	tune_bandit.py random $(DATA_PATH)/exp482 \
		--exp_name='softcount_bandit' \
		--env_name=BanditOneHigh121-v0 \
		--num_samples=1000 \
		--num_episodes=12100 \
		--num_repeats=50 \
		--num_processes=39 \
		--log_space=True \
		--metric="total_R" \
		--beta='(0.001, 10)' \
		--temp='(0.001, 1000)' \
		--lr_R='(0.001, 0.5)' 

# entropy
exp483:
	tune_bandit.py random $(DATA_PATH)/exp483 \
		--exp_name='softentropy_bandit' \
		--env_name=BanditOneHigh121-v0 \
		--num_samples=1000 \
		--num_episodes=12100 \
		--num_repeats=50 \
		--num_processes=39 \
		--log_space=True \
		--metric="total_R" \
		--beta='(0.001, 10)' \
		--temp='(0.001, 1000)' \
		--lr_R='(0.001, 0.5)' 


# -
# BanditHardAndSparse10
exp484_exp487: exp484 exp485 exp486 exp487


# novelty
exp484:
	tune_bandit.py random $(DATA_PATH)/exp484 \
		--exp_name='softbeta_bandit' \
		--env_name=BanditHardAndSparse10-v0 \
		--num_samples=1000 \
		--num_episodes=10000 \
		--num_repeats=50 \
		--num_processes=39 \
		--log_space=True \
		--metric="total_R" \
		--beta=0 \
		--bonus='(1, 100)' \
		--temp='(0.001, 1000)' \
		--lr_R='(0.001, 0.5)' 

# extrinsic
exp485:
	tune_bandit.py random $(DATA_PATH)/exp485 \
		--exp_name='softbeta_bandit' \
		--env_name=BanditHardAndSparse10-v0 \
		--num_samples=1000 \
		--num_episodes=10000 \
		--num_repeats=50 \
		--num_processes=39 \
		--log_space=True \
		--metric="total_R" \
		--bonus=0 \
		--beta=0 \
		--temp='(0.001, 1000)' \
		--lr_R='(0.001, 0.5)' 

# count
exp486:
	tune_bandit.py random $(DATA_PATH)/exp486 \
		--exp_name='softcount_bandit' \
		--env_name=BanditHardAndSparse10-v0 \
		--num_samples=1000 \
		--num_episodes=10000 \
		--num_repeats=50 \
		--num_processes=39 \
		--log_space=True \
		--metric="total_R" \
		--beta='(0.001, 10)' \
		--temp='(0.001, 1000)' \
		--lr_R='(0.001, 0.5)' 

# entropy
exp487:
	tune_bandit.py random $(DATA_PATH)/exp487 \
		--exp_name='softentropy_bandit' \
		--env_name=BanditHardAndSparse10-v0 \
		--num_samples=1000 \
		--num_episodes=10000 \
		--num_repeats=50 \
		--num_processes=39 \
		--log_space=True \
		--metric="total_R" \
		--beta='(0.001, 10)' \
		--temp='(0.001, 1000)' \
		--lr_R='(0.001, 0.5)' 


# -
# DeceptiveBanditOneHigh10
exp488_exp491: exp488 exp489 exp490 exp491


# novelty
exp488:
	tune_bandit.py random $(DATA_PATH)/exp488 \
		--exp_name='softbeta_bandit' \
		--env_name=DeceptiveBanditOneHigh10-v0 \
		--num_samples=1000 \
		--num_episodes=100 \
		--num_repeats=50 \
		--num_processes=39 \
		--log_space=True \
		--metric="total_R" \
		--beta=0 \
		--bonus='(1, 100)' \
		--temp='(0.001, 1000)' \
		--lr_R='(0.001, 0.5)' 

# extrinsic
exp489:
	tune_bandit.py random $(DATA_PATH)/exp489 \
		--exp_name='softbeta_bandit' \
		--env_name=DeceptiveBanditOneHigh10-v0 \
		--num_samples=1000 \
		--num_episodes=100 \
		--num_repeats=50 \
		--num_processes=39 \
		--log_space=True \
		--metric="total_R" \
		--bonus=0 \
		--beta=0 \
		--temp='(0.001, 1000)' \
		--lr_R='(0.001, 0.5)' 

# count
exp490:
	tune_bandit.py random $(DATA_PATH)/exp490 \
		--exp_name='softcount_bandit' \
		--env_name=DeceptiveBanditOneHigh10-v0 \
		--num_samples=1000 \
		--num_episodes=100 \
		--num_repeats=50 \
		--num_processes=39 \
		--log_space=True \
		--metric="total_R" \
		--beta='(0.001, 10)' \
		--temp='(0.001, 1000)' \
		--lr_R='(0.001, 0.5)' 

# entropy
exp491:
	tune_bandit.py random $(DATA_PATH)/exp491 \
		--exp_name='softentropy_bandit' \
		--env_name=DeceptiveBanditOneHigh10-v0 \
		--num_samples=1000 \
		--num_episodes=100 \
		--num_repeats=50 \
		--num_processes=39 \
		--log_space=True \
		--metric="total_R" \
		--beta='(0.001, 10)' \
		--temp='(0.001, 1000)' \
		--lr_R='(0.001, 0.5)' 

# - 
# DistractionBanditOneHigh10
exp492_exp495: exp492 exp493 exp494 exp495

# novelty
exp492:
	tune_bandit.py random $(DATA_PATH)/exp492 \
		--exp_name='softbeta_bandit' \
		--env_name=DistractionBanditOneHigh10-v0 \
		--num_samples=1000 \
		--num_episodes=100 \
		--num_repeats=50 \
		--num_processes=39 \
		--log_space=True \
		--metric="total_R" \
		--beta=0 \
		--bonus='(1, 100)' \
		--temp='(0.001, 1000)' \
		--lr_R='(0.001, 0.5)' 

# extrinsic
exp493:
	tune_bandit.py random $(DATA_PATH)/exp493 \
		--exp_name='softbeta_bandit' \
		--env_name=DistractionBanditOneHigh10-v0 \
		--num_samples=1000 \
		--num_episodes=100 \
		--num_repeats=50 \
		--num_processes=39 \
		--log_space=True \
		--metric="total_R" \
		--bonus=0 \
		--beta=0 \
		--temp='(0.001, 1000)' \
		--lr_R='(0.001, 0.5)' 

# count
exp494:
	tune_bandit.py random $(DATA_PATH)/exp494 \
		--exp_name='softcount_bandit' \
		--env_name=DistractionBanditOneHigh10-v0 \
		--num_samples=1000 \
		--num_episodes=100 \
		--num_repeats=50 \
		--num_processes=39 \
		--log_space=True \
		--metric="total_R" \
		--beta='(0.001, 10)' \
		--temp='(0.001, 1000)' \
		--lr_R='(0.001, 0.5)' 

# entropy
exp495:
	tune_bandit.py random $(DATA_PATH)/exp495 \
		--exp_name='softentropy_bandit' \
		--env_name=DistractionBanditOneHigh10-v0 \
		--num_samples=1000 \
		--num_episodes=100 \
		--num_repeats=50 \
		--num_processes=39 \
		--log_space=True \
		--metric="total_R" \
		--beta='(0.001, 10)' \
		--temp='(0.001, 1000)' \
		--lr_R='(0.001, 0.5)' 

# -------------------------------------------------------------------
# 8-4-2020
# 42c9acf
#
# Tune softbeta, and the ep-greedys on BanditOneHigh4. I was stealing
# HP from BanditOneHigh10 on prior runs.
#
# NOTE: The meta tune version and tests for BanditOneHigh4 are found in 
# recipes:
# - exp454-exp457

exp496_exp498: exp496 exp497 exp498

# softbeta
exp496:
	tune_bandit.py random $(DATA_PATH)/exp496 \
		--exp_name='softbeta_bandit' \
		--env_name=BanditOneHigh4-v0 \
		--num_samples=1000 \
		--num_episodes=40 \
		--num_repeats=50 \
		--num_processes=39 \
		--log_space=True \
		--metric="total_R" \
		--beta='(0.001, 10)' \
		--lr_R='(0.001, 0.5)' \
		--temp='(0.001, 1000)' \

# epsilon
exp497:
	tune_bandit.py random $(DATA_PATH)/exp497 \
		--exp_name='epsilon_bandit' \
		--env_name=BanditOneHigh4-v0 \
		--num_samples=1000 \
		--num_episodes=40 \
		--num_repeats=50 \
		--num_processes=39 \
		--log_space=True \
		--metric="total_R" \
		--epsilon='(0.01, 0.99)' \
		--lr_R='(0.001, 0.5)' 

# anneal
exp498:
	tune_bandit.py random $(DATA_PATH)/exp498 \
		--exp_name='epsilon_bandit' \
		--env_name=BanditOneHigh4-v0 \
		--num_samples=1000 \
		--num_episodes=40 \
		--num_repeats=50 \
		--num_processes=39 \
		--log_space=True \
		--metric="total_R" \
		--epsilon='(0.01, 0.99)' \
		--epsilon_decay_tau='(0.0001, 0.1)' \
		--lr_R='(0.001, 0.5)'

# -------------------------------------------------------------------------
# 8-6-2020
# 
# I need an example of meta for the intro figure. Run first an example
# with not the best HP, but that has interesting dynamics. That is,
# it shows off when *can* happen.
# 
# Then run 100 trials of a top-1 HP, for comparison, perhaps, in 
# another figure

# -- meta example, N = 100 ---

# bad tie_threshold
exp499:
	# Get top 10
	head -n 2 $(DATA_PATH)/exp454_sorted.csv > tmp 
	# Run them 10 times
	parallel -j 4 \
			--joblog '$(DATA_PATH)/exp499.log' \
			--nice 19 --delay 0 --bar --colsep ',' --header : \
			'wsls_bandit.py --env_name=BanditOneHigh4-v0 --num_episodes=200 --tie_break='next' --tie_threshold=0.0001 --lr_R={lr_R} --log_dir=$(DATA_PATH)/exp499/param{index}/run{1} --master_seed={1}' ::: {0..100} :::: tmp
	# Clean up
	rm tmp

# good tie_threshold
exp500:
	# Get top 10
	head -n 2 $(DATA_PATH)/exp454_sorted.csv > tmp 
	# Run them 10 times
	parallel -j 4 \
			--joblog '$(DATA_PATH)/exp500.log' \
			--nice 19 --delay 0 --bar --colsep ',' --header : \
			'wsls_bandit.py --env_name=BanditOneHigh4-v0 --num_episodes=200 --tie_break='next' --tie_threshold={tie_threshold} --lr_R={lr_R} --log_dir=$(DATA_PATH)/exp500/param{index}/run{1} --master_seed={1}' ::: {0..100} :::: tmp
	# Clean up
	rm tmp


# -------------------------------------------------------------------------
# 8-7-2020
# 97e0495
#
# --- Top10 test runs (v1) for tuning exps: exp472-exp498 ---
# Test the 'new' agents on all the tasks


# -
# BanditOneHigh4
exp501_exp504: exp501 exp502 exp503 exp504 

exp501_exp504_clean:
	-rm -rf $(DATA_PATH)/exp501
	-rm -rf $(DATA_PATH)/exp502
	-rm -rf $(DATA_PATH)/exp503
	-rm -rf $(DATA_PATH)/exp504

# novelty - params: exp472
# beta 
exp501:
	# Get top 10
	head -n 11 $(DATA_PATH)/exp472_sorted.csv > tmp 
	# Run them 10 times
	parallel -j 40 \
			--joblog '$(DATA_PATH)/exp501.log' \
			--nice 19 --delay 0 --bar --colsep ',' --header : \
			'softbeta_bandit.py --env_name=BanditOneHigh4-v0 --num_episodes=2000 --beta=0 --bonus={bonus} --temp={temp} --lr_R={lr_R} --log_dir=$(DATA_PATH)/exp501/param{index}/run{1} --master_seed={1}' ::: {0..10} :::: tmp
	# Clean up
	rm tmp

# extrinsic - params: exp473
exp502:
	# Get top 10
	head -n 11 $(DATA_PATH)/exp473_sorted.csv > tmp 
	# Run them 10 times
	parallel -j 40 \
			--joblog '$(DATA_PATH)/exp502.log' \
			--nice 19 --delay 0 --bar --colsep ',' --header : \
			'softbeta_bandit.py --env_name=BanditOneHigh4-v0 --num_episodes=2000 --beta=0 --bonus=0 --temp={temp} --lr_R={lr_R} --log_dir=$(DATA_PATH)/exp502/param{index}/run{1} --master_seed={1}' ::: {0..10} :::: tmp
	# Clean up
	rm tmp

# count - params: exp474
exp503:
	# Get top 10
	head -n 11 $(DATA_PATH)/exp474_sorted.csv > tmp 
	# Run them 10 times
	parallel -j 40 \
			--joblog '$(DATA_PATH)/exp503.log' \
			--nice 19 --delay 0 --bar --colsep ',' --header : \
			'softcount_bandit.py --env_name=BanditOneHigh4-v0 --num_episodes=2000 --beta={beta} --temp={temp} --lr_R={lr_R} --log_dir=$(DATA_PATH)/exp503/param{index}/run{1} --master_seed={1}' ::: {0..10} :::: tmp
	# Clean up
	rm tmp

# entropy - params: exp475
exp504:
	# Get top 10
	head -n 11 $(DATA_PATH)/exp475_sorted.csv > tmp 
	# Run them 10 times
	parallel -j 40 \
			--joblog '$(DATA_PATH)/exp504.log' \
			--nice 19 --delay 0 --bar --colsep ',' --header : \
			'softentropy_bandit.py --env_name=BanditOneHigh4-v0 --num_episodes=2000 --beta={beta} --temp={temp} --lr_R={lr_R} --log_dir=$(DATA_PATH)/exp504/param{index}/run{1} --master_seed={1}' ::: {0..10} :::: tmp
	# Clean up
	rm tmp

# -
# BanditOneHigh10
exp505_exp508: exp505 exp506 exp507 exp508 


# novelty - params: exp476
# beta 
exp505:
	# Get top 10
	head -n 11 $(DATA_PATH)/exp476_sorted.csv > tmp 
	# Run them 10 times
	parallel -j 40 \
			--joblog '$(DATA_PATH)/exp505.log' \
			--nice 19 --delay 0 --bar --colsep ',' --header : \
			'softbeta_bandit.py --env_name=BanditOneHigh10-v0 --num_episodes=200 --beta=0 --bonus={bonus} --temp={temp} --lr_R={lr_R} --log_dir=$(DATA_PATH)/exp505/param{index}/run{1} --master_seed={1}' ::: {0..10} :::: tmp
	# Clean up
	rm tmp

# extrinsic - params: exp477
exp506:
	# Get top 10
	head -n 11 $(DATA_PATH)/exp477_sorted.csv > tmp 
	# Run them 10 times
	parallel -j 40 \
			--joblog '$(DATA_PATH)/exp506.log' \
			--nice 19 --delay 0 --bar --colsep ',' --header : \
			'softbeta_bandit.py --env_name=BanditOneHigh10-v0 --num_episodes=200 --beta=0 --bonus=0 --temp={temp} --lr_R={lr_R} --log_dir=$(DATA_PATH)/exp506/param{index}/run{1} --master_seed={1}' ::: {0..10} :::: tmp
	# Clean up
	rm tmp

# count - params: exp478
exp507:
	# Get top 10
	head -n 11 $(DATA_PATH)/exp478_sorted.csv > tmp 
	# Run them 10 times
	parallel -j 40 \
			--joblog '$(DATA_PATH)/exp507.log' \
			--nice 19 --delay 0 --bar --colsep ',' --header : \
			'softcount_bandit.py --env_name=BanditOneHigh10-v0 --num_episodes=200 --beta={beta} --temp={temp} --lr_R={lr_R} --log_dir=$(DATA_PATH)/exp507/param{index}/run{1} --master_seed={1}' ::: {0..10} :::: tmp
	# Clean up
	rm tmp

# entropy - params: exp479
exp508:
	# Get top 10
	head -n 11 $(DATA_PATH)/exp479_sorted.csv > tmp 
	# Run them 10 times
	parallel -j 40 \
			--joblog '$(DATA_PATH)/exp508.log' \
			--nice 19 --delay 0 --bar --colsep ',' --header : \
			'softentropy_bandit.py --env_name=BanditOneHigh10-v0 --num_episodes=200 --beta={beta} --temp={temp} --lr_R={lr_R} --log_dir=$(DATA_PATH)/exp508/param{index}/run{1} --master_seed={1}' ::: {0..10} :::: tmp
	# Clean up
	rm tmp


# -
# BanditOneHigh121
exp509_exp512: exp509 exp510 exp511 exp512

exp509_exp512_clean:
	-rm -rf $(DATA_PATH)/exp509
	-rm -rf $(DATA_PATH)/exp510
	-rm -rf $(DATA_PATH)/exp511
	-rm -rf $(DATA_PATH)/exp512

# novelty - params: exp480
# beta 
exp509:
	# Get top 10
	head -n 11 $(DATA_PATH)/exp480_sorted.csv > tmp 
	# Run them 10 times
	parallel -j 40 \
			--joblog '$(DATA_PATH)/exp509.log' \
			--nice 19 --delay 0 --bar --colsep ',' --header : \
			'softbeta_bandit.py --env_name=BanditOneHigh121-v0 --num_episodes=60500 --beta=0 --bonus={bonus} --temp={temp} --lr_R={lr_R} --log_dir=$(DATA_PATH)/exp509/param{index}/run{1} --master_seed={1}' ::: {0..10} :::: tmp
	# Clean up
	rm tmp

# extrinsic - params: exp481
exp510:
	# Get top 10
	head -n 11 $(DATA_PATH)/exp481_sorted.csv > tmp 
	# Run them 10 times
	parallel -j 40 \
			--joblog '$(DATA_PATH)/exp510.log' \
			--nice 19 --delay 0 --bar --colsep ',' --header : \
			'softbeta_bandit.py --env_name=BanditOneHigh121-v0 --num_episodes=60500 --beta=0 --bonus=0 --temp={temp} --lr_R={lr_R} --log_dir=$(DATA_PATH)/exp510/param{index}/run{1} --master_seed={1}' ::: {0..10} :::: tmp
	# Clean up
	rm tmp

# count - params: exp482
exp511:
	# Get top 10
	head -n 11 $(DATA_PATH)/exp482_sorted.csv > tmp 
	# Run them 10 times
	parallel -j 40 \
			--joblog '$(DATA_PATH)/exp511.log' \
			--nice 19 --delay 0 --bar --colsep ',' --header : \
			'softcount_bandit.py --env_name=BanditOneHigh121-v0 --num_episodes=60500 --beta={beta} --temp={temp} --lr_R={lr_R} --log_dir=$(DATA_PATH)/exp511/param{index}/run{1} --master_seed={1}' ::: {0..10} :::: tmp
	# Clean up
	rm tmp

# entropy - params: exp483
exp512:
	# Get top 10
	head -n 11 $(DATA_PATH)/exp483_sorted.csv > tmp 
	# Run them 10 times
	parallel -j 40 \
			--joblog '$(DATA_PATH)/exp512.log' \
			--nice 19 --delay 0 --bar --colsep ',' --header : \
			'softentropy_bandit.py --env_name=BanditOneHigh121-v0 --num_episodes=60500 --beta={beta} --temp={temp} --lr_R={lr_R} --log_dir=$(DATA_PATH)/exp512/param{index}/run{1} --master_seed={1}' ::: {0..10} :::: tmp
	# Clean up
	rm tmp


# -
# BanditHardAndSparse10
exp513_exp516: exp513 exp514 exp515 exp516 

exp513_exp516_clean:
	-rm -rf $(DATA_PATH)/exp513
	-rm -rf $(DATA_PATH)/exp514
	-rm -rf $(DATA_PATH)/exp515
	-rm -rf $(DATA_PATH)/exp516


# novelty - params: exp484
# beta 
exp513:
	# Get top 10
	head -n 11 $(DATA_PATH)/exp484_sorted.csv > tmp 
	# Run them 10 times
	parallel -j 40 \
			--joblog '$(DATA_PATH)/exp513.log' \
			--nice 19 --delay 0 --bar --colsep ',' --header : \
			'softbeta_bandit.py --env_name=BanditHardAndSparse10-v0 --num_episodes=50000 --beta=0 --bonus={bonus} --temp={temp} --lr_R={lr_R} --log_dir=$(DATA_PATH)/exp513/param{index}/run{1} --master_seed={1}' ::: {0..10} :::: tmp
	# Clean up
	rm tmp

# extrinsic - params: exp485
exp514:
	# Get top 10
	head -n 11 $(DATA_PATH)/exp485_sorted.csv > tmp 
	# Run them 10 times
	parallel -j 40 \
			--joblog '$(DATA_PATH)/exp514.log' \
			--nice 19 --delay 0 --bar --colsep ',' --header : \
			'softbeta_bandit.py --env_name=BanditHardAndSparse10-v0 --num_episodes=50000 --beta=0 --bonus=0 --temp={temp} --lr_R={lr_R} --log_dir=$(DATA_PATH)/exp514/param{index}/run{1} --master_seed={1}' ::: {0..10} :::: tmp
	# Clean up
	rm tmp

# count - params: exp486
exp515:
	# Get top 10
	head -n 11 $(DATA_PATH)/exp486_sorted.csv > tmp 
	# Run them 10 times
	parallel -j 40 \
			--joblog '$(DATA_PATH)/exp515.log' \
			--nice 19 --delay 0 --bar --colsep ',' --header : \
			'softcount_bandit.py --env_name=BanditHardAndSparse10-v0 --num_episodes=50000 --beta={beta} --temp={temp} --lr_R={lr_R} --log_dir=$(DATA_PATH)/exp515/param{index}/run{1} --master_seed={1}' ::: {0..10} :::: tmp
	# Clean up
	rm tmp

# entropy - params: exp487
exp516:
	# Get top 10
	head -n 11 $(DATA_PATH)/exp487_sorted.csv > tmp 
	# Run them 10 times
	parallel -j 40 \
			--joblog '$(DATA_PATH)/exp516.log' \
			--nice 19 --delay 0 --bar --colsep ',' --header : \
			'softentropy_bandit.py --env_name=BanditHardAndSparse10-v0 --num_episodes=50000 --beta={beta} --temp={temp} --lr_R={lr_R} --log_dir=$(DATA_PATH)/exp516/param{index}/run{1} --master_seed={1}' ::: {0..10} :::: tmp
	# Clean up
	rm tmp


# -
# DeceptiveBanditOneHigh10
exp517_exp520: exp517 exp518 exp519 exp520


# novelty - params: exp488
# beta 
exp517:
	# Get top 10
	head -n 11 $(DATA_PATH)/exp488_sorted.csv > tmp 
	# Run them 10 times
	parallel -j 4 \
			--joblog '$(DATA_PATH)/exp517.log' \
			--nice 19 --delay 0 --bar --colsep ',' --header : \
			'softbeta_bandit.py --env_name=DeceptiveBanditOneHigh10-v0 --num_episodes=200 --beta=0 --bonus={bonus} --temp={temp} --lr_R={lr_R} --log_dir=$(DATA_PATH)/exp517/param{index}/run{1} --master_seed={1}' ::: {0..10} :::: tmp
	# Clean up
	rm tmp

# extrinsic - params: exp489
exp518:
	# Get top 10
	head -n 11 $(DATA_PATH)/exp489_sorted.csv > tmp 
	# Run them 10 times
	parallel -j 4 \
			--joblog '$(DATA_PATH)/exp518.log' \
			--nice 19 --delay 0 --bar --colsep ',' --header : \
			'softbeta_bandit.py --env_name=DeceptiveBanditOneHigh10-v0 --num_episodes=200 --beta=0 --bonus=0 --temp={temp} --lr_R={lr_R} --log_dir=$(DATA_PATH)/exp518/param{index}/run{1} --master_seed={1}' ::: {0..10} :::: tmp
	# Clean up
	rm tmp

# count - params: exp490
exp519:
	# Get top 10
	head -n 11 $(DATA_PATH)/exp490_sorted.csv > tmp 
	# Run them 10 times
	parallel -j 4 \
			--joblog '$(DATA_PATH)/exp519.log' \
			--nice 19 --delay 0 --bar --colsep ',' --header : \
			'softcount_bandit.py --env_name=DeceptiveBanditOneHigh10-v0 --num_episodes=200 --beta={beta} --temp={temp} --lr_R={lr_R} --log_dir=$(DATA_PATH)/exp519/param{index}/run{1} --master_seed={1}' ::: {0..10} :::: tmp
	# Clean up
	rm tmp

# entropy - params: exp491
exp520:
	# Get top 10
	head -n 11 $(DATA_PATH)/exp491_sorted.csv > tmp 
	# Run them 10 times
	parallel -j 4 \
			--joblog '$(DATA_PATH)/exp520.log' \
			--nice 19 --delay 0 --bar --colsep ',' --header : \
			'softentropy_bandit.py --env_name=DeceptiveBanditOneHigh10-v0 --num_episodes=200 --beta={beta} --temp={temp} --lr_R={lr_R} --log_dir=$(DATA_PATH)/exp520/param{index}/run{1} --master_seed={1}' ::: {0..10} :::: tmp
	# Clean up
	rm tmp

# -
# DistractionBanditOneHigh10
exp521_exp524: exp521 exp522 exp523 exp524 

exp521_exp524_clean:
	-rm -rf $(DATA_PATH)/exp521
	-rm -rf $(DATA_PATH)/exp522
	-rm -rf $(DATA_PATH)/exp523
	-rm -rf $(DATA_PATH)/exp524


# novelty - params: exp492
# beta 
exp521:
	# Get top 10
	head -n 11 $(DATA_PATH)/exp492_sorted.csv > tmp 
	# Run them 10 times
	parallel -j 40 \
			--joblog '$(DATA_PATH)/exp521.log' \
			--nice 19 --delay 0 --bar --colsep ',' --header : \
			'softbeta_bandit.py --env_name=DistractionBanditOneHigh10-v0 --num_episodes=5000 --beta=0 --bonus={bonus} --temp={temp} --lr_R={lr_R} --log_dir=$(DATA_PATH)/exp521/param{index}/run{1} --master_seed={1}' ::: {0..10} :::: tmp
	# Clean up
	rm tmp

# extrinsic - params: exp493
exp522:
	# Get top 10
	head -n 11 $(DATA_PATH)/exp493_sorted.csv > tmp 
	# Run them 10 times
	parallel -j 40 \
			--joblog '$(DATA_PATH)/exp522.log' \
			--nice 19 --delay 0 --bar --colsep ',' --header : \
			'softbeta_bandit.py --env_name=DistractionBanditOneHigh10-v0 --num_episodes=5000 --beta=0 --bonus=0 --temp={temp} --lr_R={lr_R} --log_dir=$(DATA_PATH)/exp522/param{index}/run{1} --master_seed={1}' ::: {0..10} :::: tmp
	# Clean up
	rm tmp

# count - params: exp494
exp523:
	# Get top 10
	head -n 11 $(DATA_PATH)/exp494_sorted.csv > tmp 
	# Run them 10 times
	parallel -j 40 \
			--joblog '$(DATA_PATH)/exp523.log' \
			--nice 19 --delay 0 --bar --colsep ',' --header : \
			'softcount_bandit.py --env_name=DistractionBanditOneHigh10-v0 --num_episodes=5000 --beta={beta} --temp={temp} --lr_R={lr_R} --log_dir=$(DATA_PATH)/exp523/param{index}/run{1} --master_seed={1}' ::: {0..10} :::: tmp
	# Clean up
	rm tmp

# entropy - params: exp495
exp524:
	# Get top 10
	head -n 11 $(DATA_PATH)/exp495_sorted.csv > tmp 
	# Run them 10 times
	parallel -j 40 \
			--joblog '$(DATA_PATH)/exp524.log' \
			--nice 19 --delay 0 --bar --colsep ',' --header : \
			'softentropy_bandit.py --env_name=DistractionBanditOneHigh10-v0 --num_episodes=5000 --beta={beta} --temp={temp} --lr_R={lr_R} --log_dir=$(DATA_PATH)/exp524/param{index}/run{1} --master_seed={1}' ::: {0..10} :::: tmp
	# Clean up
	rm tmp



# -------------------------------------------------------------------------
# 8-7-2020
# 97e0495
#
# --- Top10 test runs (v1) for tuning exps: exp496_exp498 ---
# (This was a tune for the east bandit. I was param stealing before.)

# BanditOneHigh4
exp525_exp527: exp525 exp526 exp527 

exp525_exp527_clean:
	-rm -rf $(DATA_PATH)/exp525 
	-rm -rf $(DATA_PATH)/exp526 
	-rm -rf $(DATA_PATH)/exp527 

# softbeta - param: exp496
exp525:
	# Get top 10
	head -n 11 $(DATA_PATH)/exp496_sorted.csv > tmp 
	# Run them 10 times
	parallel -j 40 \
			--joblog '$(DATA_PATH)/exp525.log' \
			--nice 19 --delay 0 --bar --colsep ',' --header : \
			'softbeta_bandit.py --env_name=BanditOneHigh4-v0 --num_episodes=2000 --beta={beta} --temp={temp} --lr_R={lr_R} --log_dir=$(DATA_PATH)/exp525/param{index}/run{1} --master_seed={1}' ::: {0..10} :::: tmp
	# Clean up
	rm tmp

# epsilon - param: exp497
exp526:
	# Get top 10
	head -n 11 $(DATA_PATH)/exp497_sorted.csv > tmp 
	# Run them 10 times
	parallel -j 40 \
			--joblog '$(DATA_PATH)/exp526.log' \
			--nice 19 --delay 0 --bar --colsep ',' --header : \
			'epsilon_bandit.py --env_name=BanditOneHigh4-v0 --num_episodes=2000 --epsilon={epsilon} --lr_R={lr_R} --log_dir=$(DATA_PATH)/exp526/param{index}/run{1} --master_seed={1}' ::: {0..10} :::: tmp
	# Clean up
	rm tmp

# anneal - param: exp498
exp527:
	# Get top 10
	head -n 11 $(DATA_PATH)/exp498_sorted.csv > tmp 
	# Run them 10 times
	parallel -j 40 \
			--joblog '$(DATA_PATH)/exp527.log' \
			--nice 19 --delay 0 --bar --colsep ',' --header : \
			'epsilon_bandit.py --env_name=BanditOneHigh4-v0 --num_episodes=2000 --epsilon={epsilon} --epsilon_decay_tau={epsilon_decay_tau} --lr_R={lr_R} --log_dir=$(DATA_PATH)/exp527/param{index}/run{1} --master_seed={1}' ::: {0..10} :::: tmp
	# Clean up
	rm tmp


# --------------------------------------------------------------------------
# --- Tune a new count mode on all the tasks ---
#
# In optimizing the count_model I used a EB form for the count bonus
# - count(action)**(-0.5)
#
# Another standard and even more common form is the UCB:
# - ((2 * np.log(n + 1)) / count(action))**(0.5)
#
# Reference: https://arxiv.org/abs/1611.04717

exp528_exp533: exp528 exp529 exp530 exp531 exp532 exp533 


exp528:
	tune_bandit.py random $(DATA_PATH)/exp528 \
		--exp_name='softcount_bandit' \
		--env_name=BanditOneHigh4-v0 \
		--num_samples=1000 \
		--num_episodes=40 \
		--num_repeats=50 \
		--num_processes=39 \
		--log_space=True \
		--metric="total_R" \
		--mode="UCB" \
		--beta='(0.001, 10)' \
		--temp='(0.001, 1000)' \
		--lr_R='(0.001, 0.5)'

exp529:
	tune_bandit.py random $(DATA_PATH)/exp529 \
		--exp_name='softcount_bandit' \
		--env_name=BanditOneHigh10-v0 \
		--num_samples=1000 \
		--num_episodes=100 \
		--num_repeats=50 \
		--num_processes=39 \
		--log_space=True \
		--metric="total_R" \
		--mode="UCB" \
		--beta='(0.001, 10)' \
		--temp='(0.001, 1000)' \
		--lr_R='(0.001, 0.5)' 

exp530:
	tune_bandit.py random $(DATA_PATH)/exp530 \
		--exp_name='softcount_bandit' \
		--env_name=BanditOneHigh121-v0 \
		--num_samples=1000 \
		--num_episodes=12100 \
		--num_repeats=50 \
		--num_processes=39 \
		--log_space=True \
		--metric="total_R" \
		--mode="UCB" \
		--beta='(0.001, 10)' \
		--temp='(0.001, 1000)' \
		--lr_R='(0.001, 0.5)'

exp531:
	tune_bandit.py random $(DATA_PATH)/exp531 \
		--exp_name='softcount_bandit' \
		--env_name=BanditHardAndSparse10-v0 \
		--num_samples=1000 \
		--num_episodes=10000 \
		--num_repeats=50 \
		--num_processes=39 \
		--log_space=True \
		--metric="total_R" \
		--mode="UCB" \
		--beta='(0.001, 10)' \
		--temp='(0.001, 1000)' \
		--lr_R='(0.001, 0.5)' 

exp532:
	tune_bandit.py random $(DATA_PATH)/exp532 \
		--exp_name='softcount_bandit' \
		--env_name=DeceptiveBanditOneHigh10-v0 \
		--num_samples=1000 \
		--num_episodes=100 \
		--num_repeats=50 \
		--num_processes=4 \
		--log_space=True \
		--metric="total_R" \
		--mode="UCB" \
		--beta='(0.001, 10)' \
		--temp='(0.001, 1000)' \
		--lr_R='(0.001, 0.5)'

exp533:
	tune_bandit.py random $(DATA_PATH)/exp533 \
		--exp_name='softcount_bandit' \
		--env_name=DistractionBanditOneHigh10-v0 \
		--num_samples=1000 \
		--num_episodes=100 \
		--num_repeats=50 \
		--num_processes=39 \
		--log_space=True \
		--metric="total_R" \
		--mode="UCB" \
		--beta='(0.001, 10)' \
		--temp='(0.001, 1000)' \
		--lr_R='(0.001, 0.5)' 

# --------------------------------------------------------------------------
# 8-7-2020
# 9b02a60
#
# To make illustrating model dynamics easier, I created some
# hand-desgined bandits. ExampleBandits.
#
# Here is a quick recipe to test if they run right.

test_example_bandits:
	-rm -rf $(DATA_PATH)/test
	# curiosity
	curiosity_bandit.py --env_name=ExampleInfoBandit1a-v0 --num_episodes=40 --actor='DeterministicActor' --lr_E=1 --log_dir=$(DATA_PATH)/test/ExampleInfoBandit1a --tie_threshold=0.001
	# curiosity
	curiosity_bandit.py --env_name=ExampleInfoBandit1b-v0 --num_episodes=40 --actor='DeterministicActor' --lr_E=1 --log_dir=$(DATA_PATH)/test/ExampleInfoBandit1b --tie_threshold=0.001
	# curiosity
	curiosity_bandit.py --env_name=ExampleInfoBandit1c-v0 --num_episodes=40 --actor='DeterministicActor' --lr_E=1 --log_dir=$(DATA_PATH)/test/ExampleInfoBandit1c --tie_threshold=0.001
	# meta
	wsls_bandit.py --env_name=ExampleBandit4-v0 --num_episodes=80 --tie_threshold=0.001 --lr_R=0.01 --log_dir=$(DATA_PATH)/test/test_meta


# --------------------------------------------------------------------------
# 8-7-2020
# 14ca683
#
# Rerun meta on several tasks BUT! 
# ...Give it knowledge of the tasks state-space ahead of time:
#
# --initial_bins='[(0, 0), (0, 1)]'

exp534_exp538: exp534 exp535 exp536 exp537 exp538

exp534_exp538_clean: 
	-rm -rf $(DATA_PATH)/exp534 
	-rm -rf $(DATA_PATH)/exp535 
	-rm -rf $(DATA_PATH)/exp536 
	-rm -rf $(DATA_PATH)/exp537 
	-rm -rf $(DATA_PATH)/exp538

# BanditOneHigh4
exp534:
	# Get top 10
	head -n 11 $(DATA_PATH)/exp455_sorted.csv > tmp 
	# Run them 10 times
	parallel -j 4 \
			--joblog '$(DATA_PATH)/exp534.log' \
			--nice 19 --delay 0 --bar --colsep ',' --header : \
			"wsls_bandit.py --env_name=BanditOneHigh4-v0 --num_episodes=2000 --tie_break='next' --tie_threshold={tie_threshold} --lr_R={lr_R} --initial_bins='[(0, 0), (0, 1)]' --log_dir=$(DATA_PATH)/exp534/param{index}/run{1} --master_seed={1}" ::: {0..10} :::: tmp
	# Clean up
	rm tmp


# BanditHardAndSparse10 
exp535:
	# Get top 10
	head -n 11 $(DATA_PATH)/exp387_sorted.csv > tmp 
	# Run them 10 times
	parallel -j 4 \
			--joblog '$(DATA_PATH)/exp535.log' \
			--nice 19 --delay 0 --bar --colsep ',' --header : \
			"wsls_bandit.py --env_name=BanditHardAndSparse10-v0 --num_episodes=50000 --tie_break='next' --tie_threshold={tie_threshold} --lr_R={lr_R} --initial_bins='[(0, 0), (0, 1)]' --log_dir=$(DATA_PATH)/exp535/param{index}/run{1} --master_seed={1}" ::: {0..10} :::: tmp
	# Clean up
	rm tmp

# BanditOneHigh121
exp536:
	# Get top 10
	head -n 11 $(DATA_PATH)/exp383_sorted.csv > tmp 
	# Run them 10 times
	parallel -j 4 \
			--joblog '$(DATA_PATH)/exp536.log' \
			--nice 19 --delay 0 --bar --colsep ',' --header : \
			"wsls_bandit.py --env_name=BanditOneHigh121-v0 --num_episodes=60500 --tie_break='next' --tie_threshold={tie_threshold} --lr_R={lr_R} --initial_bins='[(0, 0), (0, 1)]' --log_dir=$(DATA_PATH)/exp536/param{index}/run{1} --master_seed={1}" ::: {0..10} :::: tmp
	# Clean up
	rm tmp

# DeceptiveBanditOneHigh10
exp537:
	# Get top 10
	head -n 11 $(DATA_PATH)/exp391_sorted.csv > tmp 
	# Run them 10 times
	parallel -j 4 \
			--joblog '$(DATA_PATH)/exp537.log' \
			--nice 19 --delay 0 --bar --colsep ',' --header : \
			"wsls_bandit.py --env_name=DeceptiveBanditOneHigh10-v0 --num_episodes=200 --tie_break='next' --tie_threshold={tie_threshold} --lr_R={lr_R} --initial_bins='[(0, 0), (0, 1)]' --log_dir=$(DATA_PATH)/exp537/param{index}/run{1} --master_seed={1}" ::: {0..10} :::: tmp
	# Clean up
	rm tmp

# DistractionBanditOneHigh10
exp538:
	# Get top 10
	head -n 11 $(DATA_PATH)/exp460_sorted.csv > tmp 
	# Run them 10 times
	parallel -j 4 \
			--joblog '$(DATA_PATH)/exp538.log' \
			--nice 19 --delay 0 --bar --colsep ',' --header : \
			"wsls_bandit.py --env_name=DistractionBanditOneHigh10-v0 --num_episodes=5000 --tie_break='next' --tie_threshold={tie_threshold} --lr_R={lr_R} --initial_bins='[(0, 0), (0, 1)]' --log_dir=$(DATA_PATH)/exp538/param{index}/run{1} --master_seed={1}" ::: {0..10} :::: tmp
	# Clean up
	rm tmp

# ---------------------------------------------------------------------------
# --- Top10 test ---
# Tuned UCB count mode on all the tasks in exp528_exp533.
# 
# In this recipe set we run tests using the top10 models.
exp539_exp544: exp539 exp540 exp541 exp542 exp543 exp544

exp539_exp544_clean:
	-rm -rf $(DATA_PATH)/exp539
	-rm -rf $(DATA_PATH)/exp540
	-rm -rf $(DATA_PATH)/exp541
	-rm -rf $(DATA_PATH)/exp542
	-rm -rf $(DATA_PATH)/exp543
	-rm -rf $(DATA_PATH)/exp544

# -
# BanditOneHigh4 - param: exp528
exp539:
	# Get top 10
	head -n 11 $(DATA_PATH)/exp528_sorted.csv > tmp 
	# Run them 10 times
	parallel -j 40 \
			--joblog '$(DATA_PATH)/exp539.log' \
			--nice 19 --delay 0 --bar --colsep ',' --header : \
			'softcount_bandit.py --env_name=BanditOneHigh4-v0 --num_episodes=2000 --beta={beta} --temp={temp} --lr_R={lr_R} --log_dir=$(DATA_PATH)/exp539/param{index}/run{1} --master_seed={1}' ::: {0..10} :::: tmp
	# Clean up
	rm tmp

# BanditOneHigh10 - param: exp529
exp540:
	# Get top 10
	head -n 11 $(DATA_PATH)/exp529_sorted.csv > tmp 
	# Run them 10 times
	parallel -j 40 \
			--joblog '$(DATA_PATH)/exp540.log' \
			--nice 19 --delay 0 --bar --colsep ',' --header : \
			'softcount_bandit.py --env_name=BanditOneHigh10-v0 --num_episodes=10000 --beta={beta} --temp={temp} --lr_R={lr_R} --log_dir=$(DATA_PATH)/exp540/param{index}/run{1} --master_seed={1}' ::: {0..10} :::: tmp
	# Clean up
	rm tmp

# BanditOneHigh121 - param: exp530
exp541:
	# Get top 10
	head -n 11 $(DATA_PATH)/exp530_sorted.csv > tmp 
	# Run them 10 times
	parallel -j 40 \
			--joblog '$(DATA_PATH)/exp541.log' \
			--nice 19 --delay 0 --bar --colsep ',' --header : \
			'softcount_bandit.py --env_name=BanditOneHigh121-v0 --num_episodes=60500 --beta={beta} --temp={temp} --lr_R={lr_R} --log_dir=$(DATA_PATH)/exp541/param{index}/run{1} --master_seed={1}' ::: {0..10} :::: tmp
	# Clean up
	rm tmp


# BanditHardAndSparse10 - param: exp531
exp542:
	# Get top 10
	head -n 11 $(DATA_PATH)/exp531_sorted.csv > tmp 
	# Run them 10 times
	parallel -j 40 \
			--joblog '$(DATA_PATH)/exp542.log' \
			--nice 19 --delay 0 --bar --colsep ',' --header : \
			'softcount_bandit.py --env_name=BanditHardAndSparse10-v0 --num_episodes=50000 --beta={beta} --temp={temp} --lr_R={lr_R} --log_dir=$(DATA_PATH)/exp542/param{index}/run{1} --master_seed={1}' ::: {0..10} :::: tmp
	# Clean up
	rm tmp

# DeceptiveBanditOneHigh10 - param: exp532
exp543:
	# Get top 10
	head -n 11 $(DATA_PATH)/exp532_sorted.csv > tmp 
	# Run them 10 times
	parallel -j 40 \
			--joblog '$(DATA_PATH)/exp543.log' \
			--nice 19 --delay 0 --bar --colsep ',' --header : \
			'softcount_bandit.py --env_name=DeceptiveBanditOneHigh10-v0 --num_episodes=200 --beta={beta} --temp={temp} --lr_R={lr_R} --log_dir=$(DATA_PATH)/exp543/param{index}/run{1} --master_seed={1}' ::: {0..10} :::: tmp
	# Clean up
	rm tmp

# DistractionBanditOneHigh10 - param: exp533
exp544:
	# Get top 10
	head -n 11 $(DATA_PATH)/exp533_sorted.csv > tmp 
	# Run them 10 times
	parallel -j 40 \
			--joblog '$(DATA_PATH)/exp544.log' \
			--nice 19 --delay 0 --bar --colsep ',' --header : \
			'softcount_bandit.py --env_name=DistractionBanditOneHigh10-v0 --num_episodes=10000 --beta={beta} --temp={temp} --lr_R={lr_R} --log_dir=$(DATA_PATH)/exp544/param{index}/run{1} --master_seed={1}' ::: {0..10} :::: tmp
	# Clean up
	rm tmp

# ---------------------------------------------------------------------------
# 8-10-2020
# b2047e6
#
# In an analysis notebook names `exp_sparse.Rmd` I compared several variations
# of meta on the BanditHardAndSparse10 task (2)
# 
# From the it seems that exp535 is potentially the best, but 10000 trials
# is not enough?

# - Run exp535 out to 40000. The last version with 10000 is misleading 
# and too short?

# RESULTS: comparing several exps, meta does outperform or is near the top
#          for most HP BUT you must run it long enough. anneal-ep and 
#          other methods can gues right faster, sometimes. They also guess
#          wrong. In the long-term then, and over "enough: trials meta 
#          consistenyly wins.
#
#          ALSO note the p_base metric is more a measure of speed to final
#          outcome the p(best) at the end. Nature of how I est it in the 
#          main code. Should calc a better final p_best in the Rmd, and
#          report that.

# num_episodes=40000
exp545:
	# Get top 10
	head -n 11 $(DATA_PATH)/exp387_sorted.csv > tmp 
	# Run them 10 times
	parallel -j 40 \
			--joblog '$(DATA_PATH)/exp545.log' \
			--nice 19 --delay 0 --bar --colsep ',' --header : \
			"wsls_bandit.py --env_name=BanditHardAndSparse10-v0 --num_episodes=40000 --tie_break='next' --tie_threshold={tie_threshold} --lr_R={lr_R} --initial_bins='[(0, 0), (0, 1)]' --log_dir=$(DATA_PATH)/exp545/param{index}/run{1} --master_seed={1}" ::: {0..10} :::: tmp
	# Clean up
	rm tmp

# num_episodes=60000
exp546:
	# Get top 10
	head -n 11 $(DATA_PATH)/exp387_sorted.csv > tmp 
	# Run them 10 times
	parallel -j 40 \
			--joblog '$(DATA_PATH)/exp546.log' \
			--nice 19 --delay 0 --bar --colsep ',' --header : \
			"wsls_bandit.py --env_name=BanditHardAndSparse10-v0 --num_episodes=60000 --tie_break='next' --tie_threshold={tie_threshold} --lr_R={lr_R} --initial_bins='[(0, 0), (0, 1)]' --log_dir=$(DATA_PATH)/exp546/param{index}/run{1} --master_seed={1}" ::: {0..10} :::: tmp
	# Clean up
	rm tmp

# ------------------------------------------------------------------------
# 8/20/2020
# 0b87b60
#
# On this day I realized for the last couple months I have been running the
# wrong version of the 121 task. 
# - I have been running: BanditOneHigh121
# - I need to be running: BanditUniform121
# 
# These experiments rerun tune and test recipe for only those recipes that
# were, and will be again, candidates for inclusion in the paper (#1)
# 
# On this date, there were -> and became
# (tune file shown in parens)
#
# Ours
# - meta: exp403 
# Random
# - epsilon: exp404 
# - decay: exp405 
# - random: exp407 
# Reward
# - extrinsic: exp510
# Intrinsic 
# - info: exp406 
# - novelty: exp509 
# - entropy:  exp512
# - EB:  exp511 
# - UCB:  exp541

exp547_561: exp547 exp548 exp549 exp550 exp551 exp552 exp553 exp554 exp555 exp556 exp557 exp558 exp559 exp560 exp561

# --- meta ----
exp547:
	tune_bandit.py random $(DATA_PATH)/exp547 \
		--exp_name='wsls_bandit' \
		--env_name=BanditUniform121-v0 \
		--num_samples=1000 \
		--num_episodes=12100 \
		--num_repeats=50 \
		--num_processes=39 \
		--log_space=True \
		--metric="total_R" \
		--tie_threshold='(1e-9, 1e-2)' \
		--lr_R='(0.001, 0.5)' 

exp548:
	# Get top 10
	head -n 11 $(DATA_PATH)/exp547_sorted.csv > tmp 
	# Run them 10 times
	parallel -j 39 \
			--joblog '$(DATA_PATH)/exp548.log' \
			--nice 19 --delay 0 --bar --colsep ',' --header : \
			'wsls_bandit.py --env_name=BanditUniform121-v0 --num_episodes=60500 --tie_break='next' --tie_threshold={tie_threshold} --lr_R={lr_R} --log_dir=$(DATA_PATH)/exp548/param{index}/run{1} --master_seed={1}' ::: {0..10} :::: tmp
	# Clean up
	rm tmp

# --- epsilon ----
exp549:
	tune_bandit.py random $(DATA_PATH)/exp549 \
		--exp_name='epsilon_bandit' \
		--env_name=BanditUniform121-v0 \
		--num_samples=1000 \
		--num_episodes=12100 \
		--num_repeats=50 \
		--num_processes=39 \
		--log_space=True \
		--metric="total_R" \
		--epsilon='(0.01, 0.99)' \
		--lr_R='(0.001, 0.5)' 

exp550:
	# Get top 10
	head -n 11 $(DATA_PATH)/exp549_sorted.csv > tmp 
	# Run them 10 times
	parallel -j 39 \
			--joblog '$(DATA_PATH)/exp550.log' \
			--nice 19 --delay 0 --bar --colsep ',' --header : \
			'epsilon_bandit.py --env_name=BanditUniform121-v0 --num_episodes=60500 --epsilon={epsilon} --lr_R={lr_R} --log_dir=$(DATA_PATH)/exp550/param{index}/run{1} --master_seed={1}' ::: {0..10} :::: tmp
	# Clean up
	rm tmp

# --- anneal ---
exp551:
	tune_bandit.py random $(DATA_PATH)/exp551 \
		--exp_name='epsilon_bandit' \
		--env_name=BanditUniform121-v0 \
		--num_samples=1000 \
		--num_episodes=12100 \
		--num_repeats=50 \
		--num_processes=39 \
		--log_space=True \
		--metric="total_R" \
		--epsilon='(0.01, 0.99)' \
		--epsilon_decay_tau='(0.0001, 0.1)' \
		--lr_R='(0.001, 0.5)' 

exp552:
	# Get top 10
	head -n 11 $(DATA_PATH)/exp551_sorted.csv > tmp 
	# Run them 10 times
	parallel -j 39 \
			--joblog '$(DATA_PATH)/exp552.log' \
			--nice 19 --delay 0 --bar --colsep ',' --header : \
			'epsilon_bandit.py --env_name=BanditUniform121-v0 --num_episodes=60500 --epsilon={epsilon} --epsilon_decay_tau={epsilon_decay_tau} --lr_R={lr_R} --log_dir=$(DATA_PATH)/exp552/param{index}/run{1} --master_seed={1}' ::: {0..10} :::: tmp
	# Clean up
	rm tmp

# --- random ---
exp553:
	parallel -j 39 \
			--joblog '$(DATA_PATH)/exp553.log' \
			--nice 19 --delay 0 --bar --colsep ',' \
			'random_bandit.py --env_name=BanditUniform121-v0 --num_episodes=60500  --lr_R=0.1 --log_dir=$(DATA_PATH)/exp553/param0/run{1} --master_seed={1}' ::: {1..100}


# --- beta (info) --- 
exp554:
	tune_bandit.py random $(DATA_PATH)/exp554 \
		--exp_name='softbeta_bandit' \
		--env_name=BanditUniform121-v0 \
		--num_samples=1000 \
		--num_episodes=12100 \
		--num_repeats=50 \
		--num_processes=39 \
		--log_space=True \
		--metric="total_R" \
		--beta='(0.001, 10)' \
		--lr_R='(0.001, 0.5)' \
		--temp='(0.001, 1000)'

exp555:
	# Get top 10
	head -n 11 $(DATA_PATH)/exp554_sorted.csv > tmp 
	# Run them 10 times
	parallel -j 39 \
			--joblog '$(DATA_PATH)/exp555.log' \
			--nice 19 --delay 0 --bar --colsep ',' --header : \
			'softbeta_bandit.py --env_name=BanditUniform121-v0 --num_episodes=60500 --beta={beta} --temp={temp} --lr_R={lr_R} --log_dir=$(DATA_PATH)/exp555/param{index}/run{1} --master_seed={1}' ::: {0..10} :::: tmp
	# Clean up
	rm tmp

# --- novelty ---
exp556:
	tune_bandit.py random $(DATA_PATH)/exp556 \
		--exp_name='softbeta_bandit' \
		--env_name=BanditUniform121-v0 \
		--num_samples=1000 \
		--num_episodes=12100 \
		--num_repeats=50 \
		--num_processes=39 \
		--log_space=True \
		--metric="total_R" \
		--beta=0 \
		--bonus='(1, 100)' \
		--temp='(0.001, 1000)' \
		--lr_R='(0.001, 0.5)' 

exp557:
	# Get top 10
	head -n 11 $(DATA_PATH)/exp556_sorted.csv > tmp 
	# Run them 10 times
	parallel -j 40 \
			--joblog '$(DATA_PATH)/exp557.log' \
			--nice 19 --delay 0 --bar --colsep ',' --header : \
			'softbeta_bandit.py --env_name=BanditUniform121-v0 --num_episodes=60500 --beta=0 --bonus={bonus} --temp={temp} --lr_R={lr_R} --log_dir=$(DATA_PATH)/exp557/param{index}/run{1} --master_seed={1}' ::: {0..10} :::: tmp
	# Clean up
	rm tmp

# --- Count (EB) ---
exp558:
	tune_bandit.py random $(DATA_PATH)/exp558 \
		--exp_name='softcount_bandit' \
		--env_name=BanditUniform121-v0 \
		--num_samples=1000 \
		--num_episodes=12100 \
		--num_repeats=50 \
		--num_processes=39 \
		--log_space=True \
		--metric="total_R" \
		--beta='(0.001, 10)' \
		--temp='(0.001, 1000)' \
		--lr_R='(0.001, 0.5)' 

exp559:
	# Get top 10
	head -n 11 $(DATA_PATH)/exp558_sorted.csv > tmp 
	# Run them 10 times
	parallel -j 40 \
			--joblog '$(DATA_PATH)/exp559.log' \
			--nice 19 --delay 0 --bar --colsep ',' --header : \
			'softcount_bandit.py --env_name=BanditUniform121-v0 --num_episodes=60500 --beta={beta} --temp={temp} --lr_R={lr_R} --log_dir=$(DATA_PATH)/exp559/param{index}/run{1} --master_seed={1}' ::: {0..10} :::: tmp
	# Clean up
	rm tmp

# --- Count (UCB) ---
exp560:
	tune_bandit.py random $(DATA_PATH)/exp560 \
		--exp_name='softcount_bandit' \
		--env_name=BanditUniform121-v0 \
		--num_samples=1000 \
		--num_episodes=12100 \
		--num_repeats=50 \
		--num_processes=39 \
		--log_space=True \
		--metric="total_R" \
		--mode="UCB" \
		--beta='(0.001, 10)' \
		--temp='(0.001, 1000)' \
		--lr_R='(0.001, 0.5)'

exp561:
	# Get top 10
	head -n 11 $(DATA_PATH)/exp560_sorted.csv > tmp 
	# Run them 10 times
	parallel -j 4 \
			--joblog '$(DATA_PATH)/exp561.log' \
			--nice 19 --delay 0 --bar --colsep ',' --header : \
			'softcount_bandit.py --env_name=BanditUniform121-v0 --num_episodes=60500 --mode='UCB' --beta={beta} --temp={temp} --lr_R={lr_R} --log_dir=$(DATA_PATH)/exp561/param{index}/run{1} --master_seed={1}' ::: {0..10} :::: tmp
	# Clean up
	rm tmp

# ------------------------------------------------------------------------
# 8/20/2020
# 
# Run a random_bandit on BanditOneHigh4 for 2000 trials
# This is the final test number for this task. 
exp562:
	parallel -j 39 \
			--joblog '$(DATA_PATH)/exp562.log' \
			--nice 19 --delay 0 --bar --colsep ',' \
			'random_bandit.py --env_name=BanditOneHigh4-v0 --num_episodes=2000  --lr_R=0.1 --log_dir=$(DATA_PATH)/exp562/param0/run{1} --master_seed={1}' ::: {1..100}


# ------------------------------------------------------------------------
# 8/20/2020
# 8550d0a
# test Uniform
test_uniform:
	-rm -rf $(DATA_PATH)/test
	wsls_bandit.py --env_name=BanditUniform121-v0 --num_episodes=60500 --tie_break='next' --tie_threshold=0.00000001 --lr_R=0.01 --log_dir=$(DATA_PATH)/test/test_uniform

# ------------------------------------------------------------------------
# 8-21-2020
#
# 
# Generate some BanditOneHigh4 example data (for a paper fig)
#
# The bandit 'ExampleBandit4' has the same p(R) as BanditOneHigh4 but
# has fixed seeds/returns. These were tweaked from random draws to have some
# surprising violations in the 'best' arm choice. This makes for good examples
# even is it is a technically a bit off.
exp563_572: exp563 exp564 exp565 exp566 exp567 exp568 exp569 exp570 exp571 exp572

exp563_572_clean:
	-rm -rf $(DATA_PATH)/exp563
	-rm -rf $(DATA_PATH)/exp564
	-rm -rf $(DATA_PATH)/exp565 
	-rm -rf $(DATA_PATH)/exp566 
	-rm -rf $(DATA_PATH)/exp567 
	-rm -rf $(DATA_PATH)/exp568 
	-rm -rf $(DATA_PATH)/exp569 
	-rm -rf $(DATA_PATH)/exp570
	-rm -rf $(DATA_PATH)/exp571
	-rm -rf $(DATA_PATH)/exp572

# meta (based on exp457)
exp563:	
	wsls_bandit.py --env_name=ExampleBandit4-v0 --num_episodes=200 --tie_threshold=0.0005 --lr_R=0.05 --log_dir=$(DATA_PATH)/exp563/param0/run1/

# ep (based on exp526)
exp564:
	epsilon_bandit.py --env_name=ExampleBandit4-v0 --num_episodes=200 --epsilon=0.15 --lr_R=0.21 --log_dir=$(DATA_PATH)/exp564/param0/run1/

# anneal (based on exp527)
exp565:
	epsilon_bandit.py --env_name=ExampleBandit4-v0 --num_episodes=200 --epsilon=0.1 --epsilon_decay_tau=0.0026 --lr_R=0.21 --log_dir=$(DATA_PATH)/exp565/param0/run1/

# random (based on exp562)
exp566:
	random_bandit.py --env_name=ExampleBandit4-v0 --num_episodes=200  --lr_R=0.01 --log_dir=$(DATA_PATH)/exp566/param0/run1

# extrinsic (based on exp502)
exp567:
	softbeta_bandit.py --env_name=ExampleBandit4-v0 --num_episodes=200 --beta=0 --temp=0.268 --lr_R=0.49 --log_dir=$(DATA_PATH)/exp567/param0/run1/

# beta (info) (based on exp525)
exp568:
	softbeta_bandit.py --env_name=ExampleBandit4-v0 --num_episodes=200 --beta=0.23 --temp=0.021 --lr_R=0.0019 --log_dir=$(DATA_PATH)/exp568/param0/run1/

# novelty (based on exp501)
exp569:
	softbeta_bandit.py --env_name=ExampleBandit4-v0 --num_episodes=200 --beta=0 --bonus=1.42 --temp=0.047 --lr_R=0.013 --log_dir=$(DATA_PATH)/exp569/param0/run1/

# entropy (based on exp504)
exp570:
	softentropy_bandit.py --env_name=ExampleBandit4-v0 --num_episodes=200 --beta=0.11 --temp=0.061 --lr_R=0.024 --log_dir=$(DATA_PATH)/exp570/param0/run1/

# EB (based on exp503)
exp571:
	softcount_bandit.py --env_name=ExampleBandit4-v0 --num_episodes=200 --beta=0.040 --temp=0.163 --lr_R=0.058 --mode='EB' --log_dir=$(DATA_PATH)/exp571/param0/run1/

# UCB (based on exp539)
exp572:
	softcount_bandit.py --env_name=ExampleBandit4-v0 --num_episodes=200 --beta=0.062 --temp=0.193 --lr_R=0.46 --mode='UCB' --log_dir=$(DATA_PATH)/exp572/param0/run1/
	

# ------------------------------------------------------------------------
# 8/20/2020
# 0b87b60
#
# A few days ago I realized for the last couple months I have been running the
# wrong version of the 121 task. 
# - I have been running: BanditOneHigh121
# - I need to be running: BanditUniform121
#
# I reran most agents already (exp547_561). Somehow I missed extrinsic and 
# entopy in that rerun. Here I fix that.

exp573_576: exp573 exp574 exp575 exp576


# --- extrinsic ---
exp573:
	tune_bandit.py random $(DATA_PATH)/exp573 \
		--exp_name='softbeta_bandit' \
		--env_name=BanditUniform121-v0 \
		--num_samples=1000 \
		--num_episodes=12100 \
		--num_repeats=50 \
		--num_processes=39 \
		--log_space=True \
		--metric="total_R" \
		--bonus=0 \
		--beta=0 \
		--temp='(0.001, 1000)' \
		--lr_R='(0.001, 0.5)' 

exp574:
	# Get top 10
	head -n 11 $(DATA_PATH)/exp573_sorted.csv > tmp 
	# Run them 10 times
	parallel -j 40 \
			--joblog '$(DATA_PATH)/exp574.log' \
			--nice 19 --delay 0 --bar --colsep ',' --header : \
			'softbeta_bandit.py --env_name=BanditUniform121-v0 --num_episodes=60500 --beta=0 --bonus=0 --temp={temp} --lr_R={lr_R} --log_dir=$(DATA_PATH)/exp574/param{index}/run{1} --master_seed={1}' ::: {0..10} :::: tmp
	# Clean up
	rm tmp


# --- entropy ---
exp575:
	tune_bandit.py random $(DATA_PATH)/exp575 \
		--exp_name='softentropy_bandit' \
		--env_name=BanditUniform121-v0 \
		--num_samples=1000 \
		--num_episodes=12100 \
		--num_repeats=50 \
		--num_processes=39 \
		--log_space=True \
		--metric="total_R" \
		--beta='(0.001, 10)' \
		--temp='(0.001, 1000)' \
		--lr_R='(0.001, 0.5)'

exp576:
	# Get top 10
	head -n 11 $(DATA_PATH)/exp575_sorted.csv > tmp 
	# Run them 10 times
	parallel -j 40 \
			--joblog '$(DATA_PATH)/exp576.log' \
			--nice 19 --delay 0 --bar --colsep ',' --header : \
			'softentropy_bandit.py --env_name=BanditUniform121-v0 --num_episodes=60500 --beta={beta} --temp={temp} --lr_R={lr_R} --log_dir=$(DATA_PATH)/exp576/param{index}/run{1} --master_seed={1}' ::: {0..10} :::: tmp
	# Clean up
	rm tmp

# ------------------------------------------------------------------------
# 9-4-2020
# cafac88
#
# Run a softmeta exp using the top10 from meta. How much worse
# does stochastic search make reward collection?
exp577_578: exp577 exp578 

exp577:
	# Get top 10
	head -n 11 $(DATA_PATH)/exp455_sorted.csv > tmp 
	# Run them 10 times
	parallel -j 4 \
			--joblog '$(DATA_PATH)/exp577.log' \
			--nice 19 --delay 0 --bar --colsep ',' --header : \
			'softwsls_bandit.py --env_name=BanditOneHigh4-v0 --num_episodes=2000 --tie_break='next' --tie_threshold={tie_threshold} --lr_R={lr_R} --temp=0.1 --log_dir=$(DATA_PATH)/exp577/param{index}/run{1} --master_seed={1}' ::: {0..10} :::: tmp
	# Clean up
	rm tmp

exp578:
	# Get top 10
	head -n 11 $(DATA_PATH)/exp455_sorted.csv > tmp 
	# Run them 10 times
	parallel -j 4 \
			--joblog '$(DATA_PATH)/exp578.log' \
			--nice 19 --delay 0 --bar --colsep ',' --header : \
			'softwsls_bandit.py --env_name=BanditOneHigh4-v0 --num_episodes=2000 --tie_break='next' --tie_threshold={tie_threshold} --lr_R={lr_R} --temp=0.05 --log_dir=$(DATA_PATH)/exp578/param{index}/run{1} --master_seed={1}' ::: {0..10} :::: tmp
	# Clean up
	rm tmp

# ------------------------------------------------------------------------
# 9-4-2020
# cafac88
#
# For all models that rely on softmax to sample, set the temp=1e-12
# to make them essentially deterministic. Observe the result.
#
# Our target exps are:
# Reward
# - extrinsic: exp502
exp579_584: exp579 exp581 exp582 exp583 exp584


# extrinsic - params: exp473
exp579:
	# Get top 10
	head -n 11 $(DATA_PATH)/exp473_sorted.csv > tmp 
	# Run them 10 times
	parallel -j 4 \
			--joblog '$(DATA_PATH)/exp579.log' \
			--nice 19 --delay 0 --bar --colsep ',' --header : \
			'softbeta_bandit.py --env_name=BanditOneHigh4-v0 --num_episodes=2000 --beta=0 --bonus=0 --temp=1e-12 --lr_R={lr_R} --log_dir=$(DATA_PATH)/exp579/param{index}/run{1} --master_seed={1}' ::: {0..10} :::: tmp
	# Clean up
	rm tmp

# Intrinsic
# - info: exp525
exp580:
	# Get top 10
	head -n 11 $(DATA_PATH)/exp496_sorted.csv > tmp 
	# Run them 10 times
	parallel -j 4 \
			--joblog '$(DATA_PATH)/exp580.log' \
			--nice 19 --delay 0 --bar --colsep ',' --header : \
			'softbeta_bandit.py --env_name=BanditOneHigh4-v0 --num_episodes=2000 --beta={beta} --temp=1e-12 --lr_R={lr_R} --log_dir=$(DATA_PATH)/exp580/param{index}/run{1} --master_seed={1}' ::: {0..10} :::: tmp
	# Clean up
	rm tmp

# - novelty: exp501
exp581:
	# Get top 10
	head -n 11 $(DATA_PATH)/exp472_sorted.csv > tmp 
	# Run them 10 times
	parallel -j 4 \
			--joblog '$(DATA_PATH)/exp581.log' \
			--nice 19 --delay 0 --bar --colsep ',' --header : \
			'softbeta_bandit.py --env_name=BanditOneHigh4-v0 --num_episodes=2000 --beta=0 --bonus={bonus} --temp=1e-12 --lr_R={lr_R} --log_dir=$(DATA_PATH)/exp581/param{index}/run{1} --master_seed={1}' ::: {0..10} :::: tmp
	# Clean up
	rm tmp

# - entropy:  exp504
exp582:
	# Get top 10
	head -n 11 $(DATA_PATH)/exp475_sorted.csv > tmp 
	# Run them 10 times
	parallel -j 4 \
			--joblog '$(DATA_PATH)/exp582.log' \
			--nice 19 --delay 0 --bar --colsep ',' --header : \
			'softentropy_bandit.py --env_name=BanditOneHigh4-v0 --num_episodes=2000 --beta={beta} --temp=1e-12 --lr_R={lr_R} --log_dir=$(DATA_PATH)/exp582/param{index}/run{1} --master_seed={1}' ::: {0..10} :::: tmp
	# Clean up
	rm tmp

# - EB:  exp503
exp583:
	# Get top 10
	head -n 11 $(DATA_PATH)/exp474_sorted.csv > tmp 
	# Run them 10 times
	parallel -j 4 \
			--joblog '$(DATA_PATH)/exp583.log' \
			--nice 19 --delay 0 --bar --colsep ',' --header : \
			'softcount_bandit.py --env_name=BanditOneHigh4-v0 --num_episodes=2000 --beta={beta} --temp=1e-12 --lr_R={lr_R} --log_dir=$(DATA_PATH)/exp583/param{index}/run{1} --master_seed={1}' ::: {0..10} :::: tmp
	# Clean up
	rm tmp

# - UCB:  exp539
exp584:
	# Get top 10
	head -n 11 $(DATA_PATH)/exp528_sorted.csv > tmp 
	# Run them 10 times
	parallel -j 4 \
			--joblog '$(DATA_PATH)/exp584.log' \
			--nice 19 --delay 0 --bar --colsep ',' --header : \
			'softcount_bandit.py --env_name=BanditOneHigh4-v0 --num_episodes=2000 --beta={beta} --temp=1e-12 --lr_R={lr_R} --log_dir=$(DATA_PATH)/exp584/param{index}/run{1} --master_seed={1}' ::: {0..10} :::: tmp
	# Clean up
	rm tmp


# ------------------------------------------------------------------------
# 9-8-2020
# eb856bc 
#
# Test loading an old result and retraining it.
test_load1:
	-rm -rf $(DATA_PATH)/test
	# ours
	wsls_bandit.py --env_name=BanditChange121-v0 --num_episodes=100 --lr_R=0.1 --log_dir=$(DATA_PATH)/test/meta --tie_threshold=0.001 --load=$(DATA_PATH)/exp548/param1/run2/result.pkl
	# beta
	softbeta_bandit.py --env_name=BanditChange121-v0 --num_episodes=100 --beta=0.11 --lr_R=0.1 --temp=0.005 --log_dir=$(DATA_PATH)/test/softbeta --load=$(DATA_PATH)/exp555/param1/run2/result.pkl
	# ep
	epsilon_bandit.py --env_name=BanditChange121-v0 --num_episodes=100 --epsilon=0.11 --lr_R=0.1 --log_dir=$(DATA_PATH)/test/epsilon --load=$(DATA_PATH)/exp550/param1/run2/result.pkl
	# extrinsic
	softbeta_bandit.py --env_name=BanditChange121-v0 --num_episodes=100 --beta=0 --lr_R=0.1 --temp=0.09 --log_dir=$(DATA_PATH)/test/extrinsic --load=$(DATA_PATH)/exp555/param1/run2/result.pkl
	# entropy
	softentropy_bandit.py --env_name=BanditChange121-v0 --num_episodes=100 --lr_R=0.1 --beta=0.001 --temp=0.08 --log_dir=$(DATA_PATH)/test/entropy --load=$(DATA_PATH)/exp576/param1/run2/result.pkl
	# count
	softcount_bandit.py --env_name=BanditChange121-v0 --num_episodes=100 --lr_R=0.1 --beta=0.21 --temp=0.08 --mode='EB' --log_dir=$(DATA_PATH)/test/count --load=$(DATA_PATH)/exp559/param1/run2/result.pkl

# ------------------------------------------------------------------------
# 9-8-2020
# eb856bc 
#
# Run BanditChange121 using top10 params EXCEPT all lr_R is overridden to 0.1
# to spped up running the experimetns and see how fast different strategies
# can adapt. 
# 
# Fix --num_episodes=100
#
# Target exps/params are:
#
exp585_594: exp585 exp586 exp587 exp589 exp590 exp591 exp592 exp593 exp594

exp585_594_clean:
	-rm -rf $(DATA_PATH)/exp585
	-rm -rf $(DATA_PATH)/exp586
	-rm -rf $(DATA_PATH)/exp587
	-rm -rf $(DATA_PATH)/exp589
	-rm -rf $(DATA_PATH)/exp590
	-rm -rf $(DATA_PATH)/exp591
	-rm -rf $(DATA_PATH)/exp592
	-rm -rf $(DATA_PATH)/exp593
	-rm -rf $(DATA_PATH)/exp594


# BanditUniform121
# Ours
# - meta: exp548
exp585:
	# Get top 10
	head -n 11 $(DATA_PATH)/exp547_sorted.csv > tmp 
	# Run them 10 times
	parallel -j 39 \
			--joblog '$(DATA_PATH)/exp585.log' \
			--nice 19 --delay 0 --bar --colsep ',' --header : \
			'wsls_bandit.py --env_name=BanditChange121-v0 --num_episodes=12100 --tie_break='next' --tie_threshold={tie_threshold} --lr_R={lr_R} --log_dir=$(DATA_PATH)/exp585/param{index}/run{1} --master_seed={1} --load=$(DATA_PATH)/exp548/param{index}/run{1}/result.pkl' ::: {0..10} :::: tmp
	# Clean up
	rm tmp

# Random
# - epsilon: exp550
exp586:
	# Get top 10
	head -n 11 $(DATA_PATH)/exp549_sorted.csv > tmp 
	# Run them 10 times
	parallel -j 39 \
			--joblog '$(DATA_PATH)/exp586.log' \
			--nice 19 --delay 0 --bar --colsep ',' --header : \
			'epsilon_bandit.py --env_name=BanditChange121-v0 --num_episodes=12100 --epsilon={epsilon} --lr_R={lr_R} --log_dir=$(DATA_PATH)/exp586/param{index}/run{1} --master_seed={1} --load=$(DATA_PATH)/exp550/param{index}/run{1}/result.pkl' ::: {0..10} :::: tmp
	# Clean up
	rm tmp

# - decay: exp552
exp587:
	# Get top 10
	head -n 11 $(DATA_PATH)/exp551_sorted.csv > tmp 
	# Run them 10 times
	parallel -j 39 \
			--joblog '$(DATA_PATH)/exp587.log' \
			--nice 19 --delay 0 --bar --colsep ',' --header : \
			'epsilon_bandit.py --env_name=BanditChange121-v0 --num_episodes=12100 --epsilon={epsilon} --epsilon_decay_tau={epsilon_decay_tau} --lr_R={lr_R} --log_dir=$(DATA_PATH)/exp587/param{index}/run{1} --master_seed={1} --load=$(DATA_PATH)/exp552/param{index}/run{1}/result.pkl' ::: {0..10} :::: tmp
	# Clean up
	rm tmp

# - random: exp553
exp588:
	parallel -j 4 \
			--joblog '$(DATA_PATH)/exp588.log' \
			--nice 19 --delay 0 --bar --colsep ',' \
			'random_bandit.py --env_name=BanditChange121-v0 --num_episodes=12100  --lr_R=0.01 --log_dir=$(DATA_PATH)/exp588/param0/run{1} --master_seed={1}' ::: {1..100}

# Reward
# - extrinsic: exp574
exp589:
	# Get top 10
	head -n 11 $(DATA_PATH)/exp573_sorted.csv > tmp 
	# Run them 10 times
	parallel -j 39 \
			--joblog '$(DATA_PATH)/exp589.log' \
			--nice 19 --delay 0 --bar --colsep ',' --header : \
			'softbeta_bandit.py --env_name=BanditChange121-v0 --num_episodes=12100 --beta=0 --bonus=0 --temp={temp} --lr_R={lr_R} --log_dir=$(DATA_PATH)/exp589/param{index}/run{1} --master_seed={1} --load=$(DATA_PATH)/exp574/param{index}/run{1}/result.pkl' ::: {0..10} :::: tmp
	# Clean up
	rm tmp

# Intrinsic
# - info: exp555
exp590:
	# Get top 10
	head -n 11 $(DATA_PATH)/exp554_sorted.csv > tmp 
	# Run them 10 times
	parallel -j 39 \
			--joblog '$(DATA_PATH)/exp590.log' \
			--nice 19 --delay 0 --bar --colsep ',' --header : \
			'softbeta_bandit.py --env_name=BanditChange121-v0 --num_episodes=12100 --beta={beta} --temp={temp} --lr_R={lr_R} --log_dir=$(DATA_PATH)/exp590/param{index}/run{1} --master_seed={1} --load=$(DATA_PATH)/exp555/param{index}/run{1}/result.pkl' ::: {0..10} :::: tmp
	# Clean up
	rm tmp

# - novelty: exp557
exp591:
	# Get top 10
	head -n 11 $(DATA_PATH)/exp556_sorted.csv > tmp 
	# Run them 10 times
	parallel -j 39 \
			--joblog '$(DATA_PATH)/exp591.log' \
			--nice 19 --delay 0 --bar --colsep ',' --header : \
			'softbeta_bandit.py --env_name=BanditChange121-v0 --num_episodes=12100 --beta=0 --bonus={bonus} --temp={temp} --lr_R={lr_R} --log_dir=$(DATA_PATH)/exp591/param{index}/run{1} --master_seed={1} --load=$(DATA_PATH)/exp557/param{index}/run{1}/result.pkl' ::: {0..10} :::: tmp
	# Clean up
	rm tmp

# - entropy:  exp576
exp592:
	# Get top 10
	head -n 11 $(DATA_PATH)/exp575_sorted.csv > tmp 
	# Run them 10 times
	parallel -j 39 \
			--joblog '$(DATA_PATH)/exp592.log' \
			--nice 19 --delay 0 --bar --colsep ',' --header : \
			'softentropy_bandit.py --env_name=BanditChange121-v0 --num_episodes=12100 --beta={beta} --temp={temp} --lr_R={lr_R} --log_dir=$(DATA_PATH)/exp592/param{index}/run{1} --master_seed={1} --load=$(DATA_PATH)/exp576/param{index}/run{1}/result.pkl' ::: {0..10} :::: tmp
	# Clean up
	rm tmp

# - EB: exp559
exp593:
	# Get top 10
	head -n 11 $(DATA_PATH)/exp558_sorted.csv > tmp 
	# Run them 10 times
	parallel -j 39 \
			--joblog '$(DATA_PATH)/exp593.log' \
			--nice 19 --delay 0 --bar --colsep ',' --header : \
			'softcount_bandit.py --env_name=BanditChange121-v0 --num_episodes=12100 --beta={beta} --temp={temp} --lr_R={lr_R} --log_dir=$(DATA_PATH)/exp593/param{index}/run{1} --master_seed={1} --load=$(DATA_PATH)/exp559/param{index}/run{1}/result.pkl' ::: {0..10} :::: tmp
	# Clean up
	rm tmp

# - UCB: exp561
exp594:
	# Get top 10
	head -n 11 $(DATA_PATH)/exp560_sorted.csv > tmp 
	# Run them 10 times
	parallel -j 39 \
			--joblog '$(DATA_PATH)/exp594.log' \
			--nice 19 --delay 0 --bar --colsep ',' --header : \
			'softcount_bandit.py --env_name=BanditChange121-v0 --num_episodes=12100 --mode='UCB' --beta={beta} --temp={temp} --lr_R={lr_R} --log_dir=$(DATA_PATH)/exp594/param{index}/run{1} --master_seed={1} --load=$(DATA_PATH)/exp561/param{index}/run{1}/result.pkl' ::: {0..10} :::: tmp
	# Clean up
	rm tmp

# -----------------------------------------------------------------------
# 9-15-2020
# Tweaks
#
# Rerun some tune searchers for OneHigh4 and Uniform121 so that the num ep
# match the summary figure, and we use the median not the mean. Again like
# the sum fig. The choice of sum doesn't seem to in past exps change the 
# final results much. No need to run everything? This is just for demo/
# robustness of parrams figures.

# --- BanditOneHigh4 --- 
exp595_exp603: exp595 exp596 exp597 exp598 exp599 exp600 exp601 exp602 exp603

# meta
exp595:
	tune_bandit.py random $(DATA_PATH)/exp595 \
		--exp_name='wsls_bandit' \
		--env_name=BanditOneHigh4-v0 \
		--num_samples=1000 \
		--num_episodes=200 \
		--num_repeats=50 \
		--num_processes=39 \
		--log_space=True \
		--metric="total_R" \
		--tie_threshold='(1e-9, 1e-2)' \
		--lr_R='(0.001, 0.5)' 

# ep
exp596:
	tune_bandit.py random $(DATA_PATH)/exp596 \
		--exp_name='epsilon_bandit' \
		--env_name=BanditOneHigh4-v0 \
		--num_samples=1000 \
		--num_episodes=200 \
		--num_repeats=50 \
		--num_processes=39 \
		--log_space=True \
		--metric="total_R" \
		--epsilon='(0.01, 0.99)' \
		--lr_R='(0.001, 0.5)' 

# anneal
exp597:
	tune_bandit.py random $(DATA_PATH)/exp597 \
		--exp_name='epsilon_bandit' \
		--env_name=BanditOneHigh4-v0 \
		--num_samples=1000 \
		--num_episodes=200 \
		--num_repeats=50 \
		--num_processes=39 \
		--log_space=True \
		--metric="total_R" \
		--epsilon='(0.01, 0.99)' \
		--epsilon_decay_tau='(0.0001, 0.1)' \
		--lr_R='(0.001, 0.5)'

# extrinsic
exp598:
	tune_bandit.py random $(DATA_PATH)/exp598 \
		--exp_name='softbeta_bandit' \
		--env_name=BanditOneHigh4-v0 \
		--num_samples=1000 \
		--num_episodes=200 \
		--num_repeats=50 \
		--num_processes=39 \
		--log_space=True \
		--metric="total_R" \
		--bonus=0 \
		--beta=0 \
		--temp='(0.001, 1000)' \
		--lr_R='(0.001, 0.5)' 

# softbeta
exp599:
	tune_bandit.py random $(DATA_PATH)/exp599 \
		--exp_name='softbeta_bandit' \
		--env_name=BanditOneHigh4-v0 \
		--num_samples=1000 \
		--num_episodes=200 \
		--num_repeats=50 \
		--num_processes=39 \
		--log_space=True \
		--metric="total_R" \
		--beta='(0.001, 10)' \
		--lr_R='(0.001, 0.5)' \
		--temp='(0.001, 1000)' \


# novelty
exp600:
	tune_bandit.py random $(DATA_PATH)/exp600 \
		--exp_name='softbeta_bandit' \
		--env_name=BanditOneHigh4-v0 \
		--num_samples=1000 \
		--num_episodes=200 \
		--num_repeats=50 \
		--num_processes=39 \
		--log_space=True \
		--metric="total_R" \
		--beta=0 \
		--bonus='(1, 100)' \
		--temp='(0.001, 1000)' \
		--lr_R='(0.001, 0.5)' 

# entropy
exp601:
	tune_bandit.py random $(DATA_PATH)/exp601 \
		--exp_name='softentropy_bandit' \
		--env_name=BanditOneHigh4-v0 \
		--num_samples=1000 \
		--num_episodes=200 \
		--num_repeats=50 \
		--num_processes=39 \
		--log_space=True \
		--metric="total_R" \
		--beta='(0.001, 10)' \
		--temp='(0.001, 1000)' \
		--lr_R='(0.001, 0.5)' 

# count EB
exp602:
	tune_bandit.py random $(DATA_PATH)/exp602 \
		--exp_name='softcount_bandit' \
		--env_name=BanditOneHigh4-v0 \
		--num_samples=1000 \
		--num_episodes=200 \
		--num_repeats=50 \
		--num_processes=39 \
		--log_space=True \
		--metric="total_R" \
		--beta='(0.001, 10)' \
		--temp='(0.001, 1000)' \
		--lr_R='(0.001, 0.5)' 

# count UCB
exp603:
	tune_bandit.py random $(DATA_PATH)/exp603 \
		--exp_name='softcount_bandit' \
		--env_name=BanditOneHigh4-v0 \
		--num_samples=1000 \
		--num_episodes=200 \
		--num_repeats=50 \
		--num_processes=39 \
		--log_space=True \
		--metric="total_R" \
		--mode="UCB" \
		--beta='(0.001, 10)' \
		--temp='(0.001, 1000)' \
		--lr_R='(0.001, 0.5)' 


# --- BanditUniform121 --- 
exp604_exp612: exp604 exp605 exp606 exp607 exp608 exp609 exp610 exp611 exp612

# meta
exp604:
	tune_bandit.py random $(DATA_PATH)/exp604 \
		--exp_name='wsls_bandit' \
		--env_name=BanditUniform121-v0 \
		--num_samples=1000 \
		--num_episodes=2420 \
		--num_repeats=50 \
		--num_processes=39 \
		--log_space=True \
		--metric="total_R" \
		--tie_threshold='(1e-9, 1e-2)' \
		--lr_R='(0.001, 0.5)' 

# ep
exp605:
	tune_bandit.py random $(DATA_PATH)/exp605 \
		--exp_name='epsilon_bandit' \
		--env_name=BanditUniform121-v0 \
		--num_samples=1000 \
		--num_episodes=2420 \
		--num_repeats=50 \
		--num_processes=39 \
		--log_space=True \
		--metric="total_R" \
		--epsilon='(0.01, 0.99)' \
		--lr_R='(0.001, 0.5)' 

# anneal
exp606:
	tune_bandit.py random $(DATA_PATH)/exp606 \
		--exp_name='epsilon_bandit' \
		--env_name=BanditUniform121-v0 \
		--num_samples=1000 \
		--num_episodes=2420 \
		--num_repeats=50 \
		--num_processes=39 \
		--log_space=True \
		--metric="total_R" \
		--epsilon='(0.01, 0.99)' \
		--epsilon_decay_tau='(0.0001, 0.1)' \
		--lr_R='(0.001, 0.5)'

# extrinsic
exp607:
	tune_bandit.py random $(DATA_PATH)/exp607 \
		--exp_name='softbeta_bandit' \
		--env_name=BanditUniform121-v0 \
		--num_samples=1000 \
		--num_episodes=2420 \
		--num_repeats=50 \
		--num_processes=39 \
		--log_space=True \
		--metric="total_R" \
		--bonus=0 \
		--beta=0 \
		--temp='(0.001, 1000)' \
		--lr_R='(0.001, 0.5)' 

# softbeta
exp608:
	tune_bandit.py random $(DATA_PATH)/exp608 \
		--exp_name='softbeta_bandit' \
		--env_name=BanditUniform121-v0 \
		--num_samples=1000 \
		--num_episodes=2420 \
		--num_repeats=50 \
		--num_processes=39 \
		--log_space=True \
		--metric="total_R" \
		--beta='(0.001, 10)' \
		--lr_R='(0.001, 0.5)' \
		--temp='(0.001, 1000)' \


# novelty
exp609:
	tune_bandit.py random $(DATA_PATH)/exp609 \
		--exp_name='softbeta_bandit' \
		--env_name=BanditUniform121-v0 \
		--num_samples=1000 \
		--num_episodes=2420 \
		--num_repeats=50 \
		--num_processes=39 \
		--log_space=True \
		--metric="total_R" \
		--beta=0 \
		--bonus='(1, 100)' \
		--temp='(0.001, 1000)' \
		--lr_R='(0.001, 0.5)' 

# entropy
exp610:
	tune_bandit.py random $(DATA_PATH)/exp610 \
		--exp_name='softentropy_bandit' \
		--env_name=BanditUniform121-v0 \
		--num_samples=1000 \
		--num_episodes=2420 \
		--num_repeats=50 \
		--num_processes=39 \
		--log_space=True \
		--metric="total_R" \
		--beta='(0.001, 10)' \
		--temp='(0.001, 1000)' \
		--lr_R='(0.001, 0.5)' 

# count EB
exp611:
	tune_bandit.py random $(DATA_PATH)/exp611 \
		--exp_name='softcount_bandit' \
		--env_name=BanditUniform121-v0 \
		--num_samples=1000 \
		--num_episodes=2420 \
		--num_repeats=50 \
		--num_processes=39 \
		--log_space=True \
		--metric="total_R" \
		--beta='(0.001, 10)' \
		--temp='(0.001, 1000)' \
		--lr_R='(0.001, 0.5)' 

# count UCB
exp612:
	tune_bandit.py random $(DATA_PATH)/exp612 \
		--exp_name='softcount_bandit' \
		--env_name=BanditUniform121-v0 \
		--num_samples=1000 \
		--num_episodes=2420 \
		--num_repeats=50 \
		--num_processes=39 \
		--log_space=True \
		--metric="total_R" \
		--mode="UCB" \
		--beta='(0.001, 10)' \
		--temp='(0.001, 1000)' \
		--lr_R='(0.001, 0.5)' 


# ---------------------------------------------------------------------------
# 9-15-2020
#
# A temp/noise to the top1 models. Study one task BanditOneHigh4.
#
# - Study only softmax based models, including softmeta
# - Use tune_bandit buyt sample around the top1 values look
#   from the various sorted files **by hand**.
# --- BanditOneHigh4 --- 
exp613_621: exp613 exp614 exp615 exp616 exp617 exp618 exp619 exp620 exp621

# meta
# index,tie_threshold,lr_R,total_R
# 0,0.009372406727188936,0.05076952043479801,28.76
exp613:
	tune_bandit.py random $(DATA_PATH)/exp613 \
		--exp_name='softwsls_bandit' \
		--env_name=BanditOneHigh4-v0 \
		--num_samples=100 \
		--num_episodes=200 \
		--num_repeats=50 \
		--num_processes=39 \
		--log_space=True \
		--metric="total_R" \
		--tie_threshold=0.0093 \
		--lr_R=0.050 \
		--temp='(0.001, 1000)' 

# ep
# index,epsilon,lr_R,total_R
# 0,0.08360017944932674,0.10073435973976101,8398.96
exp614:
	tune_bandit.py random $(DATA_PATH)/exp614 \
		--exp_name='epsilon_bandit' \
		--env_name=BanditOneHigh4-v0 \
		--num_samples=100 \
		--num_episodes=200 \
		--num_repeats=50 \
		--num_processes=39 \
		--log_space=True \
		--metric="total_R" \
		--epsilon='(0.01, 0.99)' \
		--lr_R=0.10 

# anneal
# index,epsilon,epsilon_decay_tau,lr_R,total_R
# 0,0.10504798712146907,0.00026825893334438155,0.02268823892006516,26.74
exp615:
	tune_bandit.py random $(DATA_PATH)/exp615 \
		--exp_name='epsilon_bandit' \
		--env_name=BanditOneHigh4-v0 \
		--num_samples=100 \
		--num_episodes=200 \
		--num_repeats=50 \
		--num_processes=39 \
		--log_space=True \
		--metric="total_R" \
		--epsilon='(0.01, 0.99)' \
		--epsilon_decay_tau=0.00026 \
		--lr_R=0.022

# extrinsic
# index,bonus,beta,temp,lr_R,total_R
# 0,0.0,0.0,0.0688312308621586,0.49125026755811774,27.94
exp616:
	tune_bandit.py random $(DATA_PATH)/exp616 \
		--exp_name='softbeta_bandit' \
		--env_name=BanditOneHigh4-v0 \
		--num_samples=100 \
		--num_episodes=200 \
		--num_repeats=50 \
		--num_processes=39 \
		--log_space=True \
		--metric="total_R" \
		--bonus=0 \
		--beta=0 \
		--temp='(0.001, 1000)' \
		--lr_R=0.49

# softbeta
# index,beta,lr_R,temp,total_R
# 0,0.23147498497146318,0.05157737909861323,0.00195091441202382,29.24
exp617:
	tune_bandit.py random $(DATA_PATH)/exp617 \
		--exp_name='softbeta_bandit' \
		--env_name=BanditOneHigh4-v0 \
		--num_samples=100 \
		--num_episodes=200 \
		--num_repeats=50 \
		--num_processes=39 \
		--log_space=True \
		--metric="total_R" \
		--beta=0.23 \
		--lr_R=0.051 \
		--temp='(0.001, 1000)' \

# novelty
# index,beta,bonus,temp,lr_R,total_R
# 0,0.0,1.424414376251461,0.01770480968769014,0.013688280883830697,25.64
exp618:
	tune_bandit.py random $(DATA_PATH)/exp618 \
		--exp_name='softbeta_bandit' \
		--env_name=BanditOneHigh4-v0 \
		--num_samples=100 \
		--num_episodes=200 \
		--num_repeats=50 \
		--num_processes=39 \
		--log_space=True \
		--metric="total_R" \
		--beta=0 \
		--bonus=1.42 \
		--temp='(0.001, 1000)' \
		--lr_R=0.013

# entropy
# index,beta,temp,lr_R,total_R
# 0,0.0011065909026762382,0.021771088807868443,0.02409604919058677,27.16
exp619:
	tune_bandit.py random $(DATA_PATH)/exp619 \
		--exp_name='softentropy_bandit' \
		--env_name=BanditOneHigh4-v0 \
		--num_samples=100 \
		--num_episodes=200 \
		--num_repeats=50 \
		--num_processes=39 \
		--log_space=True \
		--metric="total_R" \
		--beta=0.0011 \
		--temp='(0.001, 1000)' \
		--lr_R=0.024 

# count EB
# index,beta,temp,lr_R,total_R
# 0,0.0040849400432943165,0.04328894058680061,0.058567276441824835,27.98
exp620:
	tune_bandit.py random $(DATA_PATH)/exp620 \
		--exp_name='softcount_bandit' \
		--env_name=BanditOneHigh4-v0 \
		--num_samples=100 \
		--num_episodes=200 \
		--num_repeats=50 \
		--num_processes=39 \
		--log_space=True \
		--metric="total_R" \
		--beta=0.004 \
		--temp='(0.001, 1000)' \
		--lr_R=0.058

# count UCB
# index,mode,beta,temp,lr_R,total_R
# 0,UCB,0.06284736159496815,0.06333056942295086,0.4616074035393582,27.88
exp621:
	tune_bandit.py random $(DATA_PATH)/exp621 \
		--exp_name='softcount_bandit' \
		--env_name=BanditOneHigh4-v0 \
		--num_samples=100 \
		--num_episodes=200 \
		--num_repeats=50 \
		--num_processes=39 \
		--log_space=True \
		--metric="total_R" \
		--mode="UCB" \
		--beta=0.062 \
		--temp='(0.001, 1000)' \
		--lr_R=0.46
