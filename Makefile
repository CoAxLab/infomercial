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
# - See notebooks/exp9_analysis.ipynb for full analysis
exp9:
	parallel -j 40 -v \
			--joblog '$(DATA_PATH)/exp9.log' \
			--nice 19 --delay 2 --colsep ',' \
			'meta_bandit.py --env_name {2} --num_episodes=10000 --tie_break='next' --tie_threshold=1e-9 --lr=0.1 --save=$(DATA_PATH)/exp9_{2}_{1}.pkl --interactive=False --seed_value={1}' ::: {1..50} ::: BanditOneHigh2-v0 BanditOneHigh10-v0 BanditOneHigh121-v0 BanditOneHigh1000-v0 BanditHardAndSparse2-v0 BanditHardAndSparse10-v0 BanditHardAndSparse121-v0 BanditHardAndSparse1000-v0

# lr = .1; tie_threshold = 1e-10
#
# SUM: exp9-10 decreasing threshold not helpful w/ lr = 1. I expected the opposite.
# - See notebooks/exp10_analysis.ipynb for full analysis
exp10:
	parallel -j 40 -v \
			--joblog '$(DATA_PATH)/exp10.log' \
			--nice 19 --delay 2 --colsep ',' \
			'meta_bandit.py --env_name {2} --num_episodes=10000 --tie_break='next' --tie_threshold=1e-10 --lr=0.1 --save=$(DATA_PATH)/exp10_{2}_{1}.pkl --interactive=False --seed_value={1}' ::: {1..50} ::: BanditOneHigh2-v0 BanditOneHigh10-v0 BanditOneHigh121-v0 BanditOneHigh1000-v0 BanditHardAndSparse2-v0 BanditHardAndSparse10-v0 BanditHardAndSparse121-v0 BanditHardAndSparse1000-v0


# lr = .2; tie_threshold = 1e-8
# 
# SUM:  OneHigh121 shows an approach to 1, that a large LOSS in p_best with learning. First time I've seen a loss. Don't really understand how that can be!
# - See notebooks/exp11_analysis.ipynb for full analysis
exp11:
	parallel -j 40 -v \
			--joblog '$(DATA_PATH)/exp11.log' \
			--nice 19 --delay 2 --colsep ',' \
			'meta_bandit.py --env_name {2} --num_episodes=10000 --tie_break='next' --tie_threshold=1e-8 --lr=0.2 --save=$(DATA_PATH)/exp11_{2}_{1}.pkl --interactive=False --seed_value={1}' ::: {1..50} ::: BanditOneHigh2-v0 BanditOneHigh10-v0 BanditOneHigh121-v0 BanditOneHigh1000-v0 BanditHardAndSparse2-v0 BanditHardAndSparse10-v0 BanditHardAndSparse121-v0 BanditHardAndSparse1000-v0

# lr = .2; tie_threshold = 1e-9
#
# 121 again shows a loss (see exp11) however it is MUCH more severe here.
# - See notebooks/exp12_analysis.ipynb for full analysis
exp12:
	parallel -j 40 -v \
			--joblog '$(DATA_PATH)/exp12.log' \
			--nice 19 --delay 2 --colsep ',' \
			'meta_bandit.py --env_name {2} --num_episodes=10000 --tie_break='next' --tie_threshold=1e-9 --lr=0.2 --save=$(DATA_PATH)/exp12_{2}_{1}.pkl --interactive=False --seed_value={1}' ::: {1..50} ::: BanditOneHigh2-v0 BanditOneHigh10-v0 BanditOneHigh121-v0 BanditOneHigh1000-v0 BanditHardAndSparse2-v0 BanditHardAndSparse10-v0 BanditHardAndSparse121-v0 BanditHardAndSparse1000-v0

# lr = .2; tie_threshold = 1e-10
# 
# SUM: loss on 121 again. No improvement otherwise.
# lr = 0.2 is just too high?
# - See notebooks/exp13_analysis.ipynb for full analysis
exp13:
	parallel -j 40 -v \
			--joblog '$(DATA_PATH)/exp13.log' \
			--nice 19 --delay 2 --colsep ',' \
			'meta_bandit.py --env_name {2} --num_episodes=10000 --tie_break='next' --tie_threshold=1e-10 --lr=0.2 --save=$(DATA_PATH)/exp13_{2}_{1}.pkl --interactive=False --seed_value={1}' ::: {1..50} ::: BanditOneHigh2-v0 BanditOneHigh10-v0 BanditOneHigh121-v0 BanditOneHigh1000-v0 BanditHardAndSparse2-v0 BanditHardAndSparse10-v0 BanditHardAndSparse121-v0 BanditHardAndSparse1000-v0

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
			'meta_bandit.py --env_name {2} --num_episodes=10000 --tie_break='next' --tie_threshold=1e-6 --lr=0.1 --save=$(DATA_PATH)/exp14_{2}_{1}.pkl --interactive=False --seed_value={1}' ::: {1..50} ::: BanditOneHigh2-v0 BanditOneHigh10-v0 BanditOneHigh121-v0 BanditOneHigh1000-v0 BanditHardAndSparse2-v0 BanditHardAndSparse10-v0 BanditHardAndSparse121-v0 BanditHardAndSparse1000-v0

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
			'meta_bandit.py --env_name {2} --num_episodes=10000 --tie_break='next' --tie_threshold=1e-4 --lr=0.1 --save=$(DATA_PATH)/exp15_{2}_{1}.pkl --interactive=False --seed_value={1}' ::: {1..50} ::: BanditOneHigh2-v0 BanditOneHigh10-v0 BanditOneHigh121-v0 BanditOneHigh1000-v0 BanditHardAndSparse2-v0 BanditHardAndSparse10-v0 BanditHardAndSparse121-v0 BanditHardAndSparse1000-v0

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
			'meta_bandit.py --env_name {2} --num_episodes=10000 --tie_break='next' --tie_threshold=1e-3 --lr=0.1 --save=$(DATA_PATH)/exp16_{2}_{1}.pkl --interactive=False --seed_value={1}' ::: {1..50} ::: BanditOneHigh2-v0 BanditOneHigh10-v0 BanditOneHigh121-v0 BanditOneHigh1000-v0 BanditHardAndSparse2-v0 BanditHardAndSparse10-v0 BanditHardAndSparse121-v0 BanditHardAndSparse1000-v0


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
			'meta_bandit.py --env_name {2} --num_episodes=10000 --tie_break='next' --tie_threshold=1e-5 --lr=0.1 --save=$(DATA_PATH)/exp17_{2}_{1}.pkl --interactive=False --seed_value={1}' ::: {1..50} ::: BanditOneHigh2-v0 BanditOneHigh10-v0 BanditOneHigh121-v0 BanditOneHigh1000-v0 BanditHardAndSparse2-v0 BanditHardAndSparse10-v0 BanditHardAndSparse121-v0 BanditHardAndSparse1000-v0

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
		--exp_name='meta_bandit' \
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
# 		--exp_name='meta_bandit' \
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
		--exp_name='meta_bandit' \
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
		--exp_name='meta_bandit' \
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
		--exp_name='meta_bandit' \
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
		--exp_name='meta_bandit' \
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
		--exp_name='meta_bandit' \
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
		--exp_name='meta_bandit' \
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
		--exp_name='meta_bandit' \
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
		--exp_name='meta_bandit' \
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
		--exp_name='meta_bandit' \
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
		--exp_name='meta_bandit' \
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
		--exp_name='meta_bandit' \
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
		--exp_name='meta_bandit' \
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
		--exp_name='meta_bandit' \
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
		--exp_name='meta_bandit' \
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
		--exp_name='meta_bandit' \
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
		--exp_name='meta_bandit' \
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
		--exp_name='meta_bandit' \
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
		--exp_name='meta_bandit' \
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
		--exp_name='meta_bandit' \
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
		--exp_name='meta_bandit' \
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
		--exp_name='meta_bandit' \
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
		--exp_name='meta_bandit' \
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
		--exp_name='meta_bandit' \
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
		--exp_name='meta_bandit' \
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
		--exp_name='meta_bandit' \
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
		--exp_name='meta_bandit' \
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
		--exp_name='meta_bandit' \
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
		--exp_name='meta_bandit' \
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
		--exp_name='meta_bandit' \
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
		--exp_name='meta_bandit' \
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
		--exp_name='meta_bandit' \
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
# NEXT: 100X didn't seem to help? For eff. returning to 20X
#       Try more replicators? I've solved this before, intermittently w/ hand 
#       tuning...; Try forcing tie_treshold to small value? 
#       I think that helped before?

# --num_iterations=16; --num_replicators=40
exp60:
	tune_bandit.py replicator $(DATA_PATH)/exp60 \
		--exp_name='meta_bandit' \
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
		--exp_name='meta_bandit' \
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
		--exp_name='meta_bandit' \
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
		--exp_name='meta_bandit' \
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
		--exp_name='meta_bandit' \
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
		--exp_name='meta_bandit' \
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
		--exp_name='meta_bandit' \
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

exp67:
	tune_bandit.py replicator $(DATA_PATH)/exp67 \
		--exp_name='meta_bandit' \
		--env_name=BanditHardAndSparse121-v0 \
		--num_iterations=10 \
		--num_episodes=1210 \
		--num_replicators=800 \
		--num_processes=4 \
		--verbose=True \
		--tie_threshold='(1e-10, 1e-6)' \
		--lr=0.1
