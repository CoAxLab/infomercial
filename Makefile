SHELL=/bin/bash -O expand_aliases
# DATA_PATH=/Users/qualia/Code/infomercial/data
DATA_PATH=/home/stitch/Code/infomercial/data/

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
# - See notebooks/exp60-66_analysis.ipynb for full analysis
#
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
#
# SUM: NO opts searches found the best arm. 
exp67:
	tune_bandit.py replicator $(DATA_PATH)/exp67 \
		--exp_name='meta_bandit' \
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
		--exp_name='meta_bandit' \
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
		--exp_name='meta_bandit' \
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
		--exp_name='meta_bandit' \
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
		--exp_name='meta_bandit' \
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
		--exp_name='meta_bandit' \
		--env_name=BanditOneHigh2-v0 \
        --num_episodes=20 \
        --num_samples=2400 \
        --num_processes=40 \
		--tie_threshold='(1e-10, 1e-3)' \
		--lr='(0.001, 0.2)'

# BanditHardAndSparse2
exp73:
	tune_bandit.py replicator $(DATA_PATH)/exp73 \
		--exp_name='meta_bandit' \
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
		--exp_name='meta_bandit' \
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
		--exp_name='meta_bandit' \
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
		--exp_name='meta_bandit' \
		--env_name=BanditOneHigh10-v0 \
        --num_episodes=100 \
        --num_samples=2400 \
        --num_processes=40 \
		--tie_threshold='(1e-10, 1e-3)' \
		--lr='(0.001, 0.2)'

# BanditHardAndSparse10
exp77:
	tune_bandit.py replicator $(DATA_PATH)/exp77 \
		--exp_name='meta_bandit' \
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
		--exp_name='meta_bandit' \
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
		--exp_name='meta_bandit' \
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
		--exp_name='meta_bandit' \
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
		--exp_name='meta_bandit' \
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
		--exp_name='meta_bandit' \
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
# As a result, the API for `meta_bandit` is now changed. 
#
# !!! THIS BREAKS THE API OF ALL EXPS PREVIOUS TO THIS ONE !!!
#
# For usage examples see `notebooks/exptest_meta_bandit.ipynb`.

# Run some new exps with the Bellman form

# BanditOneHigh121
# SUM: top half of best params solve i. Exploration was truncated. That's
# OK here. Might not be O in harder tasks
exp83:
	tune_bandit.py replicator $(DATA_PATH)/exp83 \
		--exp_name='meta_bandit' \
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
		--exp_name='meta_bandit' \
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
		--exp_name='meta_bandit' \
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
		--exp_name='meta_bandit' \
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
		--exp_name='meta_bandit' \
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
		--exp_name='meta_bandit' \
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
		--exp_name='meta_bandit' \
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
		--exp_name='meta_bandit' \
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
		--exp_name='meta_bandit' \
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
		--exp_name='meta_bandit' \
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
# SUM: in the exptest_meta_bandit notebook I confirmed all oneHigh bandits
# are still easily solved.

# Now let's re-try some Sparse problems

# w/ random search

# SUM: 93 and 94 found good solutions. 
#      95 result made no obvious progress on the problem.

exp93:
	tune_bandit.py random $(DATA_PATH)/exp93 \
		--exp_name='meta_bandit' \
		--env_name=BanditHardAndSparse2-v0 \
        --num_episodes=2000 \
        --num_samples=1000 \
        --num_processes=40 \
		--tie_threshold='(1e-16, 0.01)' \
		--lr_R='(0.0000001, 0.2)' \

exp94:
	tune_bandit.py random $(DATA_PATH)/exp94 \
		--exp_name='meta_bandit' \
		--env_name=BanditHardAndSparse10-v0 \
        --num_episodes=50000 \
        --num_samples=1000 \
        --num_processes=40 \
		--tie_threshold='(1e-16, 0.01)' \
		--lr_R='(0.0000001, 0.2)' \

exp95:
	tune_bandit.py random $(DATA_PATH)/exp95 \
		--exp_name='meta_bandit' \
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
		--exp_name='meta_bandit' \
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
		--exp_name='meta_bandit' \
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
		--exp_name='meta_bandit' \
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
		--exp_name='meta_bandit' \
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
		--exp_name='meta_bandit' \
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
		--exp_name='meta_bandit' \
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
			'meta_bandit.py --env_name=BanditOneHigh10-v0 --num_episodes=500 --tie_break='next' --tie_threshold=0.0041 --lr_R=0.31 --save=$(DATA_PATH)/exp128_{1}.pkl --interactive=False --debug=False --seed_value={1}' ::: {1..100}

# beta: exp98 - learns a stable soln 
#   + 'beta': 0.37, 'lr_R': 0.0095
exp129:
	parallel -j 40 \
			--joblog '$(DATA_PATH)/exp129.log' \
			--nice 19 --delay 2 --colsep ',' \
			'beta_bandit.py --env_name=BanditOneHigh10-v0 --num_episodes=500  --lr_R=0.31 --beta=0.37 --save=$(DATA_PATH)/exp129_{1}.pkl --interactive=False --debug=False --seed_value={1}' ::: {1..100}

# softbeta: exp112 - learns a stable soln 
#   + 'beta': 0.045, 'lr_R': 0.12, 'temp': 0.10
exp130:
	parallel -j 40 \
			--joblog '$(DATA_PATH)/exp130.log' \
			--nice 19 --delay 2 --colsep ',' \
			'softbeta_bandit.py --env_name=BanditOneHigh10-v0 --num_episodes=500  --lr_R=0.12 --beta=0.045 --temp=0.01 --save=$(DATA_PATH)/exp130_{1}.pkl --interactive=False --debug=False --seed_value={1}' ::: {1..100}

# epsilon: exp97 - learns a stable soln 
#   + 'epsilon': 0.078, 'lr_R': 0.12
exp131:
	parallel -j 40 \
			--joblog '$(DATA_PATH)/exp131.log' \
			--nice 19 --delay 2 --colsep ',' \
			'epsilon_bandit.py --env_name=BanditOneHigh10-v0 --num_episodes=500  --lr_R=0.12 --epsilon=0.078 --save=$(DATA_PATH)/exp131_{1}.pkl --interactive=False --debug=False --seed_value={1}' ::: {1..100}

# -
# BanditTwoHigh10
# meta: exp100 - sees both, learns a stable soln
#   + 'tie_threshold': 0.0058, 'lr_R': 0.14
exp132:
	parallel -j 40 \
			--joblog '$(DATA_PATH)/exp132.log' \
			--nice 19 --delay 2 --colsep ',' \
			'meta_bandit.py --env_name=BanditTwoHigh10-v0 --num_episodes=500 --tie_break='next' --tie_threshold=0.0058 --lr_R=0.14 --save=$(DATA_PATH)/exp132_{1}.pkl --interactive=False --debug=False --seed_value={1}' ::: {1..100}

# beta: exp102 - learns only one arm. Never sees best arm 2
#   + 'beta': 0.025, 'lr_R': 0.073
exp133:
	parallel -j 40 \
			--joblog '$(DATA_PATH)/exp133.log' \
			--nice 19 --delay 2 --colsep ',' \
			'beta_bandit.py --env_name=BanditTwoHigh10-v0 --num_episodes=500  --lr_R=0.073 --beta=0.025 --save=$(DATA_PATH)/exp133_{1}.pkl --interactive=False --debug=False --seed_value={1}' ::: {1..100}

# softbeta: exp113 - sees both (probably?), learns a stable soln
#   + 'beta': 0.010, 'lr_R': 0.17, 'temp': 0.24
exp134:
	parallel -j 40 \
			--joblog '$(DATA_PATH)/exp134.log' \
			--nice 19 --delay 2 --colsep ',' \
			'softbeta_bandit.py --env_name=BanditTwoHigh10-v0 --num_episodes=500  --lr_R=0.17 --beta=0.010 --temp=0.24 --save=$(DATA_PATH)/exp134_{1}.pkl --interactive=False --debug=False --seed_value={1}' ::: {1..100}

# epsilon: exp101 - learns solns, flip flops between them
#   + 'epsilon': 0.078, 'lr_R': 0.12
exp135:
	parallel -j 40 \
			--joblog '$(DATA_PATH)/exp135.log' \
			--nice 19 --delay 2 --colsep ',' \
			'epsilon_bandit.py --env_name=BanditTwoHigh10-v0 --num_episodes=500  --lr_R=0.12 --epsilon=0.078 --save=$(DATA_PATH)/exp135_{1}.pkl --interactive=False --debug=False --seed_value={1}' ::: {1..100}

# -
# BanditUniform121
# meta: exp124 - found stable soln
#   + 'tie_threshold': 0.00031, 'lr_R': 0.14
exp136:
	parallel -j 40 \
			--joblog '$(DATA_PATH)/exp136.log' \
			--nice 19 --delay 2 --colsep ',' \
			'meta_bandit.py --env_name=BanditUniform121-v0 --num_episodes=120000 --tie_break='next' --tie_threshold=0.00031 --lr_R=0.14 --save=$(DATA_PATH)/exp136_{1}.pkl --interactive=False --debug=False --seed_value={1}' ::: {1..100}

# beta: exp126 - found stable soln (very eff.)
#   + 'beta': 0.090, 'lr_R': 0.061
exp137:
	parallel -j 40 \
			--joblog '$(DATA_PATH)/exp137.log' \
			--nice 19 --delay 2 --colsep ',' \
			'beta_bandit.py --env_name=BanditUniform121-v0 --num_episodes=120000  --lr_R=0.061 --beta=0.090 --save=$(DATA_PATH)/exp137_{1}.pkl --interactive=False --debug=False --seed_value={1}' ::: {1..100}

# softbeta: exp127 - no soln found. p_best low (temp too)
#   + 'beta': 0.60, 'lr_R': 0.097, 'temp': 0.13
exp138:
	parallel -j 40 \
			--joblog '$(DATA_PATH)/exp138.log' \
			--nice 19 --delay 2 --colsep ',' \
			'softbeta_bandit.py --env_name=BanditUniform121-v0 --num_episodes=120000  --lr_R=0.097 --beta=0.60 --temp=0.13 --save=$(DATA_PATH)/exp138_{1}.pkl --interactive=False --debug=False --seed_value={1}' ::: {1..100}

# epsilon: exp125: found stable soln (low ep)
#   + 'epsilon': 0.012, 'lr_R': 0.11
exp139:
	parallel -j 40 \
			--joblog '$(DATA_PATH)/exp139.log' \
			--nice 19 --delay 2 --colsep ',' \
			'epsilon_bandit.py --env_name=BanditUniform121-v0 --num_episodes=120000  --lr_R=0.11 --epsilon=0.012 --save=$(DATA_PATH)/exp139_{1}.pkl --interactive=False --debug=False --seed_value={1}' ::: {1..100}

# -
# HardAndSparse10
# meta: meta: exp116 - learns a stable soln 
#   + 'tie_threshold': 3.76-09, 'lr_R': 0.00021
exp140:
	parallel -j 40 \
			--joblog '$(DATA_PATH)/exp140.log' \
			--nice 19 --delay 2 --colsep ',' \
			'meta_bandit.py --env_name=BanditHardAndSparse10-v0 --num_episodes=100000 --tie_break='next' --tie_threshold=3.76e-09 --lr_R=0.00021 --save=$(DATA_PATH)/exp140_{1}.pkl --interactive=False --debug=False --seed_value={1}' ::: {1..100}

# beta: exp110 - Close to soln. Not stable. Narrow range?
#   + 'beta': 2.83, 'lr_R': 0.053
exp141:
	parallel -j 40 \
			--joblog '$(DATA_PATH)/exp141.log' \
			--nice 19 --delay 2 --colsep ',' \
			'beta_bandit.py --env_name= -v0 --num_episodes=100000  --lr_R=0.053 --beta=2.83 --save=$(DATA_PATH)/exp141_{1}.pkl --interactive=False --debug=False --seed_value={1}' ::: {1..100}

# softbeta: exp122 - learns the value but needs to high a temp to ever stabilize
#   + 'beta': 0.38, 'lr_R': 0.00971, 'temp': 5.9
exp142:
	parallel -j 40 \
			--joblog '$(DATA_PATH)/exp142.log' \
			--nice 19 --delay 2 --colsep ',' \
			'softbeta_bandit.py --env_name=BanditHardAndSparse10-v0 --num_episodes=100000  --lr_R=0.00971 --beta=0.38 --temp=5.9 --save=$(DATA_PATH)/exp142_{1}.pkl --interactive=False --debug=False --seed_value={1}' ::: {1..100}

# epsilon: exp121 - learns the value, final performance limited by high epsilon
#   + 'epsilon': 0.42, 'lr_R': 0.00043
exp143:
	parallel -j 40 \
			--joblog '$(DATA_PATH)/exp143.log' \
			--nice 19 --delay 2 --colsep ',' \
			'epsilon_bandit.py --env_name=BanditHardAndSparse10-v0 --num_episodes=100000  --lr_R=0.00043 --epsilon=0.42 --save=$(DATA_PATH)/exp143_{1}.pkl --interactive=False --debug=False --seed_value={1}' ::: {1..100}

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
			'epsilon_bandit.py --env_name=BanditOneHigh10-v0 --num_episodes=500  --lr_R=0.11 --epsilon=0.16 --epsilon_decay_tau=0.080 --save=$(DATA_PATH)/exp149_{1}.pkl --interactive=False --debug=False --seed_value={1}' ::: {1..100}

# BanditTwoHigh10
# {'epsilon': 0.838651023382445, 'epsilon_decay_tau': 0.07116057540412388, 'lr_R': 0.1885459873244454, 'total_R': 71.0}
exp150:
	parallel -j 40 \
			--joblog '$(DATA_PATH)/exp150.log' \
			--nice 19 --delay 2 --colsep ',' \
			'epsilon_bandit.py --env_name=BanditTwoHigh10-v0 --num_episodes=500  --lr_R=0.19 --epsilon=0.83 --epsilon_decay_tau=0.071 --save=$(DATA_PATH)/exp150_{1}.pkl --interactive=False --debug=False --seed_value={1}' ::: {1..100}

# BanditUniform121
# {'epsilon': 0.5743595655655118, 'epsilon_decay_tau': 0.03268667798766935, 'lr_R': 0.17235910245007333, 'total_R': 48586.0}
exp151:
	parallel -j 40 \
			--joblog '$(DATA_PATH)/exp151.log' \
			--nice 19 --delay 2 --colsep ',' \
			'epsilon_bandit.py --env_name=BanditUniform121-v0 --num_episodes=120000  --lr_R=0.17 --epsilon=0.57 --epsilon_decay_tau=0.032 --save=$(DATA_PATH)/exp151_{1}.pkl --interactive=False --debug=False --seed_value={1}' ::: {1..100}

# BanditHardAndSparse10
# {'epsilon': 0.7666645365811449, 'epsilon_decay_tau': 0.014058030361594634, 'lr_R': 7.504905974098415e-05, 'total_R': 1029.0}
exp152:
	parallel -j 40 \
			--joblog '$(DATA_PATH)/exp152.log' \
			--nice 19 --delay 2 --colsep ',' \
			'epsilon_bandit.py --env_name=BanditHardAndSparse10-v0 --num_episodes=100000 --lr_R=7.50e-05 --epsilon=0.76 --epsilon_decay_tau=0.014 --save=$(DATA_PATH)/exp152_{1}.pkl --interactive=False --debug=False --seed_value={1}' ::: {1..100}


# ------------------------------------------
# Run exps w/ a random 'learner'. Neg control.
# Fix lr_R at 0.1. No way to opt this.

# BanditOneHigh10
exp153:
	parallel -j 40 \
			--joblog '$(DATA_PATH)/exp153.log' \
			--nice 19 --delay 2 --colsep ',' \
			'random_bandit.py --env_name=BanditOneHigh10-v0 --num_episodes=500  --lr_R=0.1 --save=$(DATA_PATH)/exp153_{1}.pkl --interactive=False --debug=False --seed_value={1}' ::: {1..100}

# BanditTwoHigh10
exp154:
	parallel -j 40 \
			--joblog '$(DATA_PATH)/exp154.log' \
			--nice 19 --delay 2 --colsep ',' \
			'random_bandit.py --env_name=BanditTwoHigh10-v0 --num_episodes=500  --lr_R=0.1 --save=$(DATA_PATH)/exp154_{1}.pkl --interactive=False --debug=False --seed_value={1}' ::: {1..100}

# BanditUniform121
exp155:
	parallel -j 40 \
			--joblog '$(DATA_PATH)/exp155.log' \
			--nice 19 --delay 2 --colsep ',' \
			'random_bandit.py --env_name=BanditUniform121-v0 --num_episodes=120000  --lr_R=0.1 --save=$(DATA_PATH)/exp155_{1}.pkl --interactive=False --debug=False --seed_value={1}' ::: {1..100}

# BanditHardAndSparse10
exp156:
	parallel -j 40 \
			--joblog '$(DATA_PATH)/exp156.log' \
			--nice 19 --delay 2 --colsep ',' \
			'random_bandit.py --env_name=BanditHardAndSparse10-v0 --num_episodes=100000 --lr_R=0.1 --save=$(DATA_PATH)/exp156_{1}.pkl --interactive=False --debug=False --seed_value={1}' ::: {1..100}

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
			'softbeta_bandit.py --env_name=BanditOneHigh10-v0 --num_episodes=500  --lr_R=0.12 --beta=0.045 --temp=0.01 --save=$(DATA_PATH)/exp157_{1}.pkl --interactive=False --debug=False --seed_value={1}' ::: {1..100}

# replicates exp131
exp158:
	parallel -j 40 \
			--joblog '$(DATA_PATH)/exp158.log' \
			--nice 19 --delay 2 --colsep ',' \
			'epsilon_bandit.py --env_name=BanditOneHigh10-v0 --num_episodes=500  --lr_R=0.12 --epsilon=0.078 --save=$(DATA_PATH)/exp158_{1}.pkl --interactive=False --debug=False --seed_value={1}' ::: {1..100}

# -
# BanditTwoHigh10

# replicates exp134
exp159:
	parallel -j 40 \
			--joblog '$(DATA_PATH)/exp159.log' \
			--nice 19 --delay 2 --colsep ',' \
			'softbeta_bandit.py --env_name=BanditTwoHigh10-v0 --num_episodes=500  --lr_R=0.17 --beta=0.010 --temp=0.24 --save=$(DATA_PATH)/exp159_{1}.pkl --interactive=False --debug=False --seed_value={1}' ::: {1..100}


# replicates exp135
exp160:
	parallel -j 40 \
			--joblog '$(DATA_PATH)/exp160.log' \
			--nice 19 --delay 2 --colsep ',' \
			'epsilon_bandit.py --env_name=BanditTwoHigh10-v0 --num_episodes=500  --lr_R=0.12 --epsilon=0.078 --save=$(DATA_PATH)/exp160_{1}.pkl --interactive=False --debug=False --seed_value={1}' ::: {1..100}

# -
# BanditUniform121

# replicates exp138
exp161:
	parallel -j 40 \
			--joblog '$(DATA_PATH)/exp161.log' \
			--nice 19 --delay 2 --colsep ',' \
			'softbeta_bandit.py --env_name=BanditUniform121-v0 --num_episodes=120000  --lr_R=0.097 --beta=0.60 --temp=0.13 --save=$(DATA_PATH)/exp161_{1}.pkl --interactive=False --debug=False --seed_value={1}' ::: {1..100}


# replicates exp139
exp162:
	parallel -j 40 \
			--joblog '$(DATA_PATH)/exp162.log' \
			--nice 19 --delay 2 --colsep ',' \
			'epsilon_bandit.py --env_name=BanditUniform121-v0 --num_episodes=120000  --lr_R=0.11 --epsilon=0.012 --save=$(DATA_PATH)/exp162_{1}.pkl --interactive=False --debug=False --seed_value={1}' ::: {1..100}

# -
# HardAndSparse

# replicates exp142
exp163:
	parallel -j 40 \
			--joblog '$(DATA_PATH)/exp163.log' \
			--nice 19 --delay 2 --colsep ',' \
			'softbeta_bandit.py --env_name=BanditHardAndSparse10-v0 --num_episodes=100000  --lr_R=0.00971 --beta=0.38 --temp=5.9 --save=$(DATA_PATH)/exp163_{1}.pkl --interactive=False --debug=False --seed_value={1}' ::: {1..100}

# replicates exp143
exp164:
	parallel -j 40 \
			--joblog '$(DATA_PATH)/exp164.log' \
			--nice 19 --delay 2 --colsep ',' \
			'epsilon_bandit.py --env_name=BanditHardAndSparse10-v0 --num_episodes=100000  --lr_R=0.00043 --epsilon=0.42 --save=$(DATA_PATH)/exp164_{1}.pkl --interactive=False --debug=False --seed_value={1}' ::: {1..100}

# -
# ep-decay experiments (all bandits)

# BanditOneHigh10
# replicates exp149
exp165:
	parallel -j 40 \
			--joblog '$(DATA_PATH)/exp165.log' \
			--nice 19 --delay 2 --colsep ',' \
			'epsilon_bandit.py --env_name=BanditOneHigh10-v0 --num_episodes=500  --lr_R=0.11 --epsilon=0.16 --epsilon_decay_tau=0.080 --save=$(DATA_PATH)/exp165_{1}.pkl --interactive=False --debug=False --seed_value={1}' ::: {1..100}


# BanditTwoHigh10
# replicates exp150
exp166:
	parallel -j 40 \
			--joblog '$(DATA_PATH)/exp166.log' \
			--nice 19 --delay 2 --colsep ',' \
			'epsilon_bandit.py --env_name=BanditTwoHigh10-v0 --num_episodes=500  --lr_R=0.19 --epsilon=0.83 --epsilon_decay_tau=0.071 --save=$(DATA_PATH)/exp166_{1}.pkl --interactive=False --debug=False --seed_value={1}' ::: {1..100}

# BanditUniform121
# replicates exp151
exp167:
	parallel -j 40 \
			--joblog '$(DATA_PATH)/exp167.log' \
			--nice 19 --delay 2 --colsep ',' \
			'epsilon_bandit.py --env_name=BanditUniform121-v0 --num_episodes=120000  --lr_R=0.17 --epsilon=0.57 --epsilon_decay_tau=0.032 --save=$(DATA_PATH)/exp167_{1}.pkl --interactive=False --debug=False --seed_value={1}' ::: {1..100}

# BanditHardAndSparse10
# replicates exp152
exp168:
	parallel -j 40 \
			--joblog '$(DATA_PATH)/exp168.log' \
			--nice 19 --delay 2 --colsep ',' \
			'epsilon_bandit.py --env_name=BanditHardAndSparse10-v0 --num_episodes=100000 --lr_R=7.50e-05 --epsilon=0.76 --epsilon_decay_tau=0.014 --save=$(DATA_PATH)/exp168_{1}.pkl --interactive=False --debug=False --seed_value={1}' ::: {1..100}

# -
# Random

# BanditOneHigh10
# replicates exp153
exp169:
	parallel -j 40 \
			--joblog '$(DATA_PATH)/exp169.log' \
			--nice 19 --delay 2 --colsep ',' \
			'random_bandit.py --env_name=BanditOneHigh10-v0 --num_episodes=500  --lr_R=0.1 --save=$(DATA_PATH)/exp169_{1}.pkl --interactive=False --debug=False --seed_value={1}' ::: {1..100}

# BanditTwoHigh10
# replicates exp154
exp170:
	parallel -j 40 \
			--joblog '$(DATA_PATH)/exp170.log' \
			--nice 19 --delay 2 --colsep ',' \
			'random_bandit.py --env_name=BanditTwoHigh10-v0 --num_episodes=500  --lr_R=0.1 --save=$(DATA_PATH)/exp170_{1}.pkl --interactive=False --debug=False --seed_value={1}' ::: {1..100}

# BanditUniform121
# replicates exp155
exp171:
	parallel -j 40 \
			--joblog '$(DATA_PATH)/exp171.log' \
			--nice 19 --delay 2 --colsep ',' \
			'random_bandit.py --env_name=BanditUniform121-v0 --num_episodes=120000  --lr_R=0.1 --save=$(DATA_PATH)/exp171_{1}.pkl --interactive=False --debug=False --seed_value={1}' ::: {1..100}

# BanditHardAndSparse10
# replicates exp156
exp172:
	parallel -j 40 \
			--joblog '$(DATA_PATH)/exp172.log' \
			--nice 19 --delay 2 --colsep ',' \
			'random_bandit.py --env_name=BanditHardAndSparse10-v0 --num_episodes=100000 --lr_R=0.1 --save=$(DATA_PATH)/exp17_{1}.pkl --interactive=False --debug=False --seed_value={1}' ::: {1..100}



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
	# seed_value=None,
	# **config_kwargs)

# -
# eta

# BanditOneHigh10
exp173:
	tune_bandit.py replicator $(DATA_PATH)/exp173 \
		--exp_name='meta_bandit' \
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
		--exp_name='meta_bandit' \
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
		--exp_name='meta_bandit' \
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
		--exp_name='meta_bandit' \
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
			'meta_bandit.py --env_name=BanditOneHigh10-v0 --num_episodes=500 --tie_break='next' --tie_threshold=0.053 --lr_R=0.34 --save=$(DATA_PATH)/exp194_{1}.pkl --interactive=False --debug=False --seed_value={1}' ::: {1..100}

# BanditTwoHigh10
exp195:
	parallel -j 40 \
			--joblog '$(DATA_PATH)/exp195.log' \
			--nice 19 --delay 2 --colsep ',' \
			'meta_bandit.py --env_name=BanditTwoHigh10-v0 --num_episodes=500 --tie_break='next' --tie_threshold=0.0169 --lr_R=0.161 --save=$(DATA_PATH)/exp195_{1}.pkl --interactive=False --debug=False --seed_value={1}' ::: {1..100}

# BanditUniform121
exp196:
	parallel -j 40 \
			--joblog '$(DATA_PATH)/exp196.log' \
			--nice 19 --delay 2 --colsep ',' \
			'meta_bandit.py --env_name=BanditUniform121-v0 --num_episodes=60500 --tie_break='next' --tie_threshold=0.00355 --lr_R=0.147 --save=$(DATA_PATH)/exp196_{1}.pkl --interactive=False --debug=False --seed_value={1}' ::: {1..100}

# HardAndSparse10
exp197:
	parallel -j 40 \
			--joblog '$(DATA_PATH)/exp197.log' \
			--nice 19 --delay 2 --colsep ',' \
			'meta_bandit.py --env_name=BanditHardAndSparse10-v0 --num_episodes=50000 --tie_break='next' --tie_threshold=5.782e-09 --lr_R=0.00112 --save=$(DATA_PATH)/exp197_{1}.pkl --interactive=False --debug=False --seed_value={1}' ::: {1..100}

# -
# beta:
# BanditOneHigh10
exp198:
	parallel -j 40 \
			--joblog '$(DATA_PATH)/exp198.log' \
			--nice 19 --delay 2 --colsep ',' \
			'beta_bandit.py --env_name=BanditOneHigh10-v0 --num_episodes=500 --tie_break='next' --beta=0.22 --lr_R=0.18 --save=$(DATA_PATH)/exp198_{1}.pkl --interactive=False --debug=False --seed_value={1}' ::: {1..100}

# BanditTwoHigh10
exp199:
	parallel -j 40 \
			--joblog '$(DATA_PATH)/exp199.log' \
			--nice 19 --delay 2 --colsep ',' \
			'beta_bandit.py --env_name=BanditTwoHigh10-v0 --num_episodes=500 --tie_break='next' --beta=0.188 --lr_R=0.129 --save=$(DATA_PATH)/exp199_{1}.pkl --interactive=False --debug=False --seed_value={1}' ::: {1..100}

# BanditUniform121
exp200:
	parallel -j 40 \
			--joblog '$(DATA_PATH)/exp200.log' \
			--nice 19 --delay 2 --colsep ',' \
			'beta_bandit.py --env_name=BanditUniform121-v0 --num_episodes=60500 --tie_break='next' --beta=0.056 --lr_R=0.141 --save=$(DATA_PATH)/exp200_{1}.pkl --interactive=False --debug=False --seed_value={1}' ::: {1..100}

# HardAndSparse10
exp201:
	parallel -j 40 \
			--joblog '$(DATA_PATH)/exp201.log' \
			--nice 19 --delay 2 --colsep ',' \
			'beta_bandit.py --env_name=BanditHardAndSparse10-v0 --num_episodes=50000 --tie_break='next' --beta=0.217 --lr_R=0.051 --save=$(DATA_PATH)/exp201_{1}.pkl --interactive=False --debug=False --seed_value={1}' ::: {1..100}

# -
# softbeta
# BanditOneHigh10
exp202:
	parallel -j 40 \
			--joblog '$(DATA_PATH)/exp202.log' \
			--nice 19 --delay 2 --colsep ',' \
			'softbeta_bandit.py --env_name=BanditOneHigh10-v0 --num_episodes=500 --beta=0.066 --lr_R=0.13 --temp=0.13 --save=$(DATA_PATH)/exp202_{1}.pkl --interactive=False --debug=False --seed_value={1}' ::: {1..100}

# BanditTwoHigh10
exp203:
	parallel -j 40 \
			--joblog '$(DATA_PATH)/exp203.log' \
			--nice 19 --delay 2 --colsep ',' \
			'softbeta_bandit.py --env_name=BanditTwoHigh10-v0 --num_episodes=500 --beta=0.133 --lr_R=0.030 --temp=0.098 --save=$(DATA_PATH)/exp203_{1}.pkl --interactive=False --debug=False --seed_value={1}' ::: {1..100}

# BanditUniform121
exp204:
	parallel -j 40 \
			--joblog '$(DATA_PATH)/exp204.log' \
			--nice 19 --delay 2 --colsep ',' \
			'softbeta_bandit.py --env_name=BanditUniform121-v0 --num_episodes=60500 --beta=0.125 --lr_R=0.174 --temp=0.0811 --save=$(DATA_PATH)/exp204_{1}.pkl --interactive=False --debug=False --seed_value={1}' ::: {1..100}

# HardAndSparse10
exp205:
	parallel -j 40 \
			--joblog '$(DATA_PATH)/exp205.log' \
			--nice 19 --delay 2 --colsep ',' \
			'softbeta_bandit.py --env_name=BanditHardAndSparse10-v0 --num_episodes=50000 --beta=2.140 --lr_R=0.128 --temp=5.045 --save=$(DATA_PATH)/exp205_{1}.pkl --interactive=False --debug=False --seed_value={1}' ::: {1..100}

# -
# ep

# BanditOneHigh10
exp206:
	parallel -j 40 \
			--joblog '$(DATA_PATH)/exp206.log' \
			--nice 19 --delay 2 --colsep ',' \
			'epsilon_bandit.py --env_name=BanditOneHigh10-v0 --num_episodes=500 --epsilon=0.14 --lr_R=0.087 --save=$(DATA_PATH)/exp206_{1}.pkl --interactive=False --debug=False --seed_value={1}' ::: {1..100}

# BanditTwoHigh10
exp207:
	parallel -j 40 \
			--joblog '$(DATA_PATH)/exp207.log' \
			--nice 19 --delay 2 --colsep ',' \
		'epsilon_bandit.py --env_name=BanditTwoHigh10-v0 --num_episodes=500 --epsilon=0.087 --lr_R=0.08583 --save=$(DATA_PATH)/exp207_{1}.pkl --interactive=False --debug=False --seed_value={1}' ::: {1..100}

# BanditUniform121
exp208:
	parallel -j 40 \
			--joblog '$(DATA_PATH)/exp208.log' \
			--nice 19 --delay 2 --colsep ',' \
			'epsilon_bandit.py --env_name=BanditUniform121-v0 --num_episodes=60500 --epsilon=0.0117 --lr_R=0.137 --save=$(DATA_PATH)/exp208_{1}.pkl --interactive=False --debug=False --seed_value={1}' ::: {1..100}

# HardAndSparse10
exp209:
	parallel -j 40 \
			--joblog '$(DATA_PATH)/exp209.log' \
			--nice 19 --delay 2 --colsep ',' \
			'epsilon_bandit.py --env_name=BanditHardAndSparse10-v0 --num_episodes=50000 --epsilon=0.4057 --lr_R=0.000484 --save=$(DATA_PATH)/exp209_{1}.pkl --interactive=False --debug=False --seed_value={1}' ::: {1..100}

# -
# anneal-ep

# BanditOneHigh10
exp210:
	parallel -j 40 \
			--joblog '$(DATA_PATH)/exp210.log' \
			--nice 19 --delay 2 --colsep ',' \
			'epsilon_bandit.py --env_name=BanditOneHigh10-v0 --num_episodes=500 --epsilon=0.45 --epsilon_decay_tau=0.061 --lr_R=0.14 --save=$(DATA_PATH)/exp210_{1}.pkl --interactive=False --debug=False --seed_value={1}' ::: {1..100}

# BanditTwoHigh10
exp211:
	parallel -j 40 \
			--joblog '$(DATA_PATH)/exp211.log' \
			--nice 19 --delay 2 --colsep ',' \
			'epsilon_bandit.py --env_name=BanditTwoHigh10-v0 --num_episodes=500 --epsilon=0.980 --epsilon_decay_tau=0.084 --lr_R=0.194  --save=$(DATA_PATH)/exp211_{1}.pkl --interactive=False --debug=False --seed_value={1}' ::: {1..100}

# BanditUniform121
exp212:
	parallel -j 40 \
			--joblog '$(DATA_PATH)/exp212.log' \
			--nice 19 --delay 2 --colsep ',' \
			'epsilon_bandit.py --env_name=BanditUniform121-v0 --num_episodes=60500 --epsilon=0.850 --lr_R=0.173 --epsilon_decay_tau=0.00777 --save=$(DATA_PATH)/exp212_{1}.pkl --interactive=False --debug=False --seed_value={1}' ::: {1..100}

# HardAndSparse10
exp213:
	parallel -j 40 \
			--joblog '$(DATA_PATH)/exp213.log' \
			--nice 19 --delay 2 --colsep ',' \
			'epsilon_bandit.py --env_name=BanditHardAndSparse10-v0 --num_episodes=50000 --epsilon=0.5148 --epsilon_decay_tau=0.0723 --lr_R=0.000271 --save=$(DATA_PATH)/exp213_{1}.pkl --interactive=False --debug=False --seed_value={1}' ::: {1..100}
			

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
			'meta_bandit.py --env_name=DeceptiveBanditOneHigh10-v0 --num_episodes=500 --tie_break='next' --tie_threshold=0.053 --lr_R=0.34 --save=$(DATA_PATH)/exp214_{1}.pkl --interactive=False --debug=False --seed_value={1}' ::: {1..100}

exp215:
	parallel -j 40 \
			--joblog '$(DATA_PATH)/exp215.log' \
			--nice 19 --delay 2 --colsep ',' \
			'softbeta_bandit.py --env_name=DeceptiveBanditOneHigh10-v0 --num_episodes=500 --beta=0.066 --lr_R=0.13 --temp=0.13 --save=$(DATA_PATH)/exp215_{1}.pkl --interactive=False --debug=False --seed_value={1}' ::: {1..100}

exp216:
	parallel -j 40 \
			--joblog '$(DATA_PATH)/exp216.log' \
			--nice 19 --delay 2 --colsep ',' \
			'epsilon_bandit.py --env_name=DeceptiveBanditOneHigh10-v0 --num_episodes=500 --epsilon=0.14 --lr_R=0.087 --save=$(DATA_PATH)/exp216_{1}.pkl --interactive=False --debug=False --seed_value={1}' ::: {1..100}

exp217:
	parallel -j 40 \
			--joblog '$(DATA_PATH)/exp217.log' \
			--nice 19 --delay 2 --colsep ',' \
			'epsilon_bandit.py --env_name=DeceptiveBanditOneHigh10-v0 --num_episodes=500 --epsilon=0.45 --epsilon_decay_tau=0.061 --lr_R=0.14 --save=$(DATA_PATH)/exp217_{1}.pkl --interactive=False --debug=False --seed_value={1}' ::: {1..100}


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
			'meta_bandit.py --env_name=DeceptiveBanditOneHigh10-v0 --num_episodes=20 --tie_break='next' --tie_threshold=0.053 --lr_R=0.34 --save=$(DATA_PATH)/exp218_{1}.pkl --interactive=False --debug=False --seed_value={1}' ::: {1..100}

exp219:
	parallel -j 40 \
			--joblog '$(DATA_PATH)/exp219.log' \
			--nice 19 --delay 2 --colsep ',' \
			'softbeta_bandit.py --env_name=DeceptiveBanditOneHigh10-v0 --num_episodes=20 --beta=0.066 --lr_R=0.13 --temp=0.13 --save=$(DATA_PATH)/exp219_{1}.pkl --interactive=False --debug=False --seed_value={1}' ::: {1..100}

exp220:
	parallel -j 40 \
			--joblog '$(DATA_PATH)/exp220.log' \
			--nice 19 --delay 2 --colsep ',' \
			'epsilon_bandit.py --env_name=DeceptiveBanditOneHigh10-v0 --num_episodes=20 --epsilon=0.14 --lr_R=0.087 --save=$(DATA_PATH)/exp220_{1}.pkl --interactive=False --debug=False --seed_value={1}' ::: {1..100}

exp221:
	parallel -j 40 \
			--joblog '$(DATA_PATH)/exp221.log' \
			--nice 19 --delay 2 --colsep ',' \
			'epsilon_bandit.py --env_name=DeceptiveBanditOneHigh10-v0 --num_episodes=20 --epsilon=0.45 --epsilon_decay_tau=0.061 --lr_R=0.14 --save=$(DATA_PATH)/exp221_{1}.pkl --interactive=False --debug=False --seed_value={1}' ::: {1..100}


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
			'meta_bandit.py --env_name=DeceptiveBanditOneHigh10-v0 --num_episodes=50 --tie_break='next' --tie_threshold=0.053 --lr_R=0.34 --save=$(DATA_PATH)/exp222_{1}.pkl --interactive=False --debug=False --seed_value={1}' ::: {1..100}

exp223:
	parallel -j 40 \
			--joblog '$(DATA_PATH)/exp223.log' \
			--nice 19 --delay 2 --colsep ',' \
			'softbeta_bandit.py --env_name=DeceptiveBanditOneHigh10-v0 --num_episodes=50 --beta=0.066 --lr_R=0.13 --temp=0.13 --save=$(DATA_PATH)/exp223_{1}.pkl --interactive=False --debug=False --seed_value={1}' ::: {1..100}

exp224:
	parallel -j 40 \
			--joblog '$(DATA_PATH)/exp224.log' \
			--nice 19 --delay 2 --colsep ',' \
			'epsilon_bandit.py --env_name=DeceptiveBanditOneHigh10-v0 --num_episodes=50 --epsilon=0.14 --lr_R=0.087 --save=$(DATA_PATH)/exp224_{1}.pkl --interactive=False --debug=False --seed_value={1}' ::: {1..100}

exp225:
	parallel -j 40 \
			--joblog '$(DATA_PATH)/exp225.log' \
			--nice 19 --delay 2 --colsep ',' \
			'epsilon_bandit.py --env_name=DeceptiveBanditOneHigh10-v0 --num_episodes=50 --epsilon=0.45 --epsilon_decay_tau=0.061 --lr_R=0.14 --save=$(DATA_PATH)/exp225_{1}.pkl --interactive=False --debug=False --seed_value={1}' ::: {1..100}

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
			'meta_bandit.py --env_name=DeceptiveBanditOneHigh10-v0 --num_episodes=100 --tie_break='next' --tie_threshold=0.053 --lr_R=0.34 --save=$(DATA_PATH)/exp226_{1}.pkl --interactive=False --debug=False --seed_value={1}' ::: {1..100}

exp227:
	parallel -j 40 \
			--joblog '$(DATA_PATH)/exp227.log' \
			--nice 19 --delay 2 --colsep ',' \
			'softbeta_bandit.py --env_name=DeceptiveBanditOneHigh10-v0 --num_episodes=100 --beta=0.066 --lr_R=0.13 --temp=0.13 --save=$(DATA_PATH)/exp227_{1}.pkl --interactive=False --debug=False --seed_value={1}' ::: {1..100}

exp228:
	parallel -j 40 \
			--joblog '$(DATA_PATH)/exp228.log' \
			--nice 19 --delay 2 --colsep ',' \
			'epsilon_bandit.py --env_name=DeceptiveBanditOneHigh10-v0 --num_episodes=100 --epsilon=0.14 --lr_R=0.087 --save=$(DATA_PATH)/exp228_{1}.pkl --interactive=False --debug=False --seed_value={1}' ::: {1..100}

exp229:
	parallel -j 40 \
			--joblog '$(DATA_PATH)/exp229.log' \
			--nice 19 --delay 2 --colsep ',' \
			'epsilon_bandit.py --env_name=DeceptiveBanditOneHigh10-v0 --num_episodes=100 --epsilon=0.45 --epsilon_decay_tau=0.061 --lr_R=0.14 --save=$(DATA_PATH)/exp229_{1}.pkl --interactive=False --debug=False --seed_value={1}' ::: {1..100}

# ----------------------------------------------------------------------------
# 2-15-202
# cbc1dac28a486aa2360d9fcd5a201fbb603f9bd2
#
# Replicator HP tuning - round 1 - DeceptiveBanditOneHigh10

exp230:
	tune_bandit.py replicator $(DATA_PATH)/exp230 \
		--exp_name='meta_bandit' \
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
		--exp_name='meta_bandit' \
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
			'meta_bandit.py --env_name=BanditOneHigh10-v0 --num_episodes=500 --tie_break='next' --tie_threshold=0.053 --lr_R=0.34 --save=$(DATA_PATH)/exp238_{1}.pkl --interactive=False --debug=False --seed_value={1}' ::: {1..100}

# BanditUniform121
exp239:
	parallel -j 40 \
			--joblog '$(DATA_PATH)/exp239.log' \
			--nice 19 --delay 2 --colsep ',' \
			'meta_bandit.py --env_name=BanditUniform121-v0 --num_episodes=60500 --tie_break='next' --tie_threshold=0.00355 --lr_R=0.147 --save=$(DATA_PATH)/exp239_{1}.pkl --interactive=False --debug=False --seed_value={1}' ::: {1..100}

# HardAndSparse10
exp240:
	parallel -j 40 \
			--joblog '$(DATA_PATH)/exp240.log' \
			--nice 19 --delay 2 --colsep ',' \
			'meta_bandit.py --env_name=BanditHardAndSparse10-v0 --num_episodes=50000 --tie_break='next' --tie_threshold=5.782e-09 --lr_R=0.00112 --save=$(DATA_PATH)/exp240_{1}.pkl --interactive=False --debug=False --seed_value={1}' ::: {1..100}

# -
# beta:
# BanditOneHigh10
exp241:
	parallel -j 40 \
			--joblog '$(DATA_PATH)/exp241.log' \
			--nice 19 --delay 2 --colsep ',' \
			'beta_bandit.py --env_name=BanditOneHigh10-v0 --num_episodes=500 --tie_break='next' --beta=0.22 --lr_R=0.18 --save=$(DATA_PATH)/exp241_{1}.pkl --interactive=False --debug=False --seed_value={1}' ::: {1..100}

# BanditUniform121
exp242:
	parallel -j 40 \
			--joblog '$(DATA_PATH)/exp242.log' \
			--nice 19 --delay 2 --colsep ',' \
			'beta_bandit.py --env_name=BanditUniform121-v0 --num_episodes=60500 --tie_break='next' --beta=0.056 --lr_R=0.141 --save=$(DATA_PATH)/exp242_{1}.pkl --interactive=False --debug=False --seed_value={1}' ::: {1..100}

# HardAndSparse10
exp243:
	parallel -j 40 \
			--joblog '$(DATA_PATH)/exp243.log' \
			--nice 19 --delay 2 --colsep ',' \
			'beta_bandit.py --env_name=BanditHardAndSparse10-v0 --num_episodes=50000 --tie_break='next' --beta=0.217 --lr_R=0.051 --save=$(DATA_PATH)/exp243_{1}.pkl --interactive=False --debug=False --seed_value={1}' ::: {1..100}

# -
# softbeta
# BanditOneHigh10
exp244:
	parallel -j 40 \
			--joblog '$(DATA_PATH)/exp244.log' \
			--nice 19 --delay 2 --colsep ',' \
			'softbeta_bandit.py --env_name=BanditOneHigh10-v0 --num_episodes=500 --beta=0.066 --lr_R=0.13 --temp=0.13 --save=$(DATA_PATH)/exp244_{1}.pkl --interactive=False --debug=False --seed_value={1}' ::: {1..100}

# BanditUniform121
exp245:
	parallel -j 40 \
			--joblog '$(DATA_PATH)/exp245.log' \
			--nice 19 --delay 2 --colsep ',' \
			'softbeta_bandit.py --env_name=BanditUniform121-v0 --num_episodes=60500 --beta=0.125 --lr_R=0.174 --temp=0.0811 --save=$(DATA_PATH)/exp245_{1}.pkl --interactive=False --debug=False --seed_value={1}' ::: {1..100}

# HardAndSparse10
exp246:
	parallel -j 40 \
			--joblog '$(DATA_PATH)/exp246.log' \
			--nice 19 --delay 2 --colsep ',' \
			'softbeta_bandit.py --env_name=BanditHardAndSparse10-v0 --num_episodes=50000 --beta=2.140 --lr_R=0.128 --temp=5.045 --save=$(DATA_PATH)/exp246_{1}.pkl --interactive=False --debug=False --seed_value={1}' ::: {1..100}

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
			'epsilon_bandit.py --env_name=BanditOneHigh10-v0 --num_episodes=500 --epsilon=0.14 --lr_R=0.087 --save=$(DATA_PATH)/exp247_{1}.pkl --interactive=False --debug=False --seed_value={1}' ::: {1..100}

# BanditUniform121
exp248:
	parallel -j 40 \
			--joblog '$(DATA_PATH)/exp248.log' \
			--nice 19 --delay 2 --colsep ',' \
			'epsilon_bandit.py --env_name=BanditUniform121-v0 --num_episodes=60500 --epsilon=0.0117 --lr_R=0.137 --save=$(DATA_PATH)/exp248_{1}.pkl --interactive=False --debug=False --seed_value={1}' ::: {1..100}

# HardAndSparse10
exp249:
	parallel -j 40 \
			--joblog '$(DATA_PATH)/exp249.log' \
			--nice 19 --delay 2 --colsep ',' \
			'epsilon_bandit.py --env_name=BanditHardAndSparse10-v0 --num_episodes=50000 --epsilon=0.4057 --lr_R=0.000484 --save=$(DATA_PATH)/exp249_{1}.pkl --interactive=False --debug=False --seed_value={1}' ::: {1..100}

# -
# anneal-ep

# BanditOneHigh10
exp250:
	parallel -j 40 \
			--joblog '$(DATA_PATH)/exp250.log' \
			--nice 19 --delay 2 --colsep ',' \
			'epsilon_bandit.py --env_name=BanditOneHigh10-v0 --num_episodes=500 --epsilon=0.45 --epsilon_decay_tau=0.061 --lr_R=0.14 --save=$(DATA_PATH)/exp250_{1}.pkl --interactive=False --debug=False --seed_value={1}' ::: {1..100}

# BanditUniform121
exp251:
	parallel -j 40 \
			--joblog '$(DATA_PATH)/exp251.log' \
			--nice 19 --delay 2 --colsep ',' \
			'epsilon_bandit.py --env_name=BanditUniform121-v0 --num_episodes=60500 --epsilon=0.850 --lr_R=0.173 --epsilon_decay_tau=0.00777 --save=$(DATA_PATH)/exp251_{1}.pkl --interactive=False --debug=False --seed_value={1}' ::: {1..100}

# HardAndSparse10
exp252:
	parallel -j 40 \
			--joblog '$(DATA_PATH)/exp252.log' \
			--nice 19 --delay 2 --colsep ',' \
			'epsilon_bandit.py --env_name=BanditHardAndSparse10-v0 --num_episodes=50000 --epsilon=0.5148 --epsilon_decay_tau=0.0723 --lr_R=0.000271 --save=$(DATA_PATH)/exp252_{1}.pkl --interactive=False --debug=False --seed_value={1}' ::: {1..100}
			
# -
# Random
# BanditOneHigh10
exp253:
	parallel -j 40 \
			--joblog '$(DATA_PATH)/exp253.log' \
			--nice 19 --delay 2 --colsep ',' \
			'random_bandit.py --env_name=BanditOneHigh10-v0 --num_episodes=500  --lr_R=0.1 --save=$(DATA_PATH)/exp253_{1}.pkl --interactive=False --debug=False --seed_value={1}' ::: {1..100}

# BanditUniform121
exp254:
	parallel -j 40 \
			--joblog '$(DATA_PATH)/exp254.log' \
			--nice 19 --delay 2 --colsep ',' \
			'random_bandit.py --env_name=BanditUniform121-v0 --num_episodes=60500  --lr_R=0.1 --save=$(DATA_PATH)/exp254_{1}.pkl --interactive=False --debug=False --seed_value={1}' ::: {1..100}

# BanditHardAndSparse10
exp255:
	parallel -j 40 \
			--joblog '$(DATA_PATH)/exp255.log' \
			--nice 19 --delay 2 --colsep ',' \
			'random_bandit.py --env_name=BanditHardAndSparse10-v0 --num_episodes=50000 --lr_R=0.1 --save=$(DATA_PATH)/exp255_{1}.pkl --interactive=False --debug=False --seed_value={1}' ::: {1..100}


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
		--exp_name='meta_bandit' \
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
		--exp_name='meta_bandit' \
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
		--exp_name='meta_bandit' \
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
			'meta_bandit.py --env_name=DeceptiveBanditOneHigh10-v0 --num_episodes=100 --tie_break='next' --tie_threshold=0.0014 --lr_R=0.35 --save=$(DATA_PATH)/exp269_{1}.pkl --interactive=False --debug=False --seed_value={1}' ::: {1..100}

# ep - sample results from exp266 best_params
exp270:
	parallel -j 40 \
			--joblog '$(DATA_PATH)/exp270.log' \
			--nice 19 --delay 2 --colsep ',' \
			'epsilon_bandit.py --env_name=DeceptiveBanditOneHigh10-v0 --num_episodes=100 --epsilon=0.011 --lr_R=0.115 --save=$(DATA_PATH)/exp270_{1}.pkl --interactive=False --debug=False --seed_value={1}' ::: {1..100}


# anneal - sample results from exp267 best_params
exp271:
	parallel -j 40 \
			--joblog '$(DATA_PATH)/exp271.log' \
			--nice 19 --delay 2 --colsep ',' \
			'epsilon_bandit.py --env_name=DeceptiveBanditOneHigh10-v0 --num_episodes=100 --epsilon=0.080 --epsilon_decay_tau=0.039 --lr_R=0.10 --save=$(DATA_PATH)/exp271_{1}.pkl --interactive=False --debug=False --seed_value={1}' ::: {1..100}

# softbeta - sample results from exp265 best_params
exp272:
	parallel -j 40 \
			--joblog '$(DATA_PATH)/exp272.log' \
			--nice 19 --delay 2 --colsep ',' \
			'softbeta_bandit.py --env_name=DeceptiveBanditOneHigh10-v0 --num_episodes=100 --beta=0.31 --lr_R=0.18 --temp=0.082 --save=$(DATA_PATH)/exp272_{1}.pkl --interactive=False --debug=False --seed_value={1}' ::: {1..100}

# random - just run 100. (There is nothing to tune.)
exp273:
	parallel -j 40 \
			--joblog '$(DATA_PATH)/exp273.log' \
			--nice 19 --delay 2 --colsep ',' \
			'random_bandit.py --env_name=DeceptiveBanditOneHigh10-v0 --num_episodes=100  --lr_R=0.1 --save=$(DATA_PATH)/exp273_{1}.pkl --interactive=False --debug=False --seed_value={1}' ::: {1..100}
		

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
		--exp_name='meta_bandit' \
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
		--exp_name='meta_bandit' \
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
		--exp_name='meta_bandit' \
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
		--exp_name='meta_bandit' \
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
		--exp_name='meta_bandit' \
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
		--exp_name='meta_bandit' \
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
		--exp_name='meta_bandit' \
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
		--exp_name='meta_bandit' \
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
			'meta_bandit.py --env_name=BanditOneHigh10-v0 --num_episodes=100 --tie_break='next' --tie_threshold={tie_threshold} --lr_R={lr_R} --save=$(DATA_PATH)/exp310_{index}_{1}.pkl --interactive=False --debug=False --seed_value={1}' ::: {0..10} :::: tmp
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
			'epsilon_bandit.py --env_name=BanditOneHigh10-v0 --num_episodes=100 --epsilon={epsilon} --lr_R={lr_R} --save=$(DATA_PATH)/exp311_{index}_{1}.pkl --interactive=False --debug=False --seed_value={1}' ::: {0..10} :::: tmp
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
			'epsilon_bandit.py --env_name=BanditOneHigh10-v0 --num_episodes=100 --epsilon={epsilon} --epsilon_decay_tau={epsilon_decay_tau} --lr_R={lr_R} --save=$(DATA_PATH)/exp312_{index}_{1}.pkl --interactive=False --debug=False --seed_value={1}' ::: {0..10} :::: tmp
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
			'softbeta_bandit.py --env_name=BanditOneHigh10-v0 --num_episodes=100 --beta={beta} --temp={temp} --lr_R={lr_R} --save=$(DATA_PATH)/exp313_{index}_{1}.pkl --interactive=False --debug=False --seed_value={1}' ::: {0..10} :::: tmp
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
			'meta_bandit.py --env_name=BanditHardAndSparse10-v0 --num_episodes=50000 --tie_break='next' --tie_threshold={tie_threshold} --lr_R={lr_R} --save=$(DATA_PATH)/exp314_{index}_{1}.pkl --interactive=False --debug=False --seed_value={1}' ::: {0..10} :::: tmp
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
			'epsilon_bandit.py --env_name=BanditHardAndSparse10-v0 --num_episodes=50000 --epsilon={epsilon} --lr_R={lr_R} --save=$(DATA_PATH)/exp315_{index}_{1}.pkl --interactive=False --debug=False --seed_value={1}' ::: {0..10} :::: tmp
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
			'epsilon_bandit.py --env_name=BanditHardAndSparse10-v0 --num_episodes=50000 --epsilon={epsilon} --epsilon_decay_tau={epsilon_decay_tau} --lr_R={lr_R} --save=$(DATA_PATH)/exp316_{index}_{1}.pkl --interactive=False --debug=False --seed_value={1}' ::: {0..10} :::: tmp
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
			'softbeta_bandit.py --env_name=BanditHardAndSparse10-v0 --num_episodes=50000 --beta={beta} --temp={temp} --lr_R={lr_R} --save=$(DATA_PATH)/exp317_{index}_{1}.pkl --interactive=False --debug=False --seed_value={1}' ::: {0..10} :::: tmp
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
			'meta_bandit.py --env_name=BanditUniform121-v0 --num_episodes=60500 --tie_break='next' --tie_threshold={tie_threshold} --lr_R={lr_R} --save=$(DATA_PATH)/exp318_{index}_{1}.pkl --interactive=False --debug=False --seed_value={1}' ::: {0..10} :::: tmp
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
			'epsilon_bandit.py --env_name=BanditUniform121-v0 --num_episodes=60500 --epsilon={epsilon} --lr_R={lr_R} --save=$(DATA_PATH)/exp319_{index}_{1}.pkl --interactive=False --debug=False --seed_value={1}' ::: {0..10} :::: tmp
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
			'epsilon_bandit.py --env_name=BanditUniform121-v0 --num_episodes=60500 --epsilon={epsilon} --epsilon_decay_tau={epsilon_decay_tau} --lr_R={lr_R} --save=$(DATA_PATH)/exp320_{index}_{1}.pkl --interactive=False --debug=False --seed_value={1}' ::: {0..10} :::: tmp
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
			'softbeta_bandit.py --env_name=BanditUniform121-v0 --num_episodes=60500 --temp={temp} --beta={beta} --lr_R={lr_R} --save=$(DATA_PATH)/exp321_{index}_{1}.pkl --interactive=False --debug=False --seed_value={1}' ::: {0..10} :::: tmp
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
			'meta_bandit.py --env_name=DeceptiveBanditOneHigh10-v0 --num_episodes=100 --tie_break='next' --tie_threshold={tie_threshold} --lr_R={lr_R} --save=$(DATA_PATH)/exp322_{index}_{1}.pkl --interactive=False --debug=False --seed_value={1}' ::: {0..10} :::: tmp
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
			'epsilon_bandit.py --env_name=DeceptiveBanditOneHigh10-v0 --num_episodes=100 --epsilon={epsilon} --lr_R={lr_R} --save=$(DATA_PATH)/exp323_{index}_{1}.pkl --interactive=False --debug=False --seed_value={1}' ::: {0..10} :::: tmp
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
			'epsilon_bandit.py --env_name=DeceptiveBanditOneHigh10-v0 --num_episodes=100 --epsilon={epsilon} --epsilon_decay_tau={epsilon_decay_tau} --lr_R={lr_R} --save=$(DATA_PATH)/exp324_{index}_{1}.pkl --interactive=False --debug=False --seed_value={1}' ::: {0..10} :::: tmp
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
			'softbeta_bandit.py --env_name=DeceptiveBanditOneHigh10-v0 --num_episodes=100 --temp={temp} --beta={beta} --lr_R={lr_R} --save=$(DATA_PATH)/exp325_{index}_{1}.pkl --interactive=False --debug=False --seed_value={1}' ::: {0..10} :::: tmp
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
        --seed_value=42 \
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
        --seed_value=42 \
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
        --seed_value=127 \
        --reward_mode=False \
        --log_dir=$(DATA_PATH)/exp328/run1/
	curiosity_bandit.py \
		--env_name='InfoBlueYellow4b-v0' \
        --num_episodes=40 \
        --tie_break='next' \
        --tie_threshold=1e-5 \
        --beta=None \
        --seed_value=23 \
        --reward_mode=False \
        --log_dir=$(DATA_PATH)/exp328/run2/
	curiosity_bandit.py \
		--env_name='InfoBlueYellow4b-v0' \
        --num_episodes=40 \
        --tie_break='next' \
        --tie_threshold=1e-5 \
        --beta=None \
        --seed_value=802 \
        --reward_mode=False \
        --log_dir=$(DATA_PATH)/exp328/run3/
	curiosity_bandit.py \
		--env_name='InfoBlueYellow4b-v0' \
        --num_episodes=40 \
        --tie_break='next' \
        --tie_threshold=1e-5 \
        --beta=None \
        --seed_value=42 \
        --reward_mode=False \
        --log_dir=$(DATA_PATH)/exp328/run4/
	curiosity_bandit.py \
		--env_name='InfoBlueYellow4b-v0' \
        --num_episodes=40 \
        --tie_break='next' \
        --tie_threshold=1e-5 \
        --beta=None \
        --seed_value=673 \
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
		'curiosity_bandit.py --env_name='InfoBlueYellow4b-v0' --num_episodes=320 --lr_E=1 --tie_break='next' --tie_threshold=1e-4 --beta=None --seed_value={1} --reward_mode=False --log_dir=$(DATA_PATH)/exp329/run{1}' ::: {1..100}

# Softmx mode
exp330:
	parallel -j 39 \
			--joblog '$(DATA_PATH)/exp330.log' \
			--nice 19 --delay 0 \
		'curiosity_bandit.py --env_name='InfoBlueYellow4b-v0' --num_episodes=320 --lr_E=1 --tie_break='next' --tie_threshold=1e-4 --beta=1000 --seed_value={1} --reward_mode=False --log_dir=$(DATA_PATH)/exp330/run{1}' ::: {1..100}


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
		'curiosity_bandit.py --env_name='InfoBlueYellow4a-v0' --num_episodes=80 --lr_E=1 --tie_break='next' --tie_threshold=1e-5 --beta=None --seed_value={1} --reward_mode=False --log_dir=$(DATA_PATH)/exp331/run{1}' ::: {1..100}

# Softmx mode
exp332:
	parallel -j 39 \
			--joblog '$(DATA_PATH)/exp332.log' \
			--nice 19 --delay 0 \
		'curiosity_bandit.py --env_name='InfoBlueYellow4a-v0' --num_episodes=80 --lr_E=1 --tie_break='next' --tie_threshold=1e-5 --beta=1000 --seed_value={1} --reward_mode=False --log_dir=$(DATA_PATH)/exp332/run{1}' ::: {1..100}

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
		'curiosity_bandit.py --env_name='InfoBlueYellow4c-v0' --num_episodes=320 --lr_E=1 --tie_break='next' --tie_threshold=1e-5 --beta=None --seed_value={1} --reward_mode=False --log_dir=$(DATA_PATH)/exp333/run{1}' ::: {1..100}

# Softmx mode
exp334:
	parallel -j 39 \
			--joblog '$(DATA_PATH)/exp334.log' \
			--nice 19 --delay 0 \
		'curiosity_bandit.py --env_name='InfoBlueYellow4c-v0' --num_episodes=320 --lr_E=1 --tie_break='next' --tie_threshold=1e-5 --beta=1000 --seed_value={1} --reward_mode=False --log_dir=$(DATA_PATH)/exp334/run{1}' ::: {1..100}


# TODO: the important of boredom
# TODO: eta sensitivity for reward bandits. -> Supp.
# TODO: info/reward bandits. Dense, dense. Sparse/Dense S+R vectors.
# TODO: port all models to SummaryWriter


