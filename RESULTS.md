Summarize result for 4 agents, over 4 prototypical bandit tasks.

`BanditOneHigh10` is a classic 10 armed bandit, with one clear winning arm (p_R = 0.8), all other arms are p_R = 0.2. `BanditTwoHigh10` is the same as `BanditOneHigh10` but there are two winning arms.

`BanditUniform121` is a high dimensional random bandit. All but 1 arms have a p_R draw uniformly from (0.2-0.6). One winner arm has a p_R = 0.8.

`HardAndSparse10` has a winner arm with p_R = 0.02. All other arms have a p_R = 0.01.



# BanditOneHigh10
- [DONE] meta: exp96 - learns a stable soln 
  + 'tie_threshold': 0.0041, 'lr_R': 0.31

- [DONE] beta: exp98 - learns a stable soln 
  + 'beta': 0.37, 'lr_R': 0.0095

- [DONE] softbeta: exp112 - learns a stable soln 
  + 'beta': 0.045, 'lr_R': 0.12, 'temp': 0.10

- [DONE] epsilon: exp97 - learns a stable soln 
  + 'epsilon': 0.078, 'lr_R': 0.12

# BanditTwoHigh10
- [DONE] meta: exp100 - sees both, learns a stable soln
  + 'tie_threshold': 0.0058, 'lr_R': 0.14

- [DONE] beta: exp102 - learns only one arm. Never sees best arm 2
  + 'beta': 0.025, 'lr_R': 0.073

- [DONE] softbeta: exp113 - sees both (probably?), learns a stable soln
  + 'beta': 0.010, 'lr_R': 0.17, 'temp': 0.24

- [DONE] epsilon: exp101 - learns solns, flip flops between them
  + 'epsilon': 0.078, 'lr_R': 0.12

# BanditUniform121
- meta: exp124 - found stable soln
  + 'tie_threshold': 0.00031, 'lr_R': 0.14

- beta: exp126 - found stable soln (very eff.)
  + 'beta': 0.090, 'lr_R': 0.061

- softbeta: exp127 - no soln found. p_best low (temp too)
  + 'beta': 0.60, 'lr_R': 0.097, 'temp': 0.13

- epsilon - exp125: found stable soln (low ep)
  + 'epsilon': 0.012, 'lr_R': 0.11

# HardAndSparse10
- [DONE] meta: exp116 - learns a stable soln 
  + 'tie_threshold': 3.76-09, 'lr_R': 0.00021

- [DONE} beta: exp110 - Close to soln. Not stable. Narrow range?
  + 'beta': 2.83, 'lr_R': 0.053

- [DONE] softbeta: exp122 - learns the value but needs to high a temp to ever stabilize
  + 'beta': 0.38, 'lr_R': 0.00971, 'temp': 5.9

- [DONE] epsilon: exp121 - learns the value, but final performance limited by high epsilon
  + 'epsilon': 0.42, 'lr_R': 0.00043