# Intro

Summarize result for 4 agents, over 4 prototypical bandit tasks.

`BanditOneHigh10` is a classic 10 armed bandit, with one clear winning arm (p_R = 0.8), all other arms are p_R = 0.2. `BanditTwoHigh10` is the same as `BanditOneHigh10` but there are two winning arms.

`BanditUniform121` is a high dimensional random bandit. All but 1 arms have a p_R draw uniformly from (0.2-0.6). One winner arm has a p_R = 0.8.

`HardAndSparse10` has a winner arm with p_R = 0.02. All other arms have a p_R = 0.01.


# Random search
## BanditOneHigh10
- [DONE] meta: exp96 - learns a stable soln 
  + 'tie_threshold': 0.0041, 'lr_R': 0.31

- [DONE] beta: exp98 - learns a stable soln 
  + 'beta': 0.37, 'lr_R': 0.0095

- [DONE] softbeta: exp112 - learns a stable soln 
  + 'beta': 0.045, 'lr_R': 0.12, 'temp': 0.10

- [DONE] epsilon: exp97 - learns a stable soln 
  + 'epsilon': 0.078, 'lr_R': 0.12

## BanditTwoHigh10
- [DONE] meta: exp100 - sees both, learns a stable soln
  + 'tie_threshold': 0.0058, 'lr_R': 0.14

- [DONE] beta: exp102 - learns only one arm. Never sees best arm 2
  + 'beta': 0.025, 'lr_R': 0.073

- [DONE] softbeta: exp113 - sees both (probably?), learns a stable soln
  + 'beta': 0.010, 'lr_R': 0.17, 'temp': 0.24

- [DONE] epsilon: exp101 - learns solns, flip flops between them
  + 'epsilon': 0.078, 'lr_R': 0.12

## BanditUniform121
- meta: exp124 - found stable soln
  + 'tie_threshold': 0.00031, 'lr_R': 0.14

- beta: exp126 - found stable soln (very eff.)
  + 'beta': 0.090, 'lr_R': 0.061

- softbeta: exp127 - no soln found. p_best low (temp too)
  + 'beta': 0.60, 'lr_R': 0.097, 'temp': 0.13

- epsilon - exp125: found stable soln (low ep)
  + 'epsilon': 0.012, 'lr_R': 0.11

## HardAndSparse10
- [DONE] meta: exp116 - learns a stable soln 
  + 'tie_threshold': 3.76-09, 'lr_R': 0.00021

- [DONE} beta: exp110 - Close to soln. Not stable. Narrow range?
  + 'beta': 2.83, 'lr_R': 0.053

- [DONE] softbeta: exp122 - learns the value but needs to high a temp to ever stabilize
  + 'beta': 0.38, 'lr_R': 0.00971, 'temp': 5.9

- [DONE] epsilon: exp121 - learns the value, but final performance limited by high epsilon
  + 'epsilon': 0.42, 'lr_R': 0.00043

# Replicator
## BanditOneHigh10
- meta: {'tie_threshold': 0.053, 'lr_R': 0.34, 'total_R': 80.0}

- beta: {'beta': 0.22, 'lr_R': 0.18, 'total_R': 83.0}

- softbeta: {'beta': 0.066, 'lr_R': 0.13, 'temp': 0.13, 'total_R': 80.0}

- epsilon: {'epsilon': 0.14, 'lr_R': 0.087, 'total_R': 81.0}

- anneal-epsilon: {'epsilon': 0.45, 'epsilon_decay_tau': 0.061, 'lr_R': 0.14, 'total_R': 83.0}

## BanditTwoHigh10
- meta: {'tie_threshold': 0.0169, 'lr_R': 0.161, 'total_R': 169.0}

- beta: {'beta': 0.188, 'lr_R': 0.129, 'total_R': 169.0}

- softbeta: {'beta': 0.133, 'lr_R': 0.030, 'temp': 0.098, 'total_R': 148.0}

- epsilon: {'epsilon': 0.0393, 'lr_R': 0.08583, 'total_R': 169.0}

- anneal-epsilon: {'epsilon': 0.980, 'epsilon_decay_tau': 0.084, 'lr_R': 0.194, 'total_R': 165.0}

## BanditUniform121
- meta: {'tie_threshold': 0.00355, 'lr_R': 0.147, 'total_R': 48358.0}

- beta: {'beta': 0.05683582352190725, 'lr_R': 0.1411684862500652, 'total_R': 48381.0}

- softbeta: {'beta': 0.125, 'lr_R': 0.174, 'temp': 0.0811, 'total_R': 37218.0}

- epsilon: {'epsilon': 0.0117, 'lr_R': 0.137, 'total_R': 47899.0}

- anneal-epsilon: {'epsilon': 0.850, 'epsilon_decay_tau': 0.00777, 'lr_R': 0.173, 'total_R': 48496.0}

## HardAndSparse10
- meta: {'tie_threshold': 5.782e-09, 'lr_R': 0.00112, 'total_R': 1049.0}

- beta: {'beta': 0.217, 'lr_R': 0.0508, 'total_R': 945.0}

- softbeta: {'beta': 2.140, 'lr_R': 0.128, 'temp': 5.045, 'total_R': 613.0}

- epsilon: {'epsilon': 0.4057, 'lr_R': 0.000484, 'total_R': 878.0}

- anneal-epsilon: {'epsilon': 0.5148, 'epsilon_decay_tau': 0.0723, 'lr_R': 0.000271, 'total_R': 1084.0}