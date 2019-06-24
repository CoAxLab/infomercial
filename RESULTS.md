Summarize results:

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
- Task was not solved by any agent.

- meta: 

- beta:

- softbeta:

- epsilon:

# HardAndSparse10
- [DONE] meta: exp116 - learns a stable soln 
  + 'tie_threshold': 3.76-09, 'lr_R': 0.00021

- [DONE} beta: exp110 - Close to soln. Not stable. Narrow range?
  + 'beta': 2.83, 'lr_R': 0.053

- [DONE] softbeta: exp122 - learns the value but needs to high a temp to ever stabilize
  + 'beta': 0.38, 'lr_R': 0.00971, 'temp': 5.9

- [DONE] epsilon: exp121 - learns the value, but final performance limited by high epsilon
  + 'epsilon': 0.42, 'lr_R': 0.00043