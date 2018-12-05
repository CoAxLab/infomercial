from gym.envs.registration import register

from .bandit import BanditTenArmedRandomFixed
from .bandit import BanditTenArmedRandomRandom
from .bandit import BanditTenArmedGaussian
from .bandit import BanditTenArmedUniformDistributedReward
from .bandit import BanditTwoArmedDeterministicFixed
from .bandit import BanditTwoArmedHighHighFixed
from .bandit import BanditTwoArmedHighLowFixed
from .bandit import BanditTwoArmedLowLowFixed

environments = [
    ['BanditTenArmedRandomFixed', 'v0', 1],
    ['BanditTenArmedRandomRandom', 'v0', 1],
    ['BanditTenArmedGaussian', 'v0', 1],
    ['BanditTenArmedUniformDistributedReward', 'v0', 1],
    ['BanditTwoArmedDeterministicFixed', 'v0', 1],
    ['BanditTwoArmedHighHighFixed', 'v0', 1],
    ['BanditTwoArmedHighLowFixed', 'v0', 1],
    ['BanditTwoArmedLowLowFixed', 'v0', 1],
]

for environment in environments:
    register(
        id='{}-{}'.format(environment[0], environment[1]),
        entry_point='azad.local_gym:{}'.format(environment[0]),
        timestep_limit=environment[2],
        nondeterministic=True,
    )
