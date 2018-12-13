from gym.envs.registration import register

from infomercial.local_gym.bandit import BanditTenArmedRandomFixed
from infomercial.local_gym.bandit import BanditTenArmedRandomRandom
from infomercial.local_gym.bandit import BanditTenArmedGaussian
from infomercial.local_gym.bandit import BanditTenArmedUniformDistributedReward
from infomercial.local_gym.bandit import BanditTwoArmedDeterministicFixed
from infomercial.local_gym.bandit import BanditTwoArmedHighHighFixed
from infomercial.local_gym.bandit import BanditTwoArmedHighLowFixed
from infomercial.local_gym.bandit import BanditTwoArmedLowLowFixed
from infomercial.local_gym.bandit import BanditTwoArmedEvenFixed

environments = [
    ['BanditTenArmedRandomFixed', 'v0', 1],
    ['BanditTenArmedRandomRandom', 'v0', 1],
    ['BanditTenArmedGaussian', 'v0', 1],
    ['BanditTenArmedUniformDistributedReward', 'v0', 1],
    ['BanditTwoArmedDeterministicFixed', 'v0', 1],
    ['BanditTwoArmedHighHighFixed', 'v0', 1],
    ['BanditTwoArmedHighLowFixed', 'v0', 1],
    ['BanditTwoArmedLowLowFixed', 'v0', 1],
    ['BanditTwoArmedEvenFixed', 'v0', 1],
]

for environment in environments:
    register(
        id='{}-{}'.format(environment[0], environment[1]),
        entry_point='infomercial.local_gym:{}'.format(environment[0]),
        timestep_limit=environment[2],
        nondeterministic=True,
    )
