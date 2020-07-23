from gym.envs.registration import register

from infomercial.local_gym.bandit import BanditEnv
from infomercial.local_gym.bandit import BanditOneHot2
from infomercial.local_gym.bandit import BanditOneHot10
from infomercial.local_gym.bandit import BanditOneHot121
from infomercial.local_gym.bandit import BanditOneHot1000
from infomercial.local_gym.bandit import BanditEvenOdds2
from infomercial.local_gym.bandit import BanditOneHigh2
from infomercial.local_gym.bandit import BanditOneHigh4
from infomercial.local_gym.bandit import BanditOneHigh10
from infomercial.local_gym.bandit import BanditOneHigh121
from infomercial.local_gym.bandit import BanditTwoHigh10
from infomercial.local_gym.bandit import BanditTwoHigh121
from infomercial.local_gym.bandit import BanditOneHigh1000
from infomercial.local_gym.bandit import BanditTwoHigh1000
from infomercial.local_gym.bandit import BanditHardAndSparse2
from infomercial.local_gym.bandit import BanditHardAndSparse10
from infomercial.local_gym.bandit import BanditHardAndSparse121
from infomercial.local_gym.bandit import BanditHardAndSparse1000
from infomercial.local_gym.bandit import BanditGaussian10
from infomercial.local_gym.bandit import BanditTwoExtreme1000
from infomercial.local_gym.bandit import BanditUniform10
from infomercial.local_gym.bandit import BanditUniform121

from infomercial.local_gym.bandit import DeceptiveBanditOneHigh10

from infomercial.local_gym.bandit import InfoBanditEnv
from infomercial.local_gym.bandit import InfoBlueYellow2a
from infomercial.local_gym.bandit import InfoBlueYellow2b
from infomercial.local_gym.bandit import InfoBlueYellow4a
from infomercial.local_gym.bandit import InfoBlueYellow4b
from infomercial.local_gym.bandit import InfoBlueYellow4c

# Gym is annoying these days...
import warnings
warnings.filterwarnings("ignore")

environments = [
    ['BanditOneHot2', 'v0', 1],
    ['BanditOneHot10', 'v0', 1],
    ['BanditOneHot121', 'v0', 1],
    ['BanditOneHot1000', 'v0', 1],
    ['BanditEvenOdds2', 'v0', 1],
    ['BanditOneHigh2', 'v0', 1],
    ['BanditOneHigh4', 'v0', 1],
    ['BanditOneHigh10', 'v0', 1],
    ['BanditOneHigh121', 'v0', 1],
    ['BanditTwoHigh121', 'v0', 1],
    ['BanditTwoHigh10', 'v0', 1],
    ['BanditOneHigh1000', 'v0', 1],
    ['BanditTwoHigh1000', 'v0', 1],
    ['BanditTwoExtreme1000', 'v0', 1],
    ['BanditHardAndSparse2', 'v0', 1],
    ['BanditHardAndSparse10', 'v0', 1],
    ['BanditHardAndSparse121', 'v0', 1],
    ['BanditHardAndSparse1000', 'v0', 1],
    ['BanditUniform10', 'v0', 1],
    ['BanditUniform121', 'v0', 1],
    ['BanditGaussian10', 'v0', 1],
    ['DeceptiveBanditOneHigh10', 'v0', 1],
    ['InfoBlueYellow2a', 'v0', 1],
    ['InfoBlueYellow2b', 'v0', 1],
    ['InfoBlueYellow4a', 'v0', 1],
    ['InfoBlueYellow4b', 'v0', 1],
    ['InfoBlueYellow4c', 'v0', 1],
]

for environment in environments:
    register(
        id='{}-{}'.format(environment[0], environment[1]),
        entry_point='infomercial.local_gym:{}'.format(environment[0]),
        # timestep_limit=environment[2],
        nondeterministic=True,
    )
