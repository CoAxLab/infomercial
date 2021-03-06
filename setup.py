from setuptools import setup

setup(name='infomercial',
      version='0.0.1',
      description="Agents who seek information.",
      url='',
      author='Erik J. Peterson',
      author_email='erik.exists@gmail.com',
      license='MIT',
      packages=['infomercial'],
      scripts=[
          'infomercial/exp/meta_bandit.py',
          'infomercial/exp/softmeta_bandit.py',
          'infomercial/exp/epsilon_bandit.py',
          'infomercial/exp/beta_bandit.py',
          'infomercial/exp/softbeta_bandit.py',
          'infomercial/exp/random_bandit.py',
          'infomercial/exp/count_bandit.py',
          'infomercial/exp/entropy_bandit.py',
          'infomercial/exp/tune_bandit.py',
          'infomercial/exp/curiosity_bandit.py',
      ],
      zip_safe=False)
