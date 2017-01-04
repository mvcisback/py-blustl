from setuptools import setup, find_packages

setup(name='blustl',
      version='0.1',
      description='TODO',
      url='http://github.com/mvcisback/py-blustl',
      author='Marcell Vazquez-Chanlatte',
      author_email='marcell.vc@eecs.berkeley.edu',
      license='MIT',
      install_requires=[
          'funcy',
          'PyYAML',
          'lenses',
          'py-stl',
          'pulp',
          'pyeda',
      ],
      packages=find_packages(),
)
