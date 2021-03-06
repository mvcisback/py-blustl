from setuptools import setup, find_packages

setup(
    name='magnumSTL',
    version='0.1',
    description='TODO',
    url='http://github.com/mvcisback/magnumSTL',
    author='Marcell Vazquez-Chanlatte',
    author_email='marcell.vc@eecs.berkeley.edu',
    license='MIT',
    install_requires=[
        'bidict',
        'funcy',
        'lenses',
        'py-stl',
        'optlang',
        'pysmt',
        'traces'
    ],
    packages=find_packages(),
)
