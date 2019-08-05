import os

from setuptools import setup, find_packages


def readme():
    """
    Utility function to read the README file.
    Used for the long_description.  It's nice, because now 1) we have a top level
    README file and 2) it's easier to type in the README file than to put a raw
    string in below ...
    :return: String
    """
    return open(os.path.join(os.path.dirname(__file__), 'README.md')).read()


def requirements():
    """
    Parse requirements.txt to array of packages to have installed, so we
    maintain dependencies in requirements.txt and make setup.py use it
    :return: list of requirements
    """
    with open('requirements.txt') as f:
        return f.read().splitlines()


setup(
    name='ieee-fraud-detection',
    version='0.1.0',
    packages=find_packages(exclude=['contrib', 'docs', 'tests*']),
    url='https://github.com/andfanilo/ieee-fraud-detection',
    license='',
    author='Fanilo ANDRIANASOLO',
    author_email='andfanilo@gmail.com',
    description='IEEE Fraud Detection Kaggle challenge',
    python_requires='>=3.6',
    long_description=readme(),
    install_requires=requirements(),
    entry_points='''
    [console_scripts]
    run_experiment=src.app:run_experiment
    '''
)