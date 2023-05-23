import glob
import pathlib
import setuptools
from setuptools import find_namespace_packages


setuptools.setup(
    name='viper_rl',
    version='0.0.3',
    description='Video Prediction Models as Rewards for Reinforcement Learning',
    author='Alejandro Escontrela',
    url='http://github.com/alescontrela/viper_rl',
    long_description=pathlib.Path('README.md').read_text(),
    long_description_content_type='text/markdown',
    packages=find_namespace_packages(exclude=['runs', 'notebooks', 'scripts', 'viper_rl_data.datasets', 'viper_rl_data.datasets.*']),
    include_package_data=True,
    scripts=glob.glob('viper_rl_data/download/*.sh'),
    install_requires=pathlib.Path('requirements.txt').read_text().splitlines(),
    classifiers=[
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
    ],
)