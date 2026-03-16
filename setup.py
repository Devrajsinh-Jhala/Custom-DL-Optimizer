from setuptools import setup, find_packages

setup(
    name='custom-dl-optimizer', 
    version='1.0.1',
    description='A Deep Learning Auto-Optimizer using FX and Triton',
    packages=find_packages(),
    install_requires=['torch', 'triton'],
)