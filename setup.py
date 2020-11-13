"""Author: Brandon Trabucco, Copyright 2019, MIT License"""


from setuptools import find_packages
from setuptools import setup


REQUIRED_PACKAGES = [
    'tensorflow-gpu==2.3.1',
    'numpy',
    'matplotlib']


setup(
    name='vq_vae_2',
    version='0.1',
    install_requires=REQUIRED_PACKAGES,
    include_package_data=True,
    packages=[p for p in find_packages() if p.startswith('vq_vae_2')],
    description='Vector Quantized Latent Variable Models In TF 2.0')
