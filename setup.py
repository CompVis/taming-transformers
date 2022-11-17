from setuptools import setup, find_namespace_packages

setup(
    name='taming-transformers',
    version='0.0.1',
    description='Taming Transformers for High-Resolution Image Synthesis',
    packages=find_namespace_packages(include=["taming", "taming.*"]),
    install_requires=[
        'torch',
        'numpy',
        'tqdm',
        'pytorch-lightning',
        'einops'
    ],
)
