from setuptools import find_packages, setup


setup(
    name='FreSH',
    version='0.1.0',
    description=(
        'Frequency-Segmented Hierarchical Multi-Expert Framework '
        'for Multivariate Time Series Classification'
    ),
    long_description=(
        'Official implementation of the FreSH model. FreSH transforms '
        'multivariate time series into the frequency domain and applies a '
        'hierarchical mixture-of-experts architecture with adaptive gating.'
    ),
    packages=find_packages(exclude=('test', 'tests', 'scripts')),
    python_requires='>=3.8',
    install_requires=[
        'torch>=2.0',
        'numpy>=1.23,<2',
        'pandas>=1.5,<3',
        'scikit-learn>=1.2',
        'scipy>=1.10',
        'sktime>=0.16',
        'tqdm>=4.64',
        'matplotlib>=3.7',
        'patool>=1.12',
        'PyWavelets',
    ],
)
