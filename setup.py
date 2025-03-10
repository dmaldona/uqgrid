# setup.py

from setuptools import setup, find_packages

setup(
    name='UQGrid',
    version='0.1.0',
    description='Uncertainty quantification for the electrical grid.',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    author='D. Adrian Maldonado',
    author_email='maldonadod@anl.gov',
    url='https://github.com/dmaldona/uqgrid',  # Replace with your repository URL
    packages=find_packages(),
    include_package_data=True,
    install_requires=[
        'numpy',
        'scipy',
        'pytest',
        'numba',
        'petsc4py',
        'matplotlib',
        'networkx',
        'pydantic',
    ],
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.6',
    tests_require=[
        'pytest',
        'nose',
    ],
)