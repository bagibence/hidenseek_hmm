from setuptools import find_packages, setup
import os

setup(
    name='hidenseek',
    packages=find_packages(),
    version='0.1.0',
    description='Analysis of multiunit recordings from rat prefrontal cortex during playing hide and seek',
    author='Bence Bagi',
    license='BSD-3',
    install_requires = [
        'numpy',
        'scipy',
        'pandas',
        'xarray==0.15.1', 
        'matplotlib',
        'seaborn',
        'scikit-learn',
        'python-dotenv',
        'pony',
        #'ssm @ git+https://git@github.com/slinderman/ssm@master#egg=ssm',
    ]
)


# create config.dotenv file
base_dir = os.path.dirname(os.path.abspath(__file__))

config_str = """
ROOT_DIR={base_dir}

DATA_DIR=${{ROOT_DIR}}/data

RAW_DATA_DIR=${{DATA_DIR}}/raw
INTERIM_DATA_DIR=${{DATA_DIR}}/interim
PROCESSED_DATA_DIR=${{DATA_DIR}}/processed

DB_PATH=${{INTERIM_DATA_DIR}}/database.db
OBSERVING_DB_PATH=${{INTERIM_DATA_DIR}}/observing.db
""".format(base_dir = base_dir)

with open(os.path.join(base_dir, 'config.dotenv'), 'w') as dotenv_file:
    dotenv_file.write(config_str)

with open(os.path.join(base_dir, 'matlab', 'data_path.m'), 'w') as f:
    f.write(f"data_dir = '{os.path.join(base_dir, 'data')}'")
