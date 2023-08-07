from setuptools import setup, find_packages

with open('requirements.txt') as f:
    required = f.read().splitlines()

setup(
    name='opticalrs',
    version='2.0.1',
    url='https://github.com/jkibele/OpticalRS',
    author='Jared Kibele',
    author_email='jkibele@gmail.com',
    description=('OpticalRS is a free and open source Python implementation of passive optical remote sensing methods '
                 'for the derivation of bathymetric maps and maps of submerged habitats.'),
    packages=['opticalrs'],
    package_dir={'opticalrs': 'opticalrs'},
    include_package_data=True,
    install_requires=required,
)
