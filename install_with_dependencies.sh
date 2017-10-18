#!/bin/sh
# This script has been tested (a bit) on Ubuntu 14.04

# This adds the repository for the gdal stuff (the last apt-get line)
sudo add-apt-repository ppa:ubuntugis/ppa -y
sudo apt-get update

# Install Scipy stack. Instructions here: http://www.scipy.org/install.html
sudo apt-get install -y python-numpy python-scipy python-matplotlib ipython ipython-notebook python-pandas python-sympy python-nose

# This is needed for the last line that actually installs OpticalRS (and a few dependencies handled by pip)
sudo apt-get install -y python-pip

# GDAL libraries are required by some of the python requirements installed by pip
sudo apt-get install -y python-gdal libgdal1h gdal-bin libgdal-dev

# Export GDAL header location
export CPLUS_INCLUDE_PATH=/usr/include/gdal
export C_INCLUDE_PATH=/usr/include/gdal

# This finishes up the dependencies and finally the actual OpticalRS code 
sudo pip install OpticalRS
