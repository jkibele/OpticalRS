# Installation

OpticalRS was developed under the Ubuntu operating system but should run on Windows and Mac as well. Methods for installing dependencies will vary by operating system but, once the dependencies are installed, installation of OpticalRS can be accomplished with the following command on any OS:

    $ pip install OpticalRS

On systems where the user has insufficient permissions, you may need to run the installation as root by using:

    $ sudo pip install OpticalRS

## Dependencies

Many of the dependencies (numpy, scipy, matplotlib, and pandas) are taken care of if you install the [scipy] stack (instructions [here](http://www.scipy.org/install.html) so it makes sense to do scipy first. The installation methods will vary depending on the operating system. Installation instructions can be found on the links below. The following are required for OpticalRS to install and run correctly:

* [numpy]
* [matplotlib]
* [scipy]
* [pandas]
* [statsmodels]
* [scikit-image]
* [scikit-learn]
* [GDAL]
* [geopandas]
* [rasterstats]

## Intallation on Ubuntu

The following shell script can be used to install OpticalRS and all of its dependencies on Ubuntu. This script is available from the [OpticalRS Repository](https://github.com/jkibele/OpticalRS) as `install_with_dependencies.sh`. You can either run the script or just copy and paste the commands to the command line one at a time (in the correct order). The result will be the same either way.

    #!/bin/sh
    # This script has been tested (a bit) on Ubuntu 14.04

    # This adds the repository for the gdal stuff (the last apt-get line)
    sudo add-apt-repository ppa:ubuntugis/ppa -y
    sudo apt-get update

    # Install Scipy stack. Instructions here: http://www.scipy.org/install.html
    sudo apt-get install -y python-numpy python-scipy python-matplotlib ipython ipython-notebook python-pandas python-sympy python-nose

    # This is needed for the last line that actually installs OpticalRS (and a few dependencies handled by pip)
    sudo apt-get install -y python-pip

    # gdal libraries are required by some of the python requirements installed by pip
    sudo apt-get install -y python-gdal libgdal1h gdal-bin libgdal-dev

    # This finishes up the dependencies and finally the actual OpticalRS code
    sudo pip install OpticalRS

[numpy]: http://www.numpy.org/
[matplotlib]: http://matplotlib.org/
[scipy]: http://scipy.org/
[pandas]: http://pandas.pydata.org/
[statsmodels]: http://statsmodels.sourceforge.net/
[scikit-image]: http://scikit-image.org/
[scikit-learn]: http://scikit-learn.org/
[GDAL]: https://pypi.python.org/pypi/GDAL/
[geopandas]: http://geopandas.org/
[rasterstats]: https://github.com/perrygeo/python-rasterstats
