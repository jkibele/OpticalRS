============
Installation
============

OpticalRS was developed under the Ubuntu operating system but should run on Windows and Mac as well. Methods for installing dependencies will vary by operating system but, once the dependencies are installed, installation of OpticalRS can be accomplished by with the following command on any OS::

    $ pip install OpticalRS

On systems where the user has insufficient permissions, you may need to run the installation as root by using::

    $ sudo pip install OpticalRS

Dependencies
------------

Many of the dependencies (numpy, scipy, matplotlib, and pandas) are taken care of if you install the scipy_ stack (instructions `here <http://www.scipy.org/install.html>`_) so it makes sense to do scipy first. The installation methods will vary depending on the operating system. Installation instructions can be found on the links below. The following are required for OpticalRS to install and run correctly:

* numpy_
* matplotlib_
* scipy_
* pandas_
* statsmodels_
* scikit-image_
* GDAL_
* geopandas_
* rasterstats_

Intallation on Ubuntu
---------------------

The following shell script can be used to install OpticalRS and all of its dependencies on Ubuntu. This script is available from the `OpticalRS Repository`__ as `install_with_dependencies.sh`. You can either run the script or just copy and paste the commands to the command line one at a time (in the correct order). The result will be the same either way.

__ https://github.com/jkibele/OpticalRS

.. code-block:: shell

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

    # Export GDAL header location
    export CPLUS_INCLUDE_PATH=/usr/include/gdal
    export C_INCLUDE_PATH=/usr/include/gdal
    
    # This finishes up the dependencies and finally the actual OpticalRS code 
    sudo pip install OpticalRS

.. _numpy: http://www.numpy.org/
.. _matplotlib: http://matplotlib.org/
.. _scipy: http://scipy.org/
.. _pandas: http://pandas.pydata.org/
.. _statsmodels: http://statsmodels.sourceforge.net/
.. _scikit-image: http://scikit-image.org/
.. _GDAL: https://pypi.python.org/pypi/GDAL/
.. _geopandas: http://geopandas.org/
.. _rasterstats: https://github.com/perrygeo/python-rasterstats
