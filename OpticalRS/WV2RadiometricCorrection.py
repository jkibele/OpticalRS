# -*- coding: utf-8 -*-
"""
WV2RadiometricCorrection
========================

This module reads parameters from a DigitalGlobe supplied xml file and applies
radiometric correction to WorldView-2 imagery according to the instructions
found here::

DigitalGlobe, 2010. Radiometric Use of WorldView-2 Imagery.
http://www.digitalglobe.com/downloads/Radiometric_Use_of_WorldView-2_Imagery.pdf

For usage instructions, execute::

    $ python WV2RadiometricCorrection.py -h

Notes
=====

This module gives results that are very close to, but not the same as, the Orfeo
Toolbox [Optical Calibration
module](https://www.orfeo-toolbox.org/CookBook/CookBooksu35.html). I think this
may be because the OTB module is using general parameters for WV-2 imagery while
this module is using specific parameters from the image's xml file. So, I am
reasonably certain this module is working as intended. On the other hand, I
really haven't investigated the issue. If you look into it, please let me know
what you find out. At some point I need to do some more testing and add in much
more thorough automated tests.
"""

from xml.etree.ElementTree import ElementTree as ET
from collections import OrderedDict
from datetime import datetime
from osgeo import gdal
from osgeo.gdalconst import *
import argparse, os, sys
import numpy as np

## Raster handling #########################################
# I'll want to integrate this module with the rest of OpticalRS by using the
# RasterDS module to do this stuff. For now, I'm doing it this way. ...this does
# have the benefit of making this a stand alone module relying only on the gdal
# import.

def open_raster(filename):
    """Take a file path as a string and return a gdal datasource object"""
    # register all of the GDAL drivers
    gdal.AllRegister()

    # open the image
    img = gdal.Open(filename, GA_ReadOnly)
    if img is None:
        print 'Could not open %s' % filename
        sys.exit(1)
    else:
        return img

def bandarr_from_ds(img):
    """Take a raster datasource and return a band array. Each band is read
    as an array and becomes one element of the band array."""
    for band in range(1,img.RasterCount + 1):
        barr = img.GetRasterBand(band).ReadAsArray()
        if band==1:
            bandarr = np.array([barr])
        else:
            bandarr = np.append(bandarr,[barr],axis=0)
    return bandarr

def output_gtif(bandarr, cols, rows, outfilename, geotransform, projection, no_data_value=-99, driver_name='GTiff', dtype=GDT_Float32):
    """Create a geotiff with gdal that will contain all the bands represented
    by arrays within bandarr which is itself array of arrays."""
    # make sure bandarr is a proper band array
    if len( bandarr.shape )==2:
        bandarr = np.array([ bandarr ])
    driver = gdal.GetDriverByName(driver_name)
    outDs = driver.Create(outfilename, cols, rows, len(bandarr), dtype)
    if outDs is None:
        print "Could not create %s" % outfilename
        sys.exit(1)
    for bandnum in range(1,len(bandarr) + 1):  # bandarr is zero based index while GetRasterBand is 1 based index
        outBand = outDs.GetRasterBand(bandnum)
        outBand.WriteArray(bandarr[bandnum - 1])
        outBand.FlushCache()
        outBand.SetNoDataValue(no_data_value)

    # georeference the image and set the projection
    outDs.SetGeoTransform(geotransform)
    outDs.SetProjection(projection)

    # build pyramids
    gdal.SetConfigOption('HFA_USE_RRD', 'YES')
    outDs.BuildOverviews(overviewlist=[2,4,8,16,32,64,128])

def output_gtif_like_img(img, bandarr, outfilename, no_data_value=-99, dtype=GDT_Float32):
    """Create a geotiff with attributes like the one passed in but make the
    values and number of bands as in bandarr."""
    cols = img.RasterXSize
    rows = img.RasterYSize
    geotransform = img.GetGeoTransform()
    projection = img.GetProjection()
    output_gtif(bandarr, cols, rows, outfilename, geotransform, projection, no_data_value, driver_name='GTiff', dtype=dtype)

## End of Raster Handling stuff ################################################

"""These values are taken from Table 4 in DigitalGlobe's Radiometric Use of WorldView-2
Imagery technical note. See references.txt for more info."""
Esun_od = OrderedDict((
        ('BAND_C', 1758.2229),
        ('BAND_B', 1974.2416),
        ('BAND_G', 1856.4104),
        ('BAND_Y', 1738.4791),
        ('BAND_R', 1559.4555),
        ('BAND_RE', 1342.0695),
        ('BAND_N', 1069.7302),
        ('BAND_N2', 861.2866),
    ))

def julian_date2(date):
    """Returns the Julian day number of a date. I got the code for this method
    from a post on the internet. I've included it just to check my results."""
    a = (14 - date.month)//12
    y = date.year + 4800 - a
    m = date.month + 12*a - 3
    return date.day + ((153*m + 2)//5) + 365*y + y//4 - y//100 + y//400 - 32045

def julian_date(date):
    """Returns the julian day of a date and includes the decimal portion. These
    calculations were taken from section 6.1.1 of Radiometric Use of WorldView-2
    Imagery. See references.txt for more information."""
    year = date.year
    month = date.month
    day = date.day
    UT = date.hour + date.minute/60.0 + date.second/3600.0
    if year <= 2:
        year = year - 1
        month += 12
    A = year // 100
    B = 2 - A + A // 4
    JD = int( 365.25 * (year + 4716) ) + int( 30.6001 * (month + 1) ) + day + UT / 24.0 + B - 1524.5
    return JD

## Methods for dealing with WV2 xml files #######################
def get_xml_filename(fname):
    """If filename has xml extension, just pass it back. If not, see
    if there's a file in the same directory with an xml file extension
    and pass that back. Deal with different cases.

    >>> get_xml_filename('porkchop.XML')
    'porkchop.XML'

    >>> get_xml_filename('porkchop.xml')
    'porkchop.xml'

    >>> get_xml_filename('porkchop.xML')
    'porkchop.xML'

    >>> get_xml_filename('porkchop.TIF')
    'porkchop.XML'

    >>> get_xml_filename('porkchop.tif')
    'porkchop.xml'

    >>> get_xml_filename('porkchop.Tif')
    'porkchop.xml'
    """
    # get file extension of input file
    fext = fname.split(os.path.extsep)[-1]
    if fext.lower() != 'xml': # Did we get passed something other than xml?
        # assume that extension case is consistent with what we've been passed
        if fext.isupper():
            fname = fname.replace(os.path.extsep+fext,os.path.extsep+'XML')
        elif fext.islower():
            fname = fname.replace(os.path.extsep+fext,os.path.extsep+'xml')
        else: # this would mean the extension is mixed case. Not likely
            fname = fname.replace(os.path.extsep+fext,os.path.extsep+'xml')
    return fname

def get_xmlroot(filepath):
    """Return ElementTree root of xml file. You can pass in the path to the actual
    xml file or a path to a tif with the same name as the xml file."""
    filepath = get_xml_filename(filepath)
    tree = ET()
    tree.parse(filepath)
    return tree

def absCalFactor_dict(xmlroot):
    """Get an ordered dictionary with the xml tags for the image bands as keys
    and the absCalFactors as values. We will assume that the order of the bands
    in the xml file is the same as the order of the bands in the tif image. This
    seems to hold true in my experience."""
    imd = xmlroot.find('IMD')
    bandtags = [ x.tag for x in imd.getchildren() if x.tag.startswith('BAND_') ]
    od = OrderedDict()
    for bt in bandtags:
        val = float( imd.find(bt + '/ABSCALFACTOR').text )
        od.update({bt:val})
    return od

def effectiveBandwidth_dict(xmlroot):
    """Get an ordered dictionary with the xml tags for the image bands as keys
    and the effectiveBandwidth as values. We will assume that the order of the bands
    in the xml file is the same as the order of the bands in the tif image. This
    seems to hold true in my experience."""
    imd = xmlroot.find('IMD')
    bandtags = [ x.tag for x in imd.getchildren() if x.tag.startswith('BAND_') ]
    od = OrderedDict()
    for bt in bandtags:
        val = float( imd.find(bt + '/EFFECTIVEBANDWIDTH').text )
        od.update({bt:val})
    return od

def firstLineTime(xmlroot):
    """Get firstLineTime from the xml. This assumes that we are dealing with a standard
    WV-2 product. Apparently the xml structure is different for Basic products. Then again,
    you would probably have to follow a different procedure for those products so it is a
    moot point."""
    date_string = xmlroot.find('IMD/MAP_PROJECTED_PRODUCT/EARLIESTACQTIME').text
    date_string = date_string.rsplit('.')[0] # ditch the decimal seconds because I can't figure out the next line with them and I don't think the distance from the earth to the sun changes much in less than a second
    return datetime.strptime(date_string,'%Y-%m-%dT%H:%M:%S')

def firstLineJD(xmlroot):
    """Get the firstLineTime with the previous method and return the result as a Julian Date."""
    return julian_date(firstLineTime(xmlroot))

def meanSunEl(xmlroot):
    """Get the mean sun elevation angle from the xml."""
    return float(xmlroot.find('IMD/IMAGE/MEANSUNEL').text)

def meanOffNadirViewAngle(xmlroot):
    """Get the mean off nadir view angle from the xml."""
    return float(xmlroot.find('IMD/IMAGE/MEANOFFNADIRVIEWANGLE').text)

def solarZenithAngle(xmlroot):
    """Calculate the solar zenith angle from the mean sun elevation."""
    return 90.0 - meanSunEl(xmlroot)

def earthSunDistance(xmlroot):
    """Calculate the earth to sun distance based on the Julian Date.
    The formula is from Radiometric Use of WorldView-2 Imagery (see references.txt)"""
    JD = firstLineJD(xmlroot)
    D = JD - 2451545.0
    g = 357.529 + 0.98560028 * D
    return 1.00014 - 0.01671 * np.cos( np.radians(g) ) - 0.00014 * np.cos( np.radians( 2 * g ) )

### End of Methods for dealing with WV2 xml ###############################

def dark_pixel_finder(bandarr,prcnt=0.001):
    """
    Note: I'm messing up the terminology here. I need to straighten this out.
    You probably shouldn't use this unless you're sure you know what you're
    doing.

    Find the darkest pixels in bandarr. Pixel darkness will be determined by
    ranking pixels according to the mean of the pixel values across all bands
    plus 2*std deviation. An array of the row,col coordinates of the darkest
    pixels will be returned.
    """
    # build an array of darkness values that will be of the same dimensions
    # as an individual band of the image
    darkness = bandarr.mean(axis=0) + 2 * bandarr.std(axis=0)
    # build an array of non-zero darkness values in the first row, row coordinates
    # in the second row, and column coordinates in the third row
    d = np.array([darkness[darkness.nonzero()],darkness.nonzero()[0],darkness.nonzero()[1]])
    # sort d by darkness values
    dsort = d[:,d[0,:].argsort()]
    # figure out the number of darkest pixels to get using prcnt of nonzero darkness values
    npix = int( len(dsort[0]) * prcnt )
    # slice the sorted array and transpose it to get an array of [row,col] pairs
    coords = dsort[1:,:npix].T
    return coords.astype(np.int16)

def dark_pixel_subtraction(bandarr,prcnt=0.001,verbose=False):
    """
    Note: I'm messing up the terminology here. I need to straighten this out.
    You probably shouldn't use this unless you're sure you know what you're
    doing.

    This method uses the dark_pixel_finder method to get pixel coordinates for
    the darkest pixels in the images and uses those pixels to calculate the value
    to subtract from each band in the image. If verbose is set to true, the values
    calculated for each band will be printed.

    The methods were taken from Equation 7.1 of Lesson 7: Compensating
    for variable water depth from UNESCO's BILKO documentation. That in turn
    was taken from Lyzenga, 1978 Passive remote sensing techniques for mapping
    water depth and bottom features. That was actually taken from Polcyn 1970, I
    think. The Bilko lesson uses the subtraction of 2 standard deviations as does
    Deidda and Sanna 2012 but I'm not seeing that subtraction in Polcyn 1970 and
    Lyzenga 1978.
    """
    coords = dark_pixel_finder(bandarr,prcnt)
    for bnum in range(len(bandarr)):
        barr = bandarr[bnum]
        dark = barr[coords[:,0],coords[:,1]]
        subvalue = mean_minus_2_stdev(dark)
        barr = barr - subvalue
        barr[np.where(barr<=0.0)] = 0.0
        bandarr[bnum] = barr
        if verbose:
            print "%.2f subtracted from band %i" % (subvalue,bnum+1)
    return bandarr

def dark_pixel_proof_output(img,prcnt=0.001,outfilename=None):
    """Produce a geotiff that shows where the dark pixels came from. This
    is pretty much just a testing method and should not see much use."""
    coords = dark_pixel_finder(img,prcnt)
    band = img.GetRasterBand(1)
    bandarr = np.zeros((band.YSize,band.XSize),int)
    bandarr[coords[:,0],coords[:,1]] = 1
    bandarr = np.array([bandarr])
    if outfilename == None:
        outfilename = img.GetDescription().replace('.tif','_dp_proof.tif')
    output_gtif_like_img(img,bandarr,outfilename)

def toa_radiance(bandarr,absCal,effBand):
    """Take an array radiometrically corrected image pixels (the standard WV2 product),
    the absolute radiometric calibration factor for the band (absCal), and the effective
    bandwidth (effBand) and return an array of top of atmosphere spectral radiance pixels.
    The formula used is toa_rad(pixel) = ( absCal * bandarr(pixel) ) / effBand
    The formula is from Radiometric Use of WorldView-2 Imagery (see references.txt)

    >>> toa_radiance(np.array([[1,2,3,4],[5,6,7,8]]),4,2)
    array([[ 2,  4,  6,  8],
           [10, 12, 14, 16]])
    """
    outarr = (absCal * bandarr) / effBand
    return outarr

def toa_radiance_multiband(bandarr,xmlorimagepath):
    """Take a band array (bandarr) representing a raster with multiple bands and
    use the toa_radiance method to modify the multi band array and then pass it back.

    xmlorimagepath can be the actual path to the xml file or it can be the path to a tif
    with the same file name. See get_xml_filename and get_xmlroot for details."""
    # get necessary values from the xml
    xmlroot = get_xmlroot(xmlorimagepath)
    abfd = absCalFactor_dict(xmlroot)
    ebd = effectiveBandwidth_dict(xmlroot)

    for band in range(len(bandarr)):
        toa_arr = toa_radiance(bandarr[band],abfd.values()[band],ebd.values()[band])
        if band==0:
            toabandarr = np.array([toa_arr])
        else:
            toabandarr = np.append(toabandarr,[toa_arr],axis=0)
    return toabandarr

def toa_reflectance(bandarr,absCal,effBand,eSun,distES,solarZenith):
    """Calculate top of atmosphere reflectance. The formula is from section 7 of Radiometric
    Use of WorldView-2 Imagery (see references.txt) bandarr here is an array for a single band."""
    toa_rad_arr = toa_radiance(bandarr,absCal,effBand)
    outarr = ( toa_rad_arr * (distES**2) * np.pi ) / ( eSun * np.cos( np.radians(solarZenith) ) )
    return np.clip(outarr, 0.0, 1.0)

def toa_reflectance_multiband(bandarr,xmlorimagepath):
    """Take a band array (bandarr) representing a raster with multiple bands and
    use the toa_reflectance method to modify the multi band array and then pass
    it back.

    xmlorimagepath can be the actual path to the xml file or it can be the path to a tif
    with the same file name. See get_xml_filename and get_xmlroot for details."""
    # get necessary values from the xml
    xmlroot = get_xmlroot(xmlorimagepath)
    abfd = absCalFactor_dict(xmlroot)
    ebd = effectiveBandwidth_dict(xmlroot)
    distES = earthSunDistance(xmlroot)
    solarZenith = solarZenithAngle(xmlroot)
    # loop through the bands
    for band in range( len(bandarr) ):
        toa_arr = toa_reflectance(bandarr[band],abfd.values()[band],ebd.values()[band],Esun_od.values()[band],distES,solarZenith)
        if band==0:
            toabandarr = np.array([toa_arr])
        else:
            toabandarr = np.append(toabandarr,[toa_arr],axis=0)
    return toabandarr

def output_toa_reflectance(img_path,outpath=None,xmlpath=None):
    """
    Calculate the top of atmosphere reflectance for all bands in img and write out
    a geotiff. If outpath is provided, that is where the geotif will be written
    otherwise, it will be written to the same directory as img with _toa_ref appended to
    the filename before the extension.

    This is the method to use if you want to operate from the python console."""
    img = open_raster(img_path)
    # get the array of band arrays - defined in common.py
    bandarr = bandarr_from_ds(img)

    if xmlpath:
        toabandarr = toa_reflectance_multiband(bandarr,xmlpath)
    else:
        toabandarr = toa_reflectance_multiband(bandarr,img_path)

    if outpath==None:
        fext = img_path.split(os.path.extsep)[-1]
        outpath = img_path.replace(os.path.extsep+fext,'_toa_ref'+os.path.extsep+fext)

    output_gtif_like_img(img, toabandarr, outpath)

def output_toa_radiance(img_path,outpath=None,xmlpath=None):
    """
    Use the toa_radiance method to calculate toa radiance for all bands in img and
    write out a geotiff. If outpath is provided, that is where the geotif will be written
    otherwise, it will be written to the same directory as img with _toa_rad appended to
    the filename before the extension.

    This is the method to use if you want to operate from the python console."""
    img = open_raster(img_path)
    # get the array of band arrays - defined in common.py
    bandarr = bandarr_from_ds(img)
    if xmlpath:
        toabandarr = toa_radiance_multiband(bandarr,xmlpath)
    else:
        toabandarr = toa_radiance_multiband(bandarr,img_path)

    if outpath==None:
        fext = img_path.split(os.path.extsep)[-1]
        outpath = img_path.replace(os.path.extsep+fext,'_toa_rad'+os.path.extsep+fext)

    output_gtif_like_img(img, toabandarr, outpath)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Perform top of atmosphere reflectance (default) or top of atmosphere radiance correction on DigitalGlobe WorldView-2 imagery by reading XML file and applying corrections based on data in the XML file.')
    parser.add_argument('infile', nargs='?', help='The file that you would like to apply corrections to.', default='doctest')
    parser.add_argument('outfile', nargs='?', help='The corrected output file. If not specified, the output will have the same file name as the input with \'_corrected\' appended to it.')
    parser.add_argument('-v','--verbose', help='Print a bunch of stuff that you probably aren\'t interested in.', action='store_true')
    parser.add_argument('--radiance', help='Return top of atmosphere radiance instead of continuing on a returning top of atmosphere reflectance.', action='store_true')
    parser.add_argument('--xmlfile', help='Path to the xml file that contains the DigitalGlobe image data. When left unspecified, we will look for a file with the same name as the input image file with a .xml file extension.')
    parser.add_argument('--darkpixel', help='Include dark pixel subtraction correction. Look at references.txt and the doc string for dark_pixel_subtraction method for more information.', action='store_true')
    parser.add_argument('--dp_fraction', type=float, help='The fraction of darkest pixels to use with the dark pixel subtraction. Default is 0.001. See the doc string for dark_pixel_finder and dark_pixel_subtraction for more information.')
    parser.add_argument('--doctest', help='Run doctests for this module.', action='store_true')

    args = parser.parse_args()

    if args.doctest or args.infile=='doctest':
        import doctest
        doctest.testmod()
        if not args.doctest:
            print "This module just got tested. If that was not what you were aiming for run with '-h' for help."
        sys.exit(1)

    if not args.outfile:
        if args.infile.find('.TIF') != -1:
            args.outfile = args.infile.replace('.TIF','_corrected.TIF')
        elif args.infile.find('.tif') != -1:
            args.outfile = args.infile.replace('.tif','_corrected.tif')
        else:
            raise Exception("Your input file doesn't seem to have a tif extension so I can not come up with an output file name for you.")

    if args.xmlfile:
        xml_path = args.xmlfile
    else:
        # this works because get_xml_filename works out all the details later on
        xml_path = args.infile

    img = open_raster(args.infile)
    bandarr = bandarr_from_ds(img)
    if args.radiance:
        bandarr = toa_radiance_multiband(bandarr,xml_path)
    else:
        bandarr = toa_reflectance_multiband(bandarr,xml_path)

    if args.darkpixel:
        bandarr = dark_pixel_subtraction(bandarr)

    output_gtif_like_img(img,bandarr,args.outfile)

    print 'Results written to: %s' % args.outfile
