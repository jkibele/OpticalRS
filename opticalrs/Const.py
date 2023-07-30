# -*- coding: utf-8 -*-
import pandas as pd
wv2_bandnames = ['coastal','blue','green','yellow','red','rededge','nir1','nir2']

# from DigitalGlobe, 2012. DigitalGlobe Core Imagery Products Guide.
wv2_center_wavelength = [427, 478, 546, 608, 659, 724, 833, 949]

# These are matplotlib color names that more or less match the wavelengths or
# (in the case of the NIR bands) at least look okay to display them.
cnames = ['cyan','blue','green','yellow','red','maroon','purple','darkviolet']

# from Jerlov 1976, Table XXVII, page 135:
# Downward irradiance attenuation coefficients
# Jerlov, N.G., 1976. Marine Optics. Elsevier.
jerlov_json = '{"I":{"350":0.062,"375":0.038,"400":0.028,"425":0.022,"450":0.019,"475":0.018,"500":0.027,"525":0.043,"550":0.063,"575":0.089,"600":0.235,"625":0.305,"650":0.36,"675":0.42,"700":0.56},"IA":{"350":0.078,"375":0.052,"400":0.038,"425":0.031,"450":0.026,"475":0.025,"500":0.032,"525":0.048,"550":0.067,"575":0.094,"600":0.24,"625":0.31,"650":0.37,"675":0.43,"700":0.57},"IB":{"350":0.1,"375":0.066,"400":0.051,"425":0.042,"450":0.036,"475":0.033,"500":0.042,"525":0.054,"550":0.072,"575":0.099,"600":0.245,"625":0.315,"650":0.375,"675":0.435,"700":0.58},"II":{"350":0.175,"375":0.122,"400":0.096,"425":0.081,"450":0.068,"475":0.062,"500":0.07,"525":0.076,"550":0.089,"575":0.115,"600":0.26,"625":0.335,"650":0.4,"675":0.465,"700":0.61},"III":{"350":0.32,"375":0.22,"400":0.185,"425":0.16,"450":0.135,"475":0.116,"500":0.115,"525":0.116,"550":0.12,"575":0.148,"600":0.295,"625":0.375,"650":0.445,"675":0.52,"700":0.66},"C1":{"350":1.2,"375":0.8,"400":0.51,"425":0.36,"450":0.25,"475":0.17,"500":0.14,"525":0.13,"550":0.12,"575":0.15,"600":0.3,"625":0.37,"650":0.45,"675":0.51,"700":0.65},"C3":{"350":1.7,"375":1.1,"400":0.78,"425":0.54,"450":0.39,"475":0.29,"500":0.22,"525":0.2,"550":0.19,"575":0.21,"600":0.33,"625":0.4,"650":0.46,"675":0.56,"700":0.71},"C5":{"350":2.3,"375":1.6,"400":1.1,"425":0.78,"450":0.56,"475":0.43,"500":0.36,"525":0.31,"550":0.3,"575":0.33,"600":0.4,"625":0.48,"650":0.54,"675":0.65,"700":0.8},"C7":{"350":3.0,"375":2.1,"400":1.6,"425":1.2,"450":0.89,"475":0.71,"500":0.58,"525":0.49,"550":0.46,"575":0.46,"600":0.48,"625":0.54,"650":0.63,"675":0.78,"700":0.92},"C9":{"350":3.9,"375":3.0,"400":2.4,"425":1.9,"450":1.6,"475":1.23,"500":0.99,"525":0.78,"550":0.63,"575":0.58,"600":0.6,"625":0.65,"650":0.76,"675":0.92,"700":1.1}}'

def jerlov_Kd():
    """
    Return a pandas dataframe containing table XXVII from Jerlov 1976, page 135:
    Downward irradiance attenuation coefficients. The units are 1/meter rather
    than 100/meter as in the original table and values for the wavelength of
    310nm have been excluded. The table is avaiable in its original units in
    `data/jerlov_tableXXVII.xls` and in these units in `data/jerlov_Kd.pkl`.

    References
    ----------
    Jerlov, N.G., 1976. Marine Optics. Elsevier.
    """
    # I have to do this with the column names to get the order right
    colnames = [u'I', u'IA', u'IB', u'II', u'III', u'C1', u'C3', u'C5', u'C7', u'C9']
    return pd.read_json(jerlov_json)[colnames]

# Approximate albedo values for kelp and sand. These values were estimated from
# plots in Werdell and Roesler, 2003 for WorldView-2 wavelengths. They are,
# therefore, pretty loose estimates.
# Werdell, P.J., Roesler, C.S., 2003. Remote assessment of benthic substrate
# composition in shallow waters using multispectral reflectance. Limnology and
# Oceanography 48, 557–567.
# from figure 4A
kelp_albedo = dict( zip(wv2_center_wavelength,[0.025,0.025,0.035,0.05,0.045,0.03]) )
# from the brighter portion of figure 4E
sand_albedo = dict( zip(wv2_center_wavelength,[0.05,0.065,0.08,0.11,0.12,0.12]) )
# from figure 4F
tropical_sand_albedo = dict(zip(wv2_center_wavelength,
                                [0.23,0.3,0.36,0.42,0.48,0.5]))
# Rd values from Werdell and Roesler 2003 figure 4C
LI_Rd = dict(zip(wv2_center_wavelength,[0.01,0.02,0.025,0.018,0.008,0.007]))
# Rd values from Werdell and Roesler 2003 figure 4D
B_Rd = dict(zip(wv2_center_wavelength,[0.075,0.055,0.015,0.01,0.005,0.001]))
# Kd values from Werdell and Roesler 2003 figure 2A
LI_Kd = dict(zip(wv2_center_wavelength,[0.79,0.54,0.42,0.5,0.7,0.8]))
# Kd values from Werdell and Roesler 2003 figure 2B
B_Kd = dict(zip(wv2_center_wavelength,[0.1,0.06,0.1,0.25,0.4,0.45]))
