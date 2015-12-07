# -*- coding: utf-8 -*-
import pandas as pd
wv2_bandnames = ['coastal','blue','green','yellow','red','rededge','nir1','nir2']

# from DigitalGlobe, 2012. DigitalGlobe Core Imagery Products Guide.
wv2_center_wavelength = [427, 478, 546, 608, 659, 724, 833, 949]

# from Jerlov's Marine Optics 2nd Addition:
wvlengths = [425,475,550,600,650,700] # 700 is as high as the table goes
c1_Kd = [0.36,0.17,0.12,0.30,0.45,0.65]
c3_Kd = [0.54,0.29,0.19,0.33,0.46,0.71]
c5_Kd = [0.78,0.43,0.30,0.40,0.54,0.80]
c7_Kd = [1.20,0.71,0.46,0.48,0.63,0.92]
o1_Kd = [0.022,0.018,0.063,0.235,0.36,0.56]
o2_Kd = [0.081,0.062,0.089,0.26,0.40,0.61]
o3_Kd = [0.16,0.116,0.12,0.295,0.445,0.66]
# These are matplotlib color names that more or less match the wavelengths
cnames = ['cyan','blue','green','yellow','red','maroon']
c1d = dict( zip(wvlengths,c1_Kd) )
c3d = dict( zip(wvlengths,c3_Kd) )
c5d = dict( zip(wvlengths,c5_Kd) )
c7d = dict( zip(wvlengths,c7_Kd) )
o1d = dict( zip(wvlengths,o1_Kd) )
o2d = dict( zip(wvlengths,o2_Kd) )
o3d = dict( zip(wvlengths,o3_Kd) )
cdict = dict( zip(wvlengths,cnames) )

wtype_names = ['Oceanic I', 'Oceanic II', 'Oceanic III', 'Coastal 1', 'Coastal 3',
          'Coastal 5', 'Coastal 7']
wtype_dict = dict( zip( wtype_names, (o1_Kd, o2_Kd, o3_Kd, c1_Kd, c3_Kd, c5_Kd, c7_Kd) ) )

jerlov_df = pd.DataFrame(wtype_dict, index=wvlengths)