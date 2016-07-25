# OpticalRS
**Optical Remote Sensing Python Library**

OpticalRS is a free and open source Python implementation of passive optical remote sensing methods for the derivation of bathymetric maps and maps of submerged habitats. OpticalRS contains most of the code that makes up MORE-MAPS: the Marine Optical Remote sEnsing Map and Assessment Production System <sup>[1]</sup>. Additionally, OpticalRS includes my interpretation of several widely used methods from the optical remote sensing literature. Citations will be found throughout this documentation to give credit to the original authors.

This documentation is a work in progress and may not currently be up to date with the code. Please contact me if you need help. This library is offered for use without any warranty of any kind in the hope that it can save other researchers the time required to implement these methods from scratch. The bits of code contained here have had various levels of testing. You must validate any results yourself. Please let me know if you find anything wrong or if you have any questions.


## Features

* Linear transform depth estimation methods developed by David Lyzenga (see references)
* KNN depth estimation methods developed by Kibele and Shears
* Lyzenga's depth invariant index
* Sagawa's bottom reflectance index
* A water column correction method similar to Sagawa et al.'s based on Maritorena et al. 1994 (described in Jared Kibele's PhD thesis and an upcoming paper)
* Heaps of additional code for accuracy assessment and visualization

## References

1. Kibele, J., In Review. Submerged habitats from space: Increasing map production capacity with new methods and software. (PhD Thesis). University of Auckland.
2. Kibele, J., Shears, N.T., In Press. Non-parametric empirical depth regression for bathymetric mapping in coastal waters. IEEE Journal of Selected Topics in Applied Earth Observations and Remote Sensing.
2. Lyzenga, D.R., 1981. Remote sensing of bottom reflectance and water attenuation parameters in shallow water using aircraft and Landsat data. International Journal of Remote Sensing 2, 71–82. doi:10.1080/01431168108948342
2. Lyzenga, D.R., 1978. Passive remote sensing techniques for mapping water depth and bottom features. Appl. Opt. 17, 379–383. doi:10.1364/AO.17.000379
3. Lyzenga, D.R., Malinas, N.P., Tanis, F.J., 2006. Multispectral bathymetry using a simple physically based algorithm. Geoscience and Remote Sensing, IEEE Transactions on 44, 2251 –2259. doi:10.1109/TGRS.2006.872909
3. Maritorena, S., Morel, A., Gentili, B., 1994. Diffuse Reflectance of Oceanic Shallow Waters: Influence of Water Depth and Bottom Albedo. Limnology and Oceanography 39, 1689–1703.
4. Sagawa, T., Boisnier, E., Komatsu, T., Mustapha, K.B., Hattour, A., Kosaka, N., Miyazaki, S., 2010. Using bottom surface reflectance to map coastal marine areas: a new application method for Lyzenga’s model. International Journal of Remote Sensing 31, 3051–3064. doi:10.1080/01431160903154341
