# Getting Started

There are two ways of using `OpticalRS`. You can use the [OpticalRS QGIS Processing Scripts](#processing-scripts) or you can invoke the code directly using the [Python API](#python-api). Presently, the processing tools represent a small fraction of the `OpticalRS` functionality. Please see the appropriate page for more detail.

## Python API

http://www.scipy-lectures.org/

## Processing Scripts

Sagawa2010
==========

In Sagawa et al. 2010 [1]\_, an alternative to Lyzenga's depth invariant
index is proposed. Sagawa's reflectance index incorporates depth
measurements in an attempt to correct for water column attenuation. This
code is my attempt at implementing this method.

Notes
-----

Sagawa's starting place is equation 1 from Lyzenga 1978 [2]\_, rewritten
as (just the symbols have been changed):

$$L_i = L_{si} + a_i r_i e^{-K_igZ}$$

$Z$ is the depth and $g$ is some a geometric constant. Sagawa's
reflectance index is described as (equation 4, Sagawa et al. 2010):

$$Index_i = rac{ L_i - L_{si} }{ e^{-K_i g Z} }$$

for each band $i$.

I know how to calculate all of those terms except for $K_i$ and $g$. I
know how to get :math:\`rac{ K\_i }{ K\_j }\` (see
Lyzenga1981.attenuation\_coef\_ratio) but not just $K_i$. However,
Sagawa offers this statement:: "Using sea truth data, we plotted
satellite data against depth for sandy bottom types. The regression
curve of Lyzenga’s model is then obtained. From this curve $–Kg$ can be
deduced, as $g$ can be geometrically calculated from sun and satellite
zenith angles."

I think that by "Lyzenga's model", Sagawa is referring to equation 1
(the first equation in this notebook). So, if my dark pixel subtracted
image is $Y$ and a particular band $i$ of that image is $Y_i$, then I
can rewrite equation 1 as:

$$Y_i = a_i r_i e^{-K_i g Z}$$

If I say $a_i r_i = A_i$, $-K_i g = B_i$, and say the depth ($Z$) is $x$
then I can rewrite again as:

$$Y_i = A_i e^{B x}$$

Then I can log both sides and have:

$$\ln{ Y_i } = Bx + \ln{ A_i }$$

Then I can use my depth estimates as $x$ values and regress against the
logged reflectance values. The slope will give me $B = -K_i g$ and the
intercept will be $\ln{ A_i }$. Then I can use the $-K_i g$ for each
band to calculate $Index_i$ according to Sagawa's equation 4 and I don't
really have to worry about seperating $-K_i$ from $g$.

References
---------
