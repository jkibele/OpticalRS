# -*- coding: utf-8 -*-
"""
DocStringer
===========

I'm trying to bang something together that'll process my numpydoc docstrings
into some markdown that I can use in my documentation. It's probably going to
take way longer than I currently think it's going to.
"""

import pypandoc
import OpticalRS

modules = ['ArrayUtils','Sagawa2010']

for m in modules:
    dstr = getattr(OpticalRS,m).__doc__
    print pypandoc.convert( dstr, 'markdown', format='rst' )