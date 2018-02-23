# -*- coding: utf-8 -*-
"""
Created on Thu Feb 22 10:57:34 2018

@author: Administrator
"""

import compare
import numpy as np
a=compare.load_and_align_data("C:/Users/Administrator/photos/modified",160, 44, 0.3)
b=compare.main(a)
c=np.array(b)
print(c.shape)