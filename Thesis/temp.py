# -*- coding: utf-8 -*-
"""
Created on Mon May 24 14:33:07 2021

@author: lucas
"""

[(i*40) for i in range(13)]
a = [[x]*100 for x in [(i*40) for i in range(1,13)]]
print(a)