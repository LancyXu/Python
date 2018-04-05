# -*- coding: utf-8 -*-
"""
Created on Wed Mar 28 23:06:59 2018

@author: lxu08
"""

text=raw_input()
data=text.split("\n")
list=[int(k) for k in data]

for item in list:
    result=1
    while result==1 and item>1:
        if item%2!=0:
            result=0
        else:
            item=item/2
    if result==1:
        print 'true'
    else: print 'false'