# -*- coding: utf-8 -*-
"""
Created on Wed Mar 28 21:32:44 2018

@author: lxu08
"""

def sum_fun():
    text=raw_input()
    data=text.split("\n")
    list=[]
    for item in data:
        n=int(item.split()[0])
        m=int(item.split()[1])
        list.append([n,m])
    
    for pair in list:
        n=pair[0]
        m=pair[1]
        output=[n]
        for i in range(m-1):
            next_item=float(output[i]**0.5)
            output.append(next_item)
        sumup=round(sum(output),2)
        print "%.2f"%sumup
sum_fun()