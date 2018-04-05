# -*- coding: utf-8 -*-
"""
Created on Wed Mar 28 22:32:57 2018

@author: lxu08
"""

text=raw_input()
data=text.split("\n")
list=[]
for item in data:
    m=int(item.split()[0])
    n=int(item.split()[1])
    list.append([m,n])

result=[]
for pair in list:
    m=pair[0]
    n=pair[1]
    for i in range(m,n+1):
        h=int(i/100)
        t=int((i-h*100)/10)
        o=i-h*100-t*10
        if i==h**3+t**3+o**3:
            result.append(i)
    if len(result)>0:
        for j in result:
            print str(j),
    else:
        print "no"