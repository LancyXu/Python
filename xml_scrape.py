# -*- coding: utf-8 -*-
"""
Created on Tue May 31 16:32:02 2016

@author: james.bradley
"""


num_periods = 6
state_id = 44
climate_div=0
num_months = 8
year = 2016
template = 'http://www.ncdc.noaa.gov/temp-and-precip/climatological-rankings/download.xml?parameter=tavg&state=%s&div=%s&month=%s&periods[]=%s&year=%s'
insert_these = (state_id, climate_div, num_months,num_periods,year)
template = template% insert_these

import requests
from lxml import objectify
response = requests.get(template).content  
root = objectify.fromstring(response)

my_wm_username = 'lxu08'

print my_wm_username
print root.data.value
print root.data.twentiethCenturyMean
print root.data.lowRank
print root.data.highRank