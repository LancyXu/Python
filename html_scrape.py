# -*- coding: utf-8 -*-
"""
Created on Wed Jun 21 11:02:01 2017

@author: lxu08
"""

import requests
from bs4 import BeautifulSoup as bsoup
    
my_wm_username = 'lxu08'
search_url = 'http://publicinterestlegal.org/county-list/'
response = requests.get(search_url, headers={
            "User-Agent": "Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/50.0.2661.102 Safari/537.36"}).content
     
# Put your program here
parsed_html = bsoup(response,"lxml")
target_text=parsed_html.find_all('tbody')
my_result_list=[]
for element in target_text:
    target_rows=element.find_all('tr')
    for tr in target_rows:
        my_result_list.append([str(td.strong.text)for td in tr])
print my_wm_username
print len(my_result_list)
print(my_result_list)
