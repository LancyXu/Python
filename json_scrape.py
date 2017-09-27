# -*- coding: utf-8 -*-
"""
Created on Wed Jun 21 11:06:43 2017

@author: jrbrad
"""

import requests

my_wm_username = 'lxu08'
search_url = 'http://buckets.peterbeshai.com/api/?player=201939&season=2015'
response = requests.get(search_url, headers={
            "User-Agent": "Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/50.0.2661.102 Safari/537.36"})#tell the website that i am not a robot and i use browser to see the data
numJumpShotsAttempt = 0
numJumpShotsMade = 0
percJumpShotMade = 0.0
# Write your program here to populate the appropriate variables
shot_type=[]
shot_made=[]
for shot in response.json():
    if shot.ACTION_TYPE=='Jump Shot':
        numJumpShotsAttempt=numJumpShotsAttempt+1
        if shot['EVENT_TYPE']=='Made Shot':
            numJumpShotsMade=numJumpShotsMade+1
#for shot in response.json():  #.json makes dictionary
    #shot_type=str(shot['ACTION_TYPE'])
    #shot_made=shot['SHOT_MADE_FLAG']
   # mydic={shot_type:shot_made}
    #key_list = mydic.keys()
    #for key in key_list:
        #if key=='Jump Shot':
            #numJumpShotsAttempt=numJumpShotsAttempt+1
        #if key=='Jump Shot' and mydic[key]==1:
           # numJumpShotsMade=numJumpShotsMade+1""""
percJumpShotMade=float(numJumpShotsMade)/numJumpShotsAttempt             
print my_wm_username
print numJumpShotsAttempt
print numJumpShotsMade
print percJumpShotMade