#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar  4 13:35:17 2019

@author: kujawski

defines folder structure for machine learning models

"""

from os.path import join, isdir,dirname
from os import mkdir
from datetime import datetime
from time import time

stamp = lambda: datetime.fromtimestamp(time()).strftime('%H:%M:%S').replace(':','_')

def create_folder(folder_path):
    if not isdir(folder_path): mkdir(folder_path)
    
def create_session_folders(top_path,sess_name,*stamp_addon):
    
    tstamp = stamp()
    for s in stamp_addon:
        tstamp = tstamp + '_{}'.format(s) 
    calc_dir = join(top_path,"calc_out_"+sess_name)
    sess_dir = join(calc_dir,tstamp)
    best_result_dir = join(sess_dir, "best_results")
    for folder in [calc_dir,sess_dir,best_result_dir]: create_folder(folder)
    return calc_dir,sess_dir,best_result_dir