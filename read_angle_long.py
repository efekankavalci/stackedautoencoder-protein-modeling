# -*- coding: utf-8 -*-
"""
Created on Sun Dec 18 21:24:52 2022

@author: CS
"""
import csv
import numpy as np

def read_angle(textfile):
    angles=[]
    row=1
    with open(textfile) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=' ')
        for line in csv_reader:
            angles=np.append(angles, line[-1])
            row=row+1
    angles = np.array(angles)
    angles = angles.astype(float)
    #angles = angles.transpose(angles)
    angles = angles.reshape(-1,1)
    return angles