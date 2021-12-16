# -*- coding: utf-8 -*-
"""
Created on Sun Nov 22 22:19:53 2020

©2020. Triad National Security, LLC. All rights reserved. This program was produced under U.S. Government contract 89233218CNA000001 for Los Alamos National Laboratory (LANL), which is operated by Triad National Security, LLC for the U.S. Department of Energy/National Nuclear Security Administration. All rights in the program are reserved by Triad National Security, LLC, and the U.S. Department of Energy/National Nuclear Security Administration. The Government is granted for itself and others acting on its behalf a nonexclusive, paid-up, irrevocable worldwide license in this material to reproduce, prepare derivative works, distribute copies to the public, perform publicly and display publicly, and to permit others to do so.

@author: weixin
"""

import os
import sys
import zipfile
import csv
import re

n=0
TP = 0
FP = 0
file_dir0 = r'D:\research\Ming_code_data\east_png_results_new'

with open("east_results_orig_labels_test.csv",'w', newline='',encoding = "utf-8") as f:   
    csv_write = csv.writer(f)                                       
    csv_head = ["imageID", "east_label"]
    csv_write.writerow(csv_head)
            
    for dirpath, dirnames, filenames in os.walk(file_dir0):  
        for txtfile in filenames :  
            if os.path.splitext(txtfile)[1] == '.txt': 
                n+=1
                print(n)
                name = os.path.splitext(txtfile)[0]
                name = name.split('_')
                imageID = name[1]
                txtfile_path = os.path.join(dirpath, txtfile)
                with open(txtfile_path, "r",encoding = "utf-8") as f1:       
                    text = f1.read()
                    a = re.findall("[F][[A-Za-z]?[G|g][.|,]?[ ]*[0-9][0-9]?", text)  
                    b = re.findall("Figure"+"[ ][0-9]", text)
                    e = re.findall(r'1.3|1.7|1.5|1.2|3.4|1.4', text)
                    #print(a)
                    #print(b)
                    
                    for x in range(0, len(b)): #if b is none, won't continue
                        if b[x] not in a:   #entlistall is list, can't use ".find"
                            a.append(b[x])
                    #print(a)
                    
                    for x in range(0, len(e)): #if b is none, won't continue
                        a.append(e[x])
                    
                    TP = TP +len(a)
                    
                    
                    text_new = re.sub(r"[F][[A-Za-z]?[G|g][.|,]?[ ]*[0-9][0-9]?",'',text)
                    text_new = re.sub(r"[F][i][g][u][r][e][ ][0-9]" ,'',text_new)
                    c = re.findall("[F][[A-Za-z][G|g|C][.|,|\s]", text_new)  
                    d = re.findall("[F][i][g][u][r][e]", text_new) 
                    print(c)
                    
                    for x in range(0, len(c)): #if b is none, won't continue
                       # if c[x] not in a:   
                            a.append(c[x])
                    
                    for x in range(0, len(d)): #if b is none, won't continue
                       # if c[x] not in a:   
                            a.append(d[x])
                    
                    print(a)
                    
                    FP = FP +len(a)
                    #if len(a) == 0:
                     #   c = re.findall("[F][[A-Za-z][G|g][.|,][ ]*[\n]", text)
                      #  print(c)
                csv_row = [imageID,a]
                csv_write.writerow(csv_row)
                
print("TP:", TP)
print("FP:", FP-TP)                

recall = TP/126
print('recall = ', recall*100,'%')

precision = TP/(TP+FP-TP)
print('precision = ', precision*100,'%')

F1 = (2*recall*precision)/(recall+precision)
print("F1 =",F1)
    
