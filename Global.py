# -*- coding: utf-8 -*-
"""
Created on Wed Mar 19 14:03:14 2025

@author: isabe
"""
import bw2data as bd
import bw2io as bi
import bw2calc as bc

import bw2analyzer as ba
import matrix_utils as mu
import bw_processing as bp

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


import difflib
import bw2data as bd
import difflib
import re


import os
print(os.getcwd())
import os
print(os.listdir())
#!/usr/bin/env python
# -*- coding: utf-8 -*-
import nbimporter
from module import Module
import nbimporter
# global variables
globals()["database"] =bd.Database("ecoinvent-3.10-cutoff")
file_path3 = 'Process_Caracteristic.xlsx'
globals()["df_Process_Caracteristic"] =pd.read_excel(file_path3)
file_path = 'Material.xlsx' 
file_path2 = 'Supplier.xlsx' 
file_path4 = 'LCI_Processes.xlsx'
file_path3 = 'Process_Caracteristic.xlsx'
globals()["df_Process_LCI"]  = pd.read_excel(file_path4) 
globals()["df_Material"] = pd.read_excel(file_path)  
df_Material = globals()["df_Material"] 

globals()["df_Supplier"]  = pd.read_excel(file_path2)  

database_bis=bd.Database("database_bis")
eidb=bd.Database("ecoinvent-3.10-cutoff")
raw_data=[]
