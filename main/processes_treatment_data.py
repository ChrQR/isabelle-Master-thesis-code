# -*- coding: utf-8 -*-
"""
Created on Wed Mar 19 14:15:23 2025

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


from geopy.geocoders import Nominatim
from geopy.distance import geodesic

import os
print(os.getcwd())
import os
print(os.listdir())
#!/usr/bin/env python
# -*- coding: utf-8 -*-
import nbimporter
from module import Module
import nbimporter
from modularsystem import ModularSystem
import Material_Treatment_Data
import uuid
bd.projects.set_current("default")

import pandas as pd

# Replace 'file.xlsx' with your actual file path
file_path3 = 'Process-EF.xlsx'
file_path4 = 'LCI_Processes.xlsx'


# Read the Excel file into a DataFrame
# global variables
df_Process_EF = pd.read_excel(file_path3)
df_Process_LCI = pd.read_excel(file_path4)
database=bd.Database("ecoinvent-3.10-cutoff")
database_2030=bd.Database("ei_cutoff_3.10_remind_SSP2-Base_2030 2025-02-16") 
database_2050=bd.Database("ei_cutoff_3.10_remind_SSP2-Base_2050 2025-02-16") 

database_bis=bd.Database("database_bis")
database_bis_2030=bd.Database("database_bis_2030")
database_bis_2050=bd.Database("database_bis_2050")
raw_data=[]
# Display the first few rows of the DataFrame


def find_closest_match(database,search_term, max_words=5):
    
    looking_for = next(
        (entry for entry in database if entry['name'] == search_term), 
        None
    )
    if looking_for:
        return(looking_for)
        
  

    
    search_term = str(search_term)
    def normalize_name(name):
        """Remove special characters and normalize the name."""
        return re.sub(r"[^\w\s]", "", name).lower()
    
    search_terms = search_term.split()
    
    def get_best_match(term):
        normalized_term = normalize_name(term)
        normalized_names = [normalize_name(a['name']) for a in database]
        matches = difflib.get_close_matches(normalized_term, normalized_names, n=5, cutoff=0.5)
        return matches

    for i in range(len(search_terms), 0, -1):
        reduced_term = " ".join(search_terms[:i])
        best_matches = get_best_match(reduced_term)
        
        if best_matches:
            matching_activities = [
                a for a in database 
                if normalize_name(a['name']) in best_matches
            ]
            
            rer_activities = [a for a in matching_activities if "RER" in a.get("location", "")]
            if rer_activities:
                return rer_activities[0]
            
            row_activities = [a for a in matching_activities if "RoW" in a.get("location", "")]
            if row_activities:
                return row_activities[0]
            
            return matching_activities[0]
    
    return None

def Create_Heat_Activity(database_bis, database, machine_name):   
    print('database_bis')
    print(database_bis)
    print('database')
    print(database)

    # Check if the process with this code already exists
    existing_process = next((act for act in database_bis if act['code'] == (machine_name + "_{heat}")), None)

    if existing_process:
        # Overwrite or update the existing process
        for exchange in list(existing_process.exchanges()):
            exchange.delete()  # Remove the exchange
        print(f"Deleted all existing exchanges for process: {machine_name}")
        # Overwrite or update the existing process
        existing_process['name'] = (machine_name + "_{heat}")
        existing_process['unit'] = 'min'
        existing_process.save()
        print(f"Updated existing process: {existing_process}")
        globals()[f"Process_heat_{machine_name}"] = existing_process
        globals()[f"Process_heat_{machine_name}"].save()# Use the existing process object
    else:
        # Create a new process
        globals()[f"Process_heat_{machine_name}"] = database_bis.new_node(code=(machine_name + "_{heat}"), name=(machine_name + "_{heat}"), unit="min")
        print(f"Created new process: {machine_name}")
        globals()[f"Process_heat_{machine_name}"].save()
    
    # Filter the rows where the name is machine_name
    filtered_rows = df_Process_LCI[df_Process_LCI["name"] == machine_name]

    # Check if any row exists with the given machine_name
    if filtered_rows.empty:
        print(f"No row found for machine_name: {machine_name}")
        return None  # or raise an error if you prefer

    # Get the first row (if there's at least one row)
    row = filtered_rows.iloc[0]

    # Iterate through the columns from 7 onward
    column_name='market for heat, district or industrial, natural gas'
    cell_value = row[column_name]
        
    print('cell_value')
    print(cell_value)
    print('column_name')
    print(column_name)
    # Ensure the cell_value is not empty or NaN
    if pd.notna(cell_value) and cell_value != "" and cell_value != "0":
        print('not empty')
        # Find the closest match for the column name
        input_name = database_2050.get('bd1f664bf4604b04864aa0bb912da7d7')
        print(input_name)        
        # Create a new exchange with the corresponding input and amount
        globals()[f"Process_heat_{machine_name}"].new_exchange(input=input_name, amount=cell_value, type='technosphere',unit='min').save()

    # Add the production exchange after processing all columns
    globals()[f"Process_heat_{machine_name}"].new_exchange(input=globals()[f"Process_heat_{machine_name}"], amount=1, type='production',unit='min').save()

    return globals()[f"Process_heat_{machine_name}"]


def Create_Machine_Activity(database, database_bis,machine_name):   
    
    # Check if the process with this code already exists
    existing_process = next((act for act in database_bis if act['code'] == (machine_name + "_{machine}")), None)

    if existing_process:
        # Overwrite or update the existing process
        for exchange in list(existing_process.exchanges()):
            exchange.delete()  # Remove the exchange
        print(f"Deleted all existing exchanges for process: {machine_name}")
        # Overwrite or update the existing process
        existing_process['name'] = (machine_name + "_{machine}")
        existing_process['unit'] = 'min'
        print(f"Updated existing process: {existing_process}")
        globals()[f"Process_machine_{machine_name}"] = existing_process
        globals()[f"Process_machine_{machine_name}"].save()
        

    else:
        # Create a new process
        globals()[f"Process_machine_{machine_name}"] = database_bis.new_node(code=(machine_name + "_{machine}"), name=(machine_name + "_{machine}"), unit="min")
        print(f"Created new process: {machine_name}")
        globals()[f"Process_machine_{machine_name}"].save()
    
    # Filter the rows where the name is machine_name
    filtered_rows = df_Process_LCI[df_Process_LCI["name"] == machine_name]

    # Check if any row exists with the given machine_name
    if filtered_rows.empty:
        print(f"No row found for machine_name: {machine_name}")
        return None  # or raise an error if you prefer

    # Get the first row (if there's at least one row)
    row = filtered_rows.iloc[0]

    # Iterate through the columns from 7 onward, the unit has been changed so the number are in kg/min
    column_name='market for metal working machine, unspecified'
    cell_value = row[column_name]
        
    print('cell_value')
    print(cell_value)
    print('column_name')
    print(column_name)
    # Ensure the cell_value is not empty or NaN
    if pd.notna(cell_value) and cell_value != "" and cell_value != "0":
        print('not empty')
        # Find the closest match for the column name
        globals()[f"machine_{machine_name}"] = find_closest_match(database, column_name, max_words=5)
        print(globals()[f"machine_{machine_name}"])
                
        # Create a new exchange with the corresponding input and amount
        globals()[f"Process_machine_{machine_name}"].new_exchange(input=globals()[f"Process_machine_{machine_name}"], amount=cell_value, type='technosphere',unit='min').save()

        # Add the production exchange after processing all columns
        
    globals()[f"Process_machine_{machine_name}"].new_exchange(input=globals()[f"Process_machine_{machine_name}"], amount=1, type='production',unit='min').save()

    return globals()[f"Process_machine_{machine_name}"]

def Create_Input_Activity(database,database_bis, machine_name):   
    
    # Check if the process with this code already exists
    existing_process = next((act for act in database_bis if act['code'] == (machine_name + "_{input}")), None)

    if existing_process:
        # Overwrite or update the existing process
        for exchange in list(existing_process.exchanges()):
            exchange.delete()  # Remove the exchange
        print(f"Deleted all existing exchanges for process: {machine_name}")
        # Overwrite or update the existing process
        existing_process['name'] = (machine_name + "_{input}")
        existing_process['unit'] = 'min'
        existing_process.save()
        print(f"Updated existing process: {existing_process}")
        globals()[f"Process_input_{machine_name}"] = existing_process  # Use the existing process object
        
        globals()[f"Process_input_{machine_name}"].save()
    else:
        # Create a new process
        globals()[f"Process_input_{machine_name}"] = database_bis.new_node(code=(machine_name + "_{input}"), name=(machine_name + "_{input}"), unit="min")
        print(f"Created new process: {machine_name}")
        globals()[f"Process_input_{machine_name}"].save()
    
    # Filter the rows where the name is machine_name
    filtered_rows = df_Process_LCI[df_Process_LCI["name"] == (machine_name)]

    # Check if any row exists with the given machine_name
    if filtered_rows.empty:
        print(f"No row found for machine_name: {(machine_name)}")
        return None  # or raise an error if you prefer

    # Get the first row (if there's at least one row)
    row = filtered_rows.iloc[0]

    # Iterate through the columns from 7 onward
    for column_name in df_Process_LCI.columns[7:]:
        # Access the cell value in the current column for the current row
        cell_value = row[column_name]
        
        print('cell_value')
        print(cell_value)
        print('column_name')
        print(column_name)
        # Ensure the cell_value is not empty or NaN
        if pd.notna(cell_value) and cell_value != "" and cell_value != "0":
            print('not empty')
            # Find the closest match for the column name
            input_name = find_closest_match(database,column_name, max_words=5)
            print(input_name)
                
            # Create a new exchange with the corresponding input and amount
            globals()[f"Process_input_{machine_name}"].new_exchange(input=input_name, amount=cell_value, type='technosphere',unit='min').save()

    # Add the production exchange after processing all columns
    globals()[f"Process_input_{machine_name}"].new_exchange(input=globals()[f"Process_input_{machine_name}"], amount=1, type='production',unit='min').save()

    return globals()[f"Process_input_{machine_name}"]

def Create_Electricity_Activity(database,database_bis, machine_name):   

    # Check if the process with this code already exists
    existing_process = next((act for act in database_bis if act['code'] == (machine_name + "_{electricity}")), None)

    if existing_process:
        # Overwrite or update the existing process
        for exchange in list(existing_process.exchanges()):
            exchange.delete()  # Remove the exchange
        print(f"Deleted all existing exchanges for process: {machine_name}")
        # Overwrite or update the existing process
        existing_process['name'] = (machine_name + "_{electricity}")
        existing_process['unit'] = 'min'
        existing_process.save()
        print(f"Updated existing process: {existing_process}")
        globals()[f"Process_electricity_{machine_name}"] = existing_process  # Use the existing process object
        globals()[f"Process_electricity_{machine_name}"].save()
    else:
        # Create a new process
        globals()[f"Process_electricity_{machine_name}"] = database_bis.new_node(code=(machine_name + "_{electricity}"), name=(machine_name + "_{electricity}"), unit="min")
        print(f"Created new process: {machine_name}")
        globals()[f"Process_electricity_{machine_name}"].save()
    
    # Filter the rows where the name is machine_name
    filtered_rows = df_Process_LCI[df_Process_LCI["name"] == (machine_name)]

    # Check if any row exists with the given machine_name
    if filtered_rows.empty:
        print(f"No row found for machine_name: {(machine_name)}")
        return None  # or raise an error if you prefer

    # Get the first row (if there's at least one row)
    row = filtered_rows.iloc[0]

    column_name='Electricity (W)'
    cell_value = row[column_name]
    quantity=cell_value/1000/60

    if pd.notna(cell_value) and cell_value != "" and cell_value != "0":
            print('not empty')
            # Find the closest match for the column name
            input_name = database_2050.get('e57a5b543b8bb28e332b9a065dfed51c')
            print(input_name)


    globals()[f"Process_electricity_{machine_name}"].new_exchange(input=input_name, amount=quantity, type='technosphere',unit='min').save()

    # Add the production exchange after processing all columns
    globals()[f"Process_electricity_{machine_name}"].new_exchange(input=globals()[f"Process_electricity_{machine_name}"], amount=1, type='production',unit='min').save()

    return globals()[f"Process_electricity_{machine_name}"]

# We will create the carbon footprint of the manufacture processs with the 4 elements: electricity, machine, input and heat
def Create_EF_machine(database,database_bis,machine_name):
    
    existing_process = next((act for act in database_bis if act['code'] == machine_name ), None)
    
    if existing_process:
        # Overwrite or update the existing process
        for exchange in list(existing_process.exchanges()):
            exchange.delete()  # Remove the exchange
        print(f"Deleted all existing exchanges for process: {machine_name}")
        
        existing_process['name'] = (machine_name )
        existing_process['unit'] = "min" 
        existing_process.save
        print(f"Updated existing process: {existing_process}")
        globals()[f"{machine_name}_Process"]= existing_process  # Use the existing process object
        globals()[f"{machine_name}_Process"].save()
    else:
         # Create a new process
        globals()[f"{machine_name}_Process"] = database_bis.new_node(code=machine_name, name=machine_name, unit="min" )
        print(f"Created new process: {machine_name}")
        globals()[f"{machine_name}_Process"].save()

        
    globals()[f"{machine_name}_input"]= Create_Input_Activity(database,database_bis,machine_name)
    globals()[f"{machine_name}_heat"]= Create_Heat_Activity(database,database_bis,machine_name)
    globals()[f"{machine_name}_machine"]= Create_Machine_Activity(database,database_bis,machine_name)
    globals()[f"{machine_name}_electricity"]= Create_Electricity_Activity(database,database_bis,machine_name)
   
    globals()[f"{machine_name}_Process"].new_exchange(input=globals()[f"{machine_name}_heat"], amount=1, type='technosphere',unit='min').save()
    globals()[f"{machine_name}_Process"].new_exchange(input=globals()[f"{machine_name}_input"], amount=1, type='technosphere',unit='min').save()
    
    globals()[f"{machine_name}_Process"].new_exchange(input=globals()[f"{machine_name}_machine"], amount=1, type='technosphere',unit='min').save()
    globals()[f"{machine_name}_Process"].new_exchange(input=globals()[f"{machine_name}_electricity"], amount=1, type='technosphere',unit='min').save()
    globals()[f"{machine_name}_Process"].new_exchange(input=globals()[f"{machine_name}_Process"], amount=1, type='production',unit='min').save()
    return(globals()[f"{machine_name}_Process"])


def all_process_calculation(database,database_bis) :
    # iterate the rows
    for index, row in df_Process_LCI.iterrows():
        # Access row data by column names
        machine_name=(row["name"]).strip()
        Process = Create_EF_machine(database,database_bis,machine_name)
        print('-------------------------------------------------')
        print(Process)
    return()
def Create_Modules_Processes():
    
    
    # Initialize a list to store the created activities
    created_modules = []
    
    for index, row in df_Process_LCI.iterrows():
        # Access row data by column names
        machine_name=(row["name"]) 
       
        
        
        # Get the process 
        Process_Activity = database_bis.get(machine_name)  # Assuming 'trans' is already created
        #Process_Activity = next(entry for entry in database_bis if entry['name'] == machine_name and entry['location'] == 'GLO')
        
        
        my_object = {
        'name': f'Module_Process_{machine_name}',
        'outputs': [
            (Process_Activity.key, machine_name, 1)
        ],
        'chain': [Process_Activity.key],
        'cuts': [],
        'output_based_scaling': False
        }
        print(my_object)
    # Return the list of created modules
    return created_modules
    
#Create modules processes: the activity already exists so it just has to take the machine name  and create a module cfor this machine
def Create_Modules_Process(machine_name, database_bis, machine_id):
    
    

    # Get the process 
    Process_Activity = database_bis.get(machine_name)  # Assuming 'trans' is already created
    #Process_Activity = next(entry for entry in database_bis if entry['name'] == machine_name and entry['location'] == 'GLO')        
    my_object = {
    'name': f'Module_Process_{machine_name}',
    'outputs': [( Process_Activity.key, machine_id, 1)],
    'chain': [Process_Activity.key],
    'cuts': [],
    'output_based_scaling': True
    }
        
    # Return the list of created modules
    return my_object