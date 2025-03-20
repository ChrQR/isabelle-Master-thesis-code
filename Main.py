# -*- coding: utf-8 -*-
"""
Created on Wed Mar 19 14:07:02 2025

@author: isabe
"""

import bw2data as bd 
import bw2io as bi 
import bw2calc as bc
from premise import *

import bw2analyzer as ba 
import matrix_utils as mu 
import bw_processing as bp

import nbimporter 
from module import Module 
import nbimporter 
from modularsystem import ModularSystem
from Processes_Treatment_Data import Create_Modules_Process 
from Material_Treatment_Data import get_coordinates,give_EF_Material,find_closest_match,Create_Modules_Subproduct,Create_Modules_Subproduct_2030,Create_Modules_Subproduct_2050

import Processes_Treatment_Data 
import matplotlib.pyplot as plt 
import numpy as np 
import pandas as pd 
import uuid

import difflib 
import bw2data as bd 
import difflib 
import re 
from geopy.geocoders import Nominatim 
from geopy.distance import geodesic 
import nbimporter 
from module import Module 
import nbimporter 
from modularsystem import ModularSystem

import os
print(os.getcwd())


class LCA:
    def __init__(self):
        self.cached_results = {}

    def lca_processes(self, method, process_names):
        if (method, tuple(process_names)) in self.cached_results:
            return self.cached_results[(method, tuple(process_names))]  # Returning cached result
        result = self._compute_lca(method, process_names)
        self.cached_results[(method, tuple(process_names))] = result
        return result



file_path = 'Material.xlsx' 
file_path2 = 'Supplier.xlsx' 
file_path4 = 'LCI_Processes.xlsx'
df_Process_LCI = pd.read_excel(file_path4) 
df_Material = pd.read_excel(file_path) 
df_Supplier = pd.read_excel(file_path2) 
main_address='Hedemølle Erhvervsvej 23. DK-8850 Bjerringbro' 
database=bd.Database("ecoinvent-3.10-cutoff") 
database_2030=bd.Database("ei_cutoff_3.10_remind_SSP2-Base_2030 2025-02-16") 
database_2050=bd.Database("ei_cutoff_3.10_remind_SSP2-Base_2050 2025-02-16") 

database_bis=bd.Database("database_bis")
database_bis_2030=bd.Database("database_bis_2030")
database_bis_2050=bd.Database("database_bis_2050")
file_path3 = 'Process_Caracteristic.xlsx'
df_Process_Caracteristic =pd.read_excel(file_path3)
electricity_price=3
raw_data=[]

geolocator = Nominatim(user_agent="distance_calculator", timeout=10) 
bd.projects.set_current("default")

lmp = ModularSystem()
lmp_2030 = ModularSystem()
lmp_2050 = ModularSystem()


database_2030=bd.Database("ei_cutoff_3.10_remind_SSP2-Base_2030 2025-02-16") 
database_2050=bd.Database("ei_cutoff_3.10_remind_SSP2-Base_2050 2025-02-16") 


increase_percentage_price=1
# Materials dictionary, the quantity are always in kg and the specificity is either produced or purchased
product = {
    "specificity": {
        "name": "Final product",
        "id": '001',
        
    },
    "routing": {

        "fiber laser": {
            "name": "1 kW Fiber laser machine",
            "time": 0.018,
            "shape": "Flat",
            "thickness": 8,
            "tolerance":1 ,
            "roughness": None,
            "mass": 1
            
            
        },
        "CNC lathe": {
            "name": "CNC Lathe",
            "time": 15.6,
            "shape": "Flat",
            "thickness": 8,
            "tolerance":1 ,
            "roughness": None,
            "mass": 1
            
            
        },

        "silbning": {
            "name": "Grinding",
            "time": 3,
            "shape": "Flat",
            "thickness": None,
            "tolerance":None ,
            "roughness": None
          
        }

        },
        

    "materials": {
        # "M001": {
        #     "name": "Polyoxymethylene (POM, Acetal)",
        #     "specificity": "Purchased",
        #     "supplier": "ABC Steel Co.",
        #     "address": "123 Industrial St, Metaltown",
        #     "quantity": 5,
        #     "scrap": 0.05,
        #     "routing": ["P001"]
        # },
        # "M002": {
        #     "name": "Galvanized steel",
        #     "specificity": "Purchased",
        #     "supplier": "XYZ Metals Ltd.",
        #     "address": "123 Industrial St, Metaltown",
        #     "quantity": 3,
        #     "scrap": 0.05,
        #     "routing": ["P002"]
        
        # },
        "100-00360": {
            "name": "Aluminum cast alloy, A356.0, permanent mold cast, T6",
            "specificity": "Purchased",
            "supplier": "Nexeo Plastics",
            "address": "123 Industrial St, Metaltown",
            "quantity": 1,
            "scrap": 0,
            "routing": ["fiber laser","CNC lathe","silbning"],
            "constraints": {
                "Yield_Strength": (">", 2.16),  # g/cm³
                "Young_Modulus": (">=", 6.91),  # W/(m·K)
                "Density": ("<", 3000),  # W/(m·K)
                "toughness": (">=", 1),  # W/(m·K)
            },
            "material_index_formula": "Young_Modulus* Yield_Strength / (Density)",

            "material_index_maxmin": "Max"
            
        },
    }
}


# Routing dictionary

'''bi.import_ecoinvent_release(
    version="3.10",
    system_model='cutoff',
    username='EcoInventQSA2',
    password='EcoinventDTUQSA22'
) '''

if 'ecoinvent-3.10-cutoff' in bd.databases:
    print('ecoinvent 3.10 is already present in the project')
else:
    bi.import_ecoinvent_release(
        version='3.10',
        system_model='cutoff', # can be cutoff / apos / consequential / EN15804
    username='EcoInventQSA2',
    password='EcoinventDTUQSA22'
    )


def Create_all_Modules(database,database_bis,product,raw_data,df_Material,df_Process_Caracteristic,df_Supplier,increase_percentage_price):
    
    #this list have double with first the activity key and second the quantity
    material=[]
    material_id_list=[]
    quantity=[]
    

    process_names = df_Process_LCI['name'].tolist()
    
    #ADD THE MODULE OF THE SUBPRODUCT
    print('NEW MATERIAL')
    columns = [
    "Sub_Product_name",
    "Quality"]
    quality_df = pd.DataFrame(columns=columns)

    columns_energy = [
    "Process_name",
    "kWh"]
    energy=pd.DataFrame(columns=columns_energy)
 
    
    for material_id, material_info in product["materials"].items():
        print(material_id)
        process_list=[]
        #for each material of the BOM find similar
        material_name = material_info["name"]
        print(material_name)
        material_names = df_Material['Material'].tolist()
        # Get the closest matches using difflib
        matches = difflib.get_close_matches(material_name, material_names, n=1, cutoff=0.5)
        print('match')
        print(matches)
        density_base_material = df_Material.loc[df_Material['Material'] == matches[0], 'Density'].values[0]
        density_base_material = float(density_base_material)
        material_mass=material_info["quantity"]
        print('MATERIALLL VOLUMEEEE')
        material_volume=material_mass/density_base_material
        print(material_volume)
        #find the wo that are associated with material_id
        for routing_id in material_info["routing"]:
            matches_process = difflib.get_close_matches(product["routing"][routing_id]["name"], process_names, n=1, cutoff=0.5)
            process_list.append((matches_process[0], product["routing"][routing_id]["time"],product["routing"][routing_id]["shape"],product["routing"][routing_id]["tolerance"],product["routing"][routing_id]["thickness"],product["routing"][routing_id]["roughness"],material_info["quantity"],routing_id))
               

        # for routing_id, routing_info in product["routing"].items():
        #     if routing_info["materials"][0] == material_id:
        #         matches_process = difflib.get_close_matches(routing_info["name"], process_names, n=1, cutoff=0.5)
        #         process_list.append((matches_process, routing_info["time"],routing_info["shape"],routing_info["tolerance"],routing_info["thickness"],routing_info["roughness"],routing_info["mass"]))
        
        #use a fucntion that takes in input the processes used and the matches material and return the modules of all material and processes associated
        print('STRAT SUB PRODUCT for')
        print(material_id)
        print('process_list')
        print(process_list)
        print('matches[0]')
        print(matches[0])

        material_index_formula= material_info["material_index_formula"]
        material_index_maxmin= material_info["material_index_maxmin"]
        constraints=material_info["constraints"]
                
        print(constraints)
        print(process_list)
        
        sub_product_module,key,micro_quality_df,micro_energy=Create_Modules_Subproduct(matches[0], main_address, geolocator, database, database_bis, material_id,material_volume,process_list,df_Material,df_Process_Caracteristic,df_Supplier,constraints,material_index_formula, material_index_maxmin,increase_percentage_price)
        quality_df = pd.concat([quality_df, micro_quality_df], ignore_index=True)
        energy = pd.concat([energy, micro_energy], ignore_index=True)
        print('-------------------------sub_product_module-------------------------------')
        print(sub_product_module)
        raw_data.extend(sub_product_module)
        material.append(key)
        print(material)
        print(key)
        print(sub_product_module)
        activity = bd.get_activity(('database_bis', key))
        print(' activity')
        print(activity)
        
        material_id_list.append(material_id)
        quantity.append(material_info["quantity"])
    print('MATERIAL KEY:')
    print(material)
    print('MATERIAL ID:')
    print(material_id_list)
    
    #ADD THE MODULE OF THE FINAL PRODUCT, need to first create the final product activity by taking the subproducts

    
    #ADD THE MODULE OF THE FINAL PRODUCT, need to first create the final product activity by taking materials and process and after the material taken will be cut when the module will be created
    # Create a new activity for the final product
    cut=[]
    chain=[]
    product_id=product["specificity"]["id"]
    existing_process = next((act for act in database_bis if act['code'] == product_id), None)
    if existing_process:
        # Overwrite or update the existing process
        for exchange in list(existing_process.exchanges()):
            exchange.delete()  # Remove the exchange
        print(f"Deleted all existing exchanges for process: {product_id}")
        # Overwrite or update the existing process
        existing_process['name'] = product["specificity"]["id"]
        existing_process['unit'] = 'unit'
        existing_process.save()
        print(f"Updated existing process: {existing_process}")
        Final_module = existing_process  # Use the existing process object
    else:
        # Create a new activity
        Final_module = database_bis.new_node(code= product["specificity"]["id"], name= product["specificity"]["id"], unit="unit")
        print(Final_module)
        Final_module.save()
    for i, j , k in zip(material, quantity, material_id_list):
        print('KEY')
        print(database_bis.get(i))
        Final_module.new_exchange(input=bd.get_activity(('database_bis', i)), amount=j, type='technosphere', unit='unit').save()
        cut.append((('database_bis',i),Final_module.key,k,j))
        
        chain.append(('database_bis',i))
    
    Final_module.new_exchange(input=Final_module, amount=1, type='production', unit='kg').save()
    # Create module
    raw_data.append({
        'name':'Assembly',
        'outputs': [
           ( Final_module.key, product["specificity"]["id"], 1.0)
        ],
        'chain': [( Final_module.key)] + chain,
        'cuts': cut,
        'output_based_scaling': True
    })
    print('THe final product has been added')
    return(raw_data,quality_df,energy)


def Create_all_Modules_2030(database_2030,database_bis_2030,product,raw_data,df_Material,df_Process_Caracteristic,df_Supplier,increase_percentage_price):
    
    #this list have double with first the activity key and second the quantity
    material=[]
    material_id_list=[]
    quantity=[]
    

    process_names = df_Process_LCI['name'].tolist()
    
    #ADD THE MODULE OF THE SUBPRODUCT
    print('NEW MATERIAL')
    for material_id, material_info in product["materials"].items():
        print(material_id)
        process_list=[]
        #for each material of the BOM find similar
        material_name = material_info["name"]
        print(material_name)
        material_names = df_Material['Material'].tolist()
        # Get the closest matches using difflib
        matches = difflib.get_close_matches(material_name, material_names, n=1, cutoff=0.5)
        print('match')
        print(matches)
        #find the wo that are associated with material_id
        for routing_id in material_info["routing"]:
            matches_process = difflib.get_close_matches(product["routing"][routing_id]["name"], process_names, n=1, cutoff=0.5)
            process_list.append((matches_process[0], product["routing"][routing_id]["time"],product["routing"][routing_id]["shape"],product["routing"][routing_id]["tolerance"],product["routing"][routing_id]["thickness"],product["routing"][routing_id]["roughness"],material_info["quantity"],routing_id))
               

        # for routing_id, routing_info in product["routing"].items():
        #     if routing_info["materials"][0] == material_id:
        #         matches_process = difflib.get_close_matches(routing_info["name"], process_names, n=1, cutoff=0.5)
        #         process_list.append((matches_process, routing_info["time"],routing_info["shape"],routing_info["tolerance"],routing_info["thickness"],routing_info["roughness"],routing_info["mass"]))
        
        #use a fucntion that takes in input the processes used and the matches material and return the modules of all material and processes associated
        print('STRAT SUB PRODUCT for')
        print(material_id)
        print('process_list')
        print(process_list)
        print('matches[0]')
        print(matches[0])
        
        density_base_material = df_Material.loc[df_Material['Material'] == matches[0], 'Density'].values[0]
        density_base_material = float(density_base_material)
        material_mass=material_info["quantity"]
        print('MATERIALLL VOLUMEEEE')
        material_volume=material_mass/density_base_material
        print(material_volume)
        
 
        material_index_formula= material_info["material_index_formula"]
        material_index_maxmin= material_info["material_index_maxmin"]
        constraints=material_info["constraints"]

        sub_product_module,key=Create_Modules_Subproduct_2030(matches[0], main_address, geolocator, database_2030, database_bis_2030, material_id,material_volume,process_list,df_Material,df_Process_Caracteristic,df_Supplier,constraints,material_index_formula, material_index_maxmin,increase_percentage_price)
        print('-------------------------sub_product_module-------------------------------')
        print(sub_product_module)
        raw_data.extend(sub_product_module)
        material.append(key)
        print(material)
        print(key)
        print(sub_product_module)
        activity = bd.get_activity(('database_bis_2030', key))
        print(' activity')
        print(activity)
        
        material_id_list.append(material_id)
        quantity.append(material_info["quantity"])
    print('MATERIAL KEY:')
    print(material)
    print('MATERIAL ID:')
    print(material_id_list)
    
    #ADD THE MODULE OF THE FINAL PRODUCT, need to first create the final product activity by taking the subproducts

    
    #ADD THE MODULE OF THE FINAL PRODUCT, need to first create the final product activity by taking materials and process and after the material taken will be cut when the module will be created
    # Create a new activity for the final product
    cut=[]
    chain=[]
    product_id=product["specificity"]["id"]
    existing_process = next((act for act in database_bis_2030 if act['code'] == product_id), None)
    if existing_process:
        # Overwrite or update the existing process
        for exchange in list(existing_process.exchanges()):
            exchange.delete()  # Remove the exchange
        print(f"Deleted all existing exchanges for process: {product_id}")
        # Overwrite or update the existing process
        existing_process['name'] = product["specificity"]["id"]
        existing_process['unit'] = 'unit'
        existing_process.save()
        print(f"Updated existing process: {existing_process}")
        Final_module = existing_process  # Use the existing process object
    else:
        # Create a new activity
        Final_module = database_bis_2030.new_node(code= product["specificity"]["id"], name= product["specificity"]["id"], unit="unit")
        print(Final_module)
        Final_module.save()
    for i, j , k in zip(material, quantity, material_id_list):
        print('KEY')
        print(database_bis_2030.get(i))
        Final_module.new_exchange(input=bd.get_activity(('database_bis_2030', i)), amount=j, type='technosphere', unit='unit').save()
        cut.append((('database_bis_2030',i),Final_module.key,k,j))
        
        chain.append(('database_bis_2030',i))
    
    Final_module.new_exchange(input=Final_module, amount=1, type='production', unit='kg').save()
    # Create module
    raw_data.append({
        'name':'Assembly',
        'outputs': [
           ( Final_module.key, product["specificity"]["id"], 1.0)
        ],
        'chain': [( Final_module.key)] + chain,
        'cuts': cut,
        'output_based_scaling': True
    })
    print('THe final product has been added')
    return(raw_data)


def Create_all_Modules_2050(database_2050,database_bis_2050,product,raw_data,df_Material,df_Process_Caracteristic,df_Supplier,increase_percentage_price):
    
    #this list have double with first the activity key and second the quantity
    material=[]
    material_id_list=[]
    quantity=[]
    

    process_names = df_Process_LCI['name'].tolist()
    
    #ADD THE MODULE OF THE SUBPRODUCT
    print('NEW MATERIAL')
    for material_id, material_info in product["materials"].items():
        print(material_id)
        process_list=[]
        #for each material of the BOM find similar
        material_name = material_info["name"]
        print(material_name)
        material_names = df_Material['Material'].tolist()
        # Get the closest matches using difflib
        matches = difflib.get_close_matches(material_name, material_names, n=1, cutoff=0.5)
        print('match')
        print(matches)
        #find the wo that are associated with material_id
        for routing_id in material_info["routing"]:
            matches_process = difflib.get_close_matches(product["routing"][routing_id]["name"], process_names, n=1, cutoff=0.5)
            process_list.append((matches_process[0], product["routing"][routing_id]["time"],product["routing"][routing_id]["shape"],product["routing"][routing_id]["tolerance"],product["routing"][routing_id]["thickness"],product["routing"][routing_id]["roughness"],material_info["quantity"],routing_id))
               

        # for routing_id, routing_info in product["routing"].items():
        #     if routing_info["materials"][0] == material_id:
        #         matches_process = difflib.get_close_matches(routing_info["name"], process_names, n=1, cutoff=0.5)
        #         process_list.append((matches_process, routing_info["time"],routing_info["shape"],routing_info["tolerance"],routing_info["thickness"],routing_info["roughness"],routing_info["mass"]))
        
        #use a fucntion that takes in input the processes used and the matches material and return the modules of all material and processes associated
        print('STRAT SUB PRODUCT for')
        print(material_id)
        print('process_list')
        print(process_list)
        print('matches[0]')
        print(matches[0])
        
        density_base_material = df_Material.loc[df_Material['Material'] == matches[0], 'Density'].values[0]
        density_base_material = float(density_base_material)
        material_mass=material_info["quantity"]
        print('MATERIALLL VOLUMEEEE')
        material_volume=material_mass/density_base_material
        print(material_volume)
        
        material_index_formula= material_info["material_index_formula"]
        material_index_maxmin= material_info["material_index_maxmin"]
        constraints=material_info["constraints"]
        
        sub_product_module,key=Create_Modules_Subproduct_2050(matches[0], main_address, geolocator, database_2050, database_bis_2050, material_id,material_volume,process_list,df_Material,df_Process_Caracteristic,df_Supplier,constraints,material_index_formula, material_index_maxmin,increase_percentage_price)
        print('-------------------------sub_product_module-------------------------------')
        print(sub_product_module)
        raw_data.extend(sub_product_module)
        material.append(key)
        print(material)
        print(key)
        print(sub_product_module)
        activity = bd.get_activity(('database_bis_2050', key))
        print(' activity')
        print(activity)
        
        material_id_list.append(material_id)
        quantity.append(material_info["quantity"])
    print('MATERIAL KEY:')
    print(material)
    print('MATERIAL ID:')
    print(material_id_list)
    
    #ADD THE MODULE OF THE FINAL PRODUCT, need to first create the final product activity by taking the subproducts

    
    #ADD THE MODULE OF THE FINAL PRODUCT, need to first create the final product activity by taking materials and process and after the material taken will be cut when the module will be created
    # Create a new activity for the final product
    cut=[]
    chain=[]
    product_id=product["specificity"]["id"]
    existing_process = next((act for act in database_bis_2050 if act['code'] == product_id), None)
    if existing_process:
        # Overwrite or update the existing process
        for exchange in list(existing_process.exchanges()):
            exchange.delete()  # Remove the exchange
        print(f"Deleted all existing exchanges for process: {product_id}")
        # Overwrite or update the existing process
        existing_process['name'] = product["specificity"]["id"]
        existing_process['unit'] = 'unit'
        existing_process.save()
        print(f"Updated existing process: {existing_process}")
        Final_module = existing_process  # Use the existing process object
    else:
        # Create a new activity
        Final_module = database_bis_2050.new_node(code= product["specificity"]["id"], name= product["specificity"]["id"], unit="unit")
        print(Final_module)
        Final_module.save()
    for i, j , k in zip(material, quantity, material_id_list):
        print('KEY')
        print(database_bis_2050.get(i))
        Final_module.new_exchange(input=bd.get_activity(('database_bis_2050', i)), amount=j, type='technosphere', unit='unit').save()
        cut.append((('database_bis_2050',i),Final_module.key,k,j))
        
        chain.append(('database_bis_2050',i))
    
    Final_module.new_exchange(input=Final_module, amount=1, type='production', unit='kg').save()
    # Create module
    raw_data.append({
        'name':'Assembly',
        'outputs': [
           ( Final_module.key, product["specificity"]["id"], 1.0)
        ],
        'chain': [( Final_module.key)] + chain,
        'cuts': cut,
        'output_based_scaling': True
    })
    print('THe final product has been added')
    return(raw_data)


raw_data=[]
rd,quality_df,energy= Create_all_Modules(database,database_bis,product, raw_data,df_Material,df_Process_Caracteristic,df_Supplier,increase_percentage_price)
print(rd)
print(quality_df)
lmp.load_from_data(rd, append=False)


raw_data_2030=[]
rd_2030= Create_all_Modules_2030(database_2030,database_bis_2030,product, raw_data_2030,df_Material,df_Process_Caracteristic,df_Supplier,increase_percentage_price)
print(rd_2030)
lmp_2030.load_from_data(rd_2030, append=False)


raw_data_2050=[]
rd_2050= Create_all_Modules_2050(database_2050,database_bis_2050,product, raw_data,df_Material,df_Process_Caracteristic,df_Supplier,increase_percentage_price)
print(rd_2050)
lmp_2050.load_from_data(rd_2050, append=False)