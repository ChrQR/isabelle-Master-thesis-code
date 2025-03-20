import bw2data as bd
import pandas as pd


import difflib
import re
from geopy.geocoders import Nominatim
from geopy.distance import geodesic
import uuid

bd.projects.set_current("default")
print(bd.projects.set_current)
current_project = bd.projects.current
print(f"Current project: {current_project}")
list(bd.databases)

# Replace 'file.xlsx' with your actual file path
file_path = "data/Material.xlsx"
file_path2 = "data/Supplier.xlsx"
file_path3 = "data/Process_Caracteristic.xlsx"
file_path4 = "data/LCI_Processes.xlsx"

df_Process_LCI = pd.read_excel(file_path4)
df_Material = pd.read_excel(file_path)
df_Supplier = pd.read_excel(file_path2)
main_address = "Hedemølle Erhvervsvej 23. DK-8850 Bjerringbro"


database = bd.Database("ecoinvent-3.10-cutoff")
database_2030 = bd.Database("ei_cutoff_3.10_remind_SSP2-Base_2030 2025-02-16")
database_2050 = bd.Database("ei_cutoff_3.10_remind_SSP2-Base_2050 2025-02-16")


database_bis = bd.Database("database_bis")
database_bis_2030 = bd.Database("database_bis_2030")
database_bis_2050 = bd.Database("database_bis_2050")
df_Process_Caracteristic = pd.read_excel(file_path3)

main_address = "Hedemølle Erhvervsvej 23. DK-8850 Bjerringbro"
database = bd.Database("ecoinvent-3.10-cutoff")

raw_data = []


print(df_Process_Caracteristic)


def find_similar_materials(
    material_name,
    df_Material,
    df_Process_Caracteristic,
    df_Supplier,
    constraints,
    material_index_formula,
    material_index_maxmin,
    price_increase_percentage,
):
    # Ensure dependencies are loaded

    # Copy dataset
    full_data = df_Material.copy()
    print(len(full_data))
    print(full_data)

    # Compute Material Index for ALL materials first (before filtering)
    try:
        full_data["Material Index"] = full_data.eval(material_index_formula)
    except Exception as e:
        print("❌ Error in formula evaluation:", e)
        return None

    # Get original material row (before filtering)
    original_row = full_data[full_data["Material"] == material_name]

    # If original material is found, store its Material Index and price
    if not original_row.empty:
        original_material_index = original_row["Material Index"].values[0]
        original_price = original_row["Price per kilo"].values[0]
    else:
        original_material_index = None  # Handle the case where material is missing
        original_price = None

    print(original_price)

    # Apply constraints to filter materials
    filtered = full_data.copy()
    print(len(filtered))
    if original_price is not None:
        max_price = (1 + price_increase_percentage) * original_price
        print(max_price)
        filtered = filtered[filtered["Price per kilo"] <= max_price]
        print("after price")
    print(len(filtered))
    for prop, (operator, value) in constraints.items():
        print(f"Processing constraint: {prop} {operator} {value}")

        try:
            # Check if prop is a formula (contains math operators) or a direct column name
            if any(op in prop for op in ["*", "/", "+", "-"]):
                computed_values = full_data.eval(prop)  # Compute formula dynamically
            else:
                computed_values = full_data[prop]  # Direct column reference

            # Apply constraints
            if operator == "<=":
                filtered = filtered[computed_values <= value]
                print(len(filtered))
            elif operator == ">=":
                filtered = filtered[computed_values >= value]
                print(len(filtered))
            elif operator == "<":
                filtered = filtered[computed_values < value]
                print(len(filtered))
            elif operator == ">":
                filtered = filtered[computed_values > value]
                print(len(filtered))
            elif operator == "==":
                filtered = filtered[computed_values == value]
                print(len(filtered))

        except Exception as e:
            print(f"❌ Error processing constraint '{prop}': {e}")

    print(filtered)

    # Determine sorting order
    ascending = material_index_maxmin.lower() == "min"

    # Sort materials based on Material Index
    filtered["Material Index"] = filtered["Material Index"].round(10)
    print("filtered with material index")
    print(len(filtered))
    print(filtered)
    filtered = filtered.sort_values(by="Material Index", ascending=ascending, kind="stable")
    print("SORTED BY VALUE")
    print(len(filtered))
    print(filtered)

    # Select the top 4 materials
    top_materials = filtered.head(4)

    # If original material was removed by constraints, re-add it with its index
    if original_row.empty or material_name not in filtered["Material"].values:
        if original_material_index is not None:
            missing_original = pd.DataFrame(
                [
                    {
                        "Material": material_name,
                        "Material Index": original_material_index,
                        "Price per kilo": original_price,
                    }
                ]
            )
            final_materials = pd.concat([missing_original, top_materials], ignore_index=True)
        else:
            final_materials = top_materials  # If original material is missing entirely, return top 4 only
    else:
        # If the original material is in the filtered list, just insert it at the top
        filtered = filtered[filtered["Material"] != material_name]  # Remove it before re-adding
        top_materials = filtered.head(4)
        final_materials = pd.concat([original_row, top_materials], ignore_index=True)

    # Compute Grade (normalized so best material = 1)
    if ascending:
        min_material_index = final_materials["Material Index"].min()
        final_materials["Grade"] = min_material_index / final_materials["Material Index"]
    else:
        max_material_index = final_materials["Material Index"].max()
        final_materials["Grade"] = final_materials["Material Index"] / max_material_index

    # Select relevant columns
    final_materials = final_materials[["Material", "Material Index", "Grade", "Price per kilo"]]
    similar_material_names = final_materials["Material"].values.flatten()

    return final_materials, similar_material_names


geolocator = Nominatim(user_agent="distance_calculator", timeout=10)


def get_coordinates(address, geolocator):
    # Initialize Nominatim geocoder

    try:
        location = geolocator.geocode(address)
        if location:
            return (location.latitude, location.longitude)
        else:
            print(f"Address not found: {address}")
            return None
    except Exception as e:
        print(f"Error fetching coordinates for {address}: {e}")
        return None


def calculate_distance(address1, address2, geolocator):
    coords_1 = get_coordinates(address1, geolocator)
    coords_2 = get_coordinates(address2, geolocator)
    if coords_1 and coords_2:
        return geodesic(coords_1, coords_2).kilometers
    else:
        return None


def shorter_distance(material_name, main_address, geolocator, df_Material, f_Process_Caracteristic, df_Supplier):
    d = 0
    pe_suppliers_row = df_Material.loc[df_Material["Material"] == material_name, "Suppliers"]
    if not pe_suppliers_row.empty:
        pe_suppliers_list = pe_suppliers_row.iloc[0].split(", ")
        d = 40000
        s = "unknown"
        for supplier in pe_suppliers_list:
            supplier_address = df_Supplier.loc[df_Supplier["Name"] == supplier, "Address"].values.tolist()[0]
            print(supplier_address)
            new_distance = calculate_distance(main_address, supplier_address, geolocator)
            new_d = int(new_distance)
            print(new_d)

            if new_d < d:
                d = new_d
                s = supplier

    else:
        return "No suppliers found"

    return (d, s)


def add_distance_to_material(
    material_list, main_address, geolocator, df_Material, df_Process_Caracteristic, df_Supplier
):
    df = pd.DataFrame(
        {
            "Material": material_list,
            "Distance": [0] * len(material_list),  # Initialize the second column with zeros
            "Supplier_name": "unknown",
        }
    )
    for i in range(len(df)):
        material = df.iloc[i]["Material"]
        df.loc[i, "Distance"] = shorter_distance(
            material, main_address, geolocator, df_Material, df_Process_Caracteristic, df_Supplier
        )[0]
        df.loc[i, "Supplier_name"] = shorter_distance(
            material, main_address, geolocator, df_Material, df_Process_Caracteristic, df_Supplier
        )[1]  # Set the Distance value for the first row
    return df


# Material_and_Distance is a list with the material name and distance associated related to the supplier
# Material_and_Distance=add_distance_to_material(similar)
# print(Material_and_Distance)


def find_closest_match(search_term, database, incineration, landfill):
    search_term = str(search_term).lower()

    # Load Ecoinvent database
    activities = list(database)  # Convert to a list

    def normalize_name(name):
        """Remove special characters and normalize the name."""
        return re.sub(r"[^\w\s]", "", name).lower()

    search_terms = search_term.split()

    def filter_valid_activities():
        """Filter activities that have unit = 'kilogram'."""
        valid_units = {"kg", "kilogram"}
        filtered = []
        for a in activities:
            try:
                activity_data = a.as_dict()  # Convert activity to dictionary
                unit = activity_data.get("unit", "").lower()
                if unit in valid_units:
                    filtered.append(a)
            except AttributeError:
                continue  # Skip if data is incorrect
        return filtered

    valid_activities = filter_valid_activities()
    print(f"Valid activities found: {len(valid_activities)}")

    def get_best_exact_match():
        """Finds activities containing all search words, minimizing extra words."""
        candidates = []
        for a in valid_activities:
            name_words = normalize_name(a.as_dict()["name"]).split()
            if all(word in name_words for word in search_terms):
                candidates.append((a, len(name_words)))

        if candidates:
            return sorted(candidates, key=lambda x: x[1])[0][0]
        return None

    def get_best_partial_match():
        """Finds activities containing only the first search word, minimizing extra words."""
        first_word = search_terms[0]
        candidates = []
        for a in valid_activities:
            name_words = normalize_name(a.as_dict()["name"]).split()
            if first_word in name_words:
                candidates.append((a, len(name_words)))

        if candidates:
            return sorted(candidates, key=lambda x: x[1])[0][0]
        return None

    def get_best_fuzzy_match():
        """Finds the closest match using fuzzy matching (difflib)."""
        normalized_names = [normalize_name(a.as_dict()["name"]) for a in valid_activities]
        matches = difflib.get_close_matches(search_term, normalized_names, n=5, cutoff=0.5)

        if matches:
            for match in matches:
                for a in valid_activities:
                    if normalize_name(a.as_dict()["name"]) == match:
                        return a
        return None

    # Step 1: Try exact word match
    best_match = get_best_exact_match()

    # Step 2: Try partial match (only first word)
    if not best_match:
        best_match = get_best_partial_match()
        print("not exact match")

    # Step 3: Try fuzzy matching
    if not best_match:
        best_match = get_best_fuzzy_match()
        print("not partial amtch")

    # Step 4: Prioritize locations (RER > RoW > GLO)
    if best_match:
        print("fuzzy match")
        rer_activities = [
            a
            for a in valid_activities
            if a.as_dict()["name"] == best_match.as_dict()["name"] and a.as_dict().get("location", "") == "RER"
        ]
        if rer_activities:
            best_match = rer_activities[0]

        row_activities = [
            a
            for a in valid_activities
            if a.as_dict()["name"] == best_match.as_dict()["name"] and a.as_dict().get("location", "") == "RoW"
        ]
        if row_activities:
            best_match = row_activities[0]

        glo_activities = [
            a
            for a in valid_activities
            if a.as_dict()["name"] == best_match.as_dict()["name"] and a.as_dict().get("location", "") == "GLO"
        ]
        if glo_activities:
            best_match = glo_activities[0]

    # Step 5: Apply incineration/landfill constraint
    if incineration:
        print("incineration true")
        keyword = "incineration"
        best_match_name = normalize_name(best_match.as_dict()["name"]) if best_match else ""
        if keyword not in best_match_name:
            best_match = database.get("e1c1582822e657797c8bf7570366ffe5")
    if landfill:
        print("landfill true")
        keyword = "landfill"
        best_match_name = normalize_name(best_match.as_dict()["name"]) if best_match else ""
        if keyword not in best_match_name:
            best_match = database.get("1f8a601136a5d5894ffdbd38395693b5")

    return best_match


def give_EF_Material(database, similar, df_Material, df_Process_Caracteristic, df_Supplier):
    df_Material_And_EF = pd.DataFrame(
        {
            "Material": similar,
            "Activity_key": ["0"] * len(similar),  # Initialize the second column with zeros
        }
    )
    for i in similar:
        search_term = f"{i} production"
        search_term = str(search_term)
        matching_activity = find_closest_match(i, database, False, False)
        if matching_activity:
            df_Material_And_EF.loc[df_Material_And_EF["Material"] == i, "Activity_key"] = matching_activity["code"]
        else:
            print("no EF found")
            df_Material_And_EF.loc[df_Material_And_EF["Material"] == i, "Activity_key"] = None

    # Material_and_Distance is a list with the material name and emission factor key associated related to the supplier
    return df_Material_And_EF


def find_similar_time(new_process, process, df_Material, df_Process_Caracteristic, df_Supplier, mat1, mat2):
    time_1 = process[1]
    SEC1 = df_Material.loc[df_Material["Material"] == str(mat1), "SEC"].iloc[0]
    SEC2 = df_Material.loc[df_Material["Material"] == str(mat2), "SEC"].iloc[0]
    efficiency_1 = df_Process_Caracteristic.loc[
        df_Process_Caracteristic["Name"] == process[0], "Average efficiency"
    ].iloc[0]
    efficiency_2 = df_Process_Caracteristic.loc[
        df_Process_Caracteristic["Name"] == new_process, "Average efficiency"
    ].iloc[0]
    power_1 = df_Process_Caracteristic.loc[df_Process_Caracteristic["Name"] == process[0], "Power"].iloc[0]
    power_2 = df_Process_Caracteristic.loc[df_Process_Caracteristic["Name"] == new_process, "Power"].iloc[0]
    time_2 = time_1 * efficiency_1 * power_1 * SEC2 / (power_2 * efficiency_2 * SEC1)

    return time_2


def find_similar_process(process_list, i, df_Material, df_Process_Caracteristic, df_Supplier):
    # Process_list is the list for one process like
    # (matches_process, routing_info["time"], routing_info["shape"], routing_info["tolerance"], routing_info["thickness"], routing_info["roughness"], routing_info["mass"],routing_info["mass"])
    # i is the material name that will replace the basis material

    tolerance_filter = process_list[3]
    weight_filter = process_list[6]
    thickness_filter = process_list[4]
    roughness_filter = process_list[5]
    process_type = df_Process_Caracteristic.loc[
        df_Process_Caracteristic["Name"] == process_list[0], "Process type"
    ].iloc[0]

    # Get the material category
    material_filter = df_Material.loc[df_Material["Material"] == i, "Category"].iloc[0]

    # Shape filter (assuming 'process_list[2]' contains the shape information)
    shape_filter = process_list[2]

    # Debugging prints to inspect data types
    print("Material Filter:", material_filter)
    print("Shape Filter:", shape_filter)
    print("Tolerance filter:", tolerance_filter)
    print("Weight filter:", weight_filter)
    print("Thickness filter:", thickness_filter)
    print("Roughness filter:", roughness_filter)

    # Check the data types of the columns in df_Process_Caracteristic
    print("\nData types of df_Process_Caracteristic:")

    # Convert relevant columns to numeric, forcing errors to NaN (useful for cleaning data)
    df_Process_Caracteristic["Tolerance min"] = pd.to_numeric(
        df_Process_Caracteristic["Tolerance min"], errors="coerce"
    )
    df_Process_Caracteristic["Mass min"] = pd.to_numeric(df_Process_Caracteristic["Mass min"], errors="coerce")
    df_Process_Caracteristic["Mass max"] = pd.to_numeric(df_Process_Caracteristic["Mass max"], errors="coerce")
    df_Process_Caracteristic["Section thickness min"] = pd.to_numeric(
        df_Process_Caracteristic["Section thickness min"], errors="coerce"
    )
    df_Process_Caracteristic["Section thickness max"] = pd.to_numeric(
        df_Process_Caracteristic["Section thickness max"], errors="coerce"
    )
    df_Process_Caracteristic["Roughness max"] = pd.to_numeric(
        df_Process_Caracteristic["Roughness max"], errors="coerce"
    )
    df_Process_Caracteristic["Roughness min"] = pd.to_numeric(
        df_Process_Caracteristic["Roughness min"], errors="coerce"
    )

    # Debugging print to confirm numeric conversion
    print("\nData types after conversion to numeric:")

    # Dynamically build the filtering conditions
    conditions = []
    if tolerance_filter is not None:
        conditions.append(df_Process_Caracteristic["Tolerance min"] <= tolerance_filter)
    if weight_filter is not None:
        conditions.append(df_Process_Caracteristic["Mass min"] <= weight_filter)
        conditions.append(df_Process_Caracteristic["Mass max"] >= weight_filter)
    if thickness_filter is not None:
        conditions.append(df_Process_Caracteristic["Section thickness min"] <= thickness_filter)
        conditions.append(df_Process_Caracteristic["Section thickness max"] >= thickness_filter)
    if roughness_filter is not None:
        conditions.append(df_Process_Caracteristic["Roughness max"] >= roughness_filter)
        conditions.append(df_Process_Caracteristic["Roughness min"] <= roughness_filter)
    if material_filter:
        conditions.append(df_Process_Caracteristic[material_filter] == True)
    if shape_filter:
        conditions.append(df_Process_Caracteristic[shape_filter] == True)
    if process_type:
        conditions.append(df_Process_Caracteristic["Process type"] == process_type)

    # Combine conditions
    if conditions:
        combined_condition = conditions[0]
        for condition in conditions[1:]:
            combined_condition &= condition
        filtered_processes = df_Process_Caracteristic[combined_condition]["Name"]
    else:
        filtered_processes = pd.Series([])
    # # Filter the DataFrame
    # filtered_processes = df_Process_Caracteristic[
    #     (df_Process_Caracteristic['Tolerance min'] <= tolerance_filter)&
    #     (df_Process_Caracteristic['Mass min'] <= weight_filter) &
    #     (df_Process_Caracteristic['Mass max'] >= weight_filter) &
    #     (df_Process_Caracteristic['Section thickness min'] <= thickness_filter) &
    #     (df_Process_Caracteristic['Section thickness max'] >= thickness_filter) &
    #     (df_Process_Caracteristic['Roughness max'] >= roughness_filter) &
    #     (df_Process_Caracteristic['Roughness min'] <= roughness_filter) &
    #     (df_Process_Caracteristic[ material_filter] == True) &
    #     (df_Process_Caracteristic['Type'] == process_type) &# Correctly filter by material category
    #     (df_Process_Caracteristic[shape_filter] == True)  # Correctly filter by shape
    #     ]['Name']

    # Debugging print to check filtered_processes
    print("\nFiltered Processes:")
    print(filtered_processes)

    # Get the first matching process
    process_match = filtered_processes.tolist()
    print("process_match")
    print(process_match)

    return process_match[:5]


def give_Process_Material(similar, process_list, df_Material, df_Process_Caracteristic, df_Supplier, base_mat):
    delete_similar = []
    """
    Create a DataFrame mapping similar materials to their associated processes and times.
    
    :param similar: List of similar materials.
    :param process_list: List of tuples (process, time) representing the processes and their times.
    :return: DataFrame with materials, processes, and times.
    """
    print("INSIDE PROCESS MATERIAL GIVE FUNCtion")
    print("similar")
    print(similar)
    # Initialize the DataFrame with processes and times for the base material
    # Initialize the DataFrame with processes and times for the base material
    df_Material_And_Process = pd.DataFrame(
        {
            "Material": similar,
            "Process": [[[[process[0]]] for process in process_list]] + [[]] * (len(similar) - 1),
            "Time": [[[[process[1]]] for process in process_list]] + [[]] * (len(similar) - 1),
        }
    )
    print("MAterial and porcess")
    print(df_Material_And_Process.to_string())

    # Iterate over similar materials (starting from the second one)
    for i in range(1, len(similar)):  # Skip the base material
        print("in the for")
        print(i)

        material = similar[i]
        new_process_list = []
        new_time_list = []
        temp_time_list = []

        # Iterate over each process and time for the base material
        for process in process_list:
            print("in the for")
            print(process)
            # Call helper functions to compute replacements for this material
            new_processes = find_similar_process(
                process, material, df_Material, df_Process_Caracteristic, df_Supplier
            )  # Replace with your logic
            print("new_process")
            print(new_processes)
            if len(new_processes) == 0:
                delete_similar.append(i)
                break

            if len(new_processes) > 0:
                separated_new_process_list = [[process] for process in new_processes]
                new_process_list.append(separated_new_process_list)
                for new_process in separated_new_process_list:
                    print("new_process")
                    print(new_process)
                    new_time = find_similar_time(
                        new_process[0], process, df_Material, df_Process_Caracteristic, df_Supplier, base_mat, material
                    )
                    temp_time_list.append([new_time])
                new_time_list.append(temp_time_list)
                # create process module

            # Update the DataFrame with the new processes and times for this material
            df_Material_And_Process.at[i, "Process"] = new_process_list
            df_Material_And_Process.at[i, "Time"] = new_time_list

    return df_Material_And_Process


def Create_Modules_Subproduct(
    material_name,
    main_address,
    geolocator,
    database,
    database_bis,
    material_id,
    material_volume,
    process_list,
    df_Material,
    df_Process_Caracteristic,
    df_Supplier,
    constraints,
    material_index_formula,
    material_index_maxmin,
    increase_percentage_price,
):
    material_volume = float(material_volume)
    print(material_volume)
    increase_percentage_price
    # The scaling is for the material selection
    scaling = ["None", 10, 10, "None", "None", "None"]

    similar_material_table, similar = find_similar_materials(
        material_name,
        df_Material,
        df_Process_Caracteristic,
        df_Supplier,
        constraints,
        material_index_formula,
        material_index_maxmin,
        increase_percentage_price,
    )

    print("similar_material_table")
    print(similar_material_table)
    df_Material_and_Distance = add_distance_to_material(
        similar, main_address, geolocator, df_Material, df_Process_Caracteristic, df_Supplier
    )
    print(df_Material_and_Distance)

    df_Material_And_EF = give_EF_Material(database, similar, df_Material, df_Process_Caracteristic, df_Supplier)
    print(df_Material_And_EF)

    df_Material_And_Process = give_Process_Material(
        similar, process_list, df_Material, df_Process_Caracteristic, df_Supplier, material_name
    )
    print(df_Material_And_Process)

    # Initialize a list to store the created activities
    created_modules = []
    material_code = []
    names = []
    quality_index = 0
    columns = ["Sub_Product_name", "Quality", "Price"]
    columns_energy = ["Process_name", "kWh"]
    quality_df = pd.DataFrame(columns=columns)
    energy_process = pd.DataFrame(columns=columns_energy)
    material_quality_index = 0
    material_price_index = 0
    transport_waste = database.get("462d96778c87f1897a413fb42bccbd72")

    for i in similar:
        search_term_incineration = f"{i} incineration"
        search_term_landfill = f"{i} landfill"
        incineration_rate = df_Material.loc[df_Material["Material"] == i, "IncinerationRate"].values
        density_similar = df_Material.loc[df_Material["Material"] == i, "Density"].values
        density_similar = float(density_similar)
        print("Density")
        print(density_similar)

        landfill_rate = df_Material.loc[df_Material["Material"] == i, "LandfillRate"].values
        # Extract values safely
        incineration_rate = df_Material.loc[df_Material["Material"] == i, "IncinerationRate"].values
        landfill_rate = df_Material.loc[df_Material["Material"] == i, "LandfillRate"].values

        # Ensure values are numeric and handle missing cases
        if len(incineration_rate) > 0:
            incineration_rate = float(incineration_rate[0])  # Convert to float
        else:
            incineration_rate = 0.0  # Default value (adjust if needed)

        if len(landfill_rate) > 0:
            landfill_rate = float(landfill_rate[0])  # Convert to float
        else:
            landfill_rate = 0.0  # Default value (adjust if needed)

        incineration = find_closest_match(search_term_incineration, database, True, False)
        incineration_key = incineration["code"]
        landfill = find_closest_match(search_term_landfill, database, False, True)
        landfill_key = landfill["code"]
        recycling_rate = 0.05

        material_quality_index = similar_material_table.loc[similar_material_table["Material"] == i, "Grade"].values[0]
        material_price_index = similar_material_table.loc[
            similar_material_table["Material"] == i, "Price per kilo"
        ].values[0]

        # Get the material's key and associated activity
        material_key = df_Material_And_EF.loc[df_Material_And_EF["Material"] == i, "Activity_key"].values[0]
        print(material_key)
        print("material_key")
        material = database.get(material_key)

        transport = database_bis.get("Transport_European")
        transport_amount = df_Material_and_Distance.loc[df_Material_and_Distance["Material"] == i, "Distance"].values[0]

        similar_process_list = df_Material_And_Process.loc[df_Material_And_Process["Material"] == i, "Process"].values[
            0
        ]
        print("similar_process_list")
        print(similar_process_list)
        time_list = df_Material_And_Process.loc[df_Material_And_Process["Material"] == i, "Time"].values[0]

        # for each process create an activity and a module
        for process, time, base_process in zip(similar_process_list, time_list, process_list):
            print("similar_process_list")
            print(similar_process_list)
            for process_alternative, time_alternative in zip(process, time):
                print("-------------------process[0]---------------")
                print(str(process_alternative[0]))
                print("time_alternative")
                print(time_alternative)
                name = process_alternative[0] + "_for_" + i + "_process_id_" + str(base_process[7])
                names.append(name)
                output = "Process_for_" + i + "_process_id_" + str(base_process[7])
                existing_process = next((act for act in database_bis if act["name"] == name), None)
                if existing_process:
                    for exchange in list(existing_process.exchanges()):
                        exchange.delete()  # Remove the exchange
                    existing_process["name"] = name
                    existing_process["code"] = name
                    module_process = existing_process  # Use the existing process object
                    module_process.save()
                if not existing_process:
                    module_process = database_bis.new_node(code=name, name=name, unit="unit")
                    module_process.save()
                process_alternative_act = database_bis.get(process_alternative[0])
                print("process alternative act found!:")
                print(process_alternative_act)
                material_mass = density_similar * material_volume
                print(density_similar)
                print(material_volume)
                print(material_mass)
                amount = time_alternative[0] / material_mass
                print(amount)
                module_process.new_exchange(
                    input=process_alternative_act, amount=amount, type="technosphere", unit="min"
                ).save()
                module_process.new_exchange(input=module_process, amount=1, type="production", unit="unit").save()
                power = df_Process_Caracteristic.loc[
                    df_Process_Caracteristic["Name"] == str(process_alternative[0]), "Power"
                ].values[0]
                energy = time_alternative[0] * power
                new_row = {"Process_name": name, "kWh": material_quality_index}
                energy_process.loc[len(energy_process)] = new_row

                my_object = {
                    "name": name,
                    "outputs": [(("database_bis", name), output, 1)],
                    "chain": [("database_bis", name)],
                    "cuts": [],
                    "output_based_scaling": True,
                }

                created_modules.append(my_object)

        print("names")
        print(names)
        # Generate a unique code for each module
        unique_code = str(uuid.uuid4())

        chain = []
        cut = []
        supplier_name = df_Material_and_Distance.loc[df_Material_and_Distance["Material"] == i, "Supplier_name"].values
        if supplier_name.size > 0:
            supplier_name = supplier_name[0]
        else:
            supplier_name = "Unknown Supplier"
        existing_process = next(
            (act for act in database_bis if act["name"] == f"{i} from {supplier_name} fully processed"), None
        )

        if existing_process:
            for exchange in list(existing_process.exchanges()):
                exchange.delete()  # Remove the exchange
            existing_process["name"] = f"{i} from {supplier_name} fully processed"

            existing_process["code"] = unique_code
            module_material = existing_process  # Use the existing process object
            module_material.save()
        else:
            module_material = database_bis.new_node(
                code=unique_code,
                name=f"{i} from {supplier_name} fully processed",  # Fixed the formatting
                unit="kg",
            )

            module_material.save()
        print("process_list")
        print(process_list)
        for processes_i, process in zip(similar_process_list, process_list):
            print("processes_i[0]")
            print(processes_i[0])

            output = "Process_for_" + i + "_process_id_" + str(process[7])

            code = processes_i[0][0] + "_for_" + i + "_process_id_" + str(process[7])
            print("code")
            print(code)
            chain.append(("database_bis", code))
            cut.append((("database_bis", code), unique_code, output, 1))
            process = database_bis.get(code)
            module_material.new_exchange(input=process, amount=1, type="technosphere", unit="unit").save()

        # Add exchanges to the module
        module_material.new_exchange(input=material, amount=1, type="technosphere", unit="unit").save()
        module_material.new_exchange(input=transport, amount=transport_amount, type="technosphere", unit="unit").save()
        module_material.new_exchange(input=module_material, amount=1, type="production", unit="unit").save()
        landfill_amount = (1 - recycling_rate) * landfill_rate
        incineration_amount = (1 - recycling_rate) * incineration_rate

        module_material.new_exchange(
            input=incineration, amount=incineration_amount, type="technosphere", unit="unit"
        ).save()
        module_material.new_exchange(input=landfill, amount=landfill_amount, type="technosphere", unit="unit").save()
        module_material.new_exchange(
            input=transport_waste, amount=1 / 1000 * 70, type="technosphere", unit="unit"
        ).save()

        module_material.save()
        material_price = material_price_index * material_mass
        new_row = {
            "Sub_Product_name": module_material["name"],
            "Quality": material_quality_index,
            "Price": material_price,
        }
        quality_df.loc[len(quality_df)] = new_row
        print("CUT")
        print(cut)
        print("Module saved")
        my_object = {
            "name": module_material["name"],
            "outputs": [(("database_bis", unique_code), material_id, 1)],
            "chain": [("database_bis", unique_code)] + chain,
            "cuts": cut,
            "output_based_scaling": True,
        }
        print("object created")

        print(my_object)

        created_modules.append(my_object)
        material_code.append(unique_code)
        print("MATERIAL CODE ADDED:")

    print(energy_process)
    # Return the list of created modules

    return (created_modules, material_code[0], quality_df, energy_process)


constraints = {
    "Yield_Strength": (">", 2.16),  # g/cm³
    "Young_Modulus": (">=", 6.91),  # W/(m·K)
    "Density": ("<", 3000),  # W/(m·K)
    "toughness": (">=", 1),  # W/(m·K)
}
material_index_formula = "Density"
material_index_maxmin = "Max"
increase_percentage_price = 0.2

module = Create_Modules_Subproduct(
    "Styrene Acrylonitrile",
    main_address,
    geolocator,
    database,
    database_bis,
    "M001",
    2,
    [("Press brake", 2.5, "Flat", None, None, None, 5, "M001")],
    df_Material,
    df_Process_Caracteristic,
    df_Supplier,
    constraints,
    material_index_formula,
    material_index_maxmin,
    increase_percentage_price,
)
print(type(module))


def Create_Modules_Subproduct_2030(
    material_name,
    main_address,
    geolocator,
    database_2030,
    database_bis_2030,
    material_id,
    material_volume,
    process_list,
    df_Material,
    df_Process_Caracteristic,
    df_Supplier,
    constraints,
    material_index_formula,
    material_index_maxmin,
    increase_percentage_price,
):
    material_volume = float(material_volume)
    print(material_volume)
    # The scaling is for the material selection
    scaling = ["None", 10, 10, "None", "None", "None"]
    similar = find_similar_materials(
        material_name,
        df_Material,
        df_Process_Caracteristic,
        df_Supplier,
        constraints,
        material_index_formula,
        material_index_maxmin,
        increase_percentage_price,
    )[1]
    transport_waste = database_2030.get("462d96778c87f1897a413fb42bccbd72")
    df_Material_and_Distance = add_distance_to_material(
        similar, main_address, geolocator, df_Material, df_Process_Caracteristic, df_Supplier
    )
    print(df_Material_and_Distance)

    df_Material_And_EF = give_EF_Material(database_2030, similar, df_Material, df_Process_Caracteristic, df_Supplier)

    df_Material_And_Process = give_Process_Material(
        similar, process_list, df_Material, df_Process_Caracteristic, df_Supplier, material_name
    )

    # Initialize a list to store the created activities
    created_modules = []
    material_code = []
    names = []

    for i in similar:
        density_similar = df_Material.loc[df_Material["Material"] == i, "Density"].values
        density_similar = float(density_similar)
        search_term_incineration = f"{i} incineration"
        search_term_landfill = f"{i} landfill"
        incineration_rate = df_Material.loc[df_Material["Material"] == i, "IncinerationRate_2030"].values

        landfill_rate = df_Material.loc[df_Material["Material"] == i, "LandfillRate_2030"].values

        # Ensure values are numeric and handle missing cases
        if len(incineration_rate) > 0:
            incineration_rate = float(incineration_rate[0])  # Convert to float
        else:
            incineration_rate = 0.0  # Default value (adjust if needed)

        if len(landfill_rate) > 0:
            landfill_rate = float(landfill_rate[0])  # Convert to float
        else:
            landfill_rate = 0.0  # Default value (adjust if needed)

        incineration = find_closest_match(search_term_incineration, database_2030, True, False)
        incineration_key = incineration["code"]
        landfill = find_closest_match(search_term_landfill, database_2030, False, True)
        landfill_key = landfill["code"]
        recycling_rate = 0.05

        # Get the material's key and associated activity
        material_key = df_Material_And_EF.loc[df_Material_And_EF["Material"] == i, "Activity_key"].values[0]
        material = database_2030.get(material_key)

        transport = database_bis_2030.get("Transport_European")
        transport_amount = df_Material_and_Distance.loc[df_Material_and_Distance["Material"] == i, "Distance"].values[0]

        similar_process_list = df_Material_And_Process.loc[df_Material_And_Process["Material"] == i, "Process"].values[
            0
        ]
        print("similar_process_list")
        print(similar_process_list)
        time_list = df_Material_And_Process.loc[df_Material_And_Process["Material"] == i, "Time"].values[0]

        # for each process create an activity and a module
        for process, time, base_process in zip(similar_process_list, time_list, process_list):
            print("similar_process_list")
            print(similar_process_list)
            for process_alternative, time_alternative in zip(process, time):
                print("process_alternative")
                print(process_alternative)
                print("time_alternative")
                print(time_alternative)
                name = process_alternative[0] + "_for_" + i + "_process_id_" + str(base_process[7])
                names.append(name)
                output = "Process_for_" + i + "_process_id_" + str(base_process[7])
                existing_process = next((act for act in database_bis_2030 if act["name"] == name), None)
                if existing_process:
                    for exchange in list(existing_process.exchanges()):
                        exchange.delete()  # Remove the exchange
                    existing_process["name"] = name
                    existing_process["code"] = name
                    module_process = existing_process  # Use the existing process object
                    module_process.save()
                if not existing_process:
                    module_process = database_bis_2030.new_node(code=name, name=name, unit="unit")
                    module_process.save()
                process_alternative_act = database_bis_2030.get(process_alternative[0])
                print("process alternative act found!:")
                print(process_alternative_act)
                material_mass = density_similar * material_volume

                amount = time_alternative[0] / material_mass
                module_process.new_exchange(
                    input=process_alternative_act, amount=amount, type="technosphere", unit="min"
                ).save()
                module_process.new_exchange(input=module_process, amount=1, type="production", unit="unit").save()

                my_object = {
                    "name": name,
                    "outputs": [(("database_bis_2030", name), output, 1)],
                    "chain": [("database_bis_2030", name)],
                    "cuts": [],
                    "output_based_scaling": True,
                }

                created_modules.append(my_object)

        print("names")
        print(names)
        # Generate a unique code for each module
        unique_code = str(uuid.uuid4())

        chain = []
        cut = []
        supplier_name = df_Material_and_Distance.loc[df_Material_and_Distance["Material"] == i, "Supplier_name"].values
        if supplier_name.size > 0:
            supplier_name = supplier_name[0]
        else:
            supplier_name = "Unknown Supplier"
        existing_process = next(
            (act for act in database_bis_2030 if act["name"] == f"{i} from {supplier_name} fully processed"), None
        )

        if existing_process:
            for exchange in list(existing_process.exchanges()):
                exchange.delete()  # Remove the exchange
            existing_process["name"] = f"{i} from {supplier_name} fully processed"

            existing_process["code"] = unique_code
            module_material = existing_process  # Use the existing process object
            module_material.save()
        else:
            module_material = database_bis_2030.new_node(
                code=unique_code,
                name=f"{i} from {supplier_name} fully processed",  # Fixed the formatting
                unit="kg",
            )

            module_material.save()
        print("process_list")
        print(process_list)
        for processes_i, process in zip(similar_process_list, process_list):
            print("processes_i[0]")
            print(processes_i[0])

            output = "Process_for_" + i + "_process_id_" + str(process[7])

            code = processes_i[0][0] + "_for_" + i + "_process_id_" + str(process[7])
            print("code")
            print(code)
            chain.append(("database_bis_2030", code))
            cut.append((("database_bis_2030", code), unique_code, output, 1))
            process = database_bis_2030.get(code)
            module_material.new_exchange(input=process, amount=1, type="technosphere", unit="unit").save()

        # Add exchanges to the module
        module_material.new_exchange(input=material, amount=1, type="technosphere", unit="unit").save()
        module_material.new_exchange(input=transport, amount=transport_amount, type="technosphere", unit="unit").save()
        module_material.new_exchange(input=module_material, amount=1, type="production", unit="unit").save()
        landfill_amount = (1 - recycling_rate) * landfill_rate
        incineration_amount = (1 - recycling_rate) * incineration_rate

        module_material.new_exchange(
            input=incineration, amount=incineration_amount, type="technosphere", unit="unit"
        ).save()
        module_material.new_exchange(input=landfill, amount=landfill_amount, type="technosphere", unit="unit").save()
        module_material.new_exchange(
            input=transport_waste, amount=1 / 1000 * 70, type="technosphere", unit="unit"
        ).save()

        module_material.save()
        print("CUT")
        print(cut)
        print("Module saved")
        my_object = {
            "name": module_material["name"],
            "outputs": [(("database_bis_2030", unique_code), material_id, 1)],
            "chain": [("database_bis_2030", unique_code), chain[0]],
            "cuts": [cut[0]],
            "output_based_scaling": True,
        }
        print("object created")

        print(my_object)

        created_modules.append(my_object)
        material_code.append(unique_code)
        print("MATERIAL CODE ADDED:")

    # Return the list of created modules

    return (created_modules, material_code[0])


def Create_Modules_Subproduct_2050(
    material_name,
    main_address,
    geolocator,
    database_2050,
    database_bis_2050,
    material_id,
    material_volume,
    process_list,
    df_Material,
    df_Process_Caracteristic,
    df_Supplier,
    constraints,
    material_index_formula,
    material_index_maxmin,
    increase_percentage_price,
):
    material_volume = float(material_volume)
    print(material_volume)
    # The scaling is for the material selection
    scaling = ["None", 10, 10, "None", "None", "None"]
    similar = find_similar_materials(
        material_name,
        df_Material,
        df_Process_Caracteristic,
        df_Supplier,
        constraints,
        material_index_formula,
        material_index_maxmin,
        increase_percentage_price,
    )[1]
    transport_waste = database_2050.get("462d96778c87f1897a413fb42bccbd72")
    df_Material_and_Distance = add_distance_to_material(
        similar, main_address, geolocator, df_Material, df_Process_Caracteristic, df_Supplier
    )
    print(df_Material_and_Distance)

    df_Material_And_EF = give_EF_Material(database_2050, similar, df_Material, df_Process_Caracteristic, df_Supplier)

    df_Material_And_Process = give_Process_Material(
        similar, process_list, df_Material, df_Process_Caracteristic, df_Supplier, material_name
    )

    # Initialize a list to store the created activities
    created_modules = []
    material_code = []
    names = []

    for i in similar:
        search_term_incineration = f"{i} incineration"
        search_term_landfill = f"{i} landfill"
        density_similar = df_Material.loc[df_Material["Material"] == i, "Density"].values
        density_similar = float(density_similar)

        incineration_rate = df_Material.loc[df_Material["Material"] == i, "IncinerationRate_2050"].values
        landfill_rate = df_Material.loc[df_Material["Material"] == i, "LandfillRate_2050"].values

        # Ensure values are numeric and handle missing cases
        if len(incineration_rate) > 0:
            incineration_rate = float(incineration_rate[0])  # Convert to float
        else:
            incineration_rate = 0.0  # Default value (adjust if needed)

        if len(landfill_rate) > 0:
            landfill_rate = float(landfill_rate[0])  # Convert to float
        else:
            landfill_rate = 0.0  # Default value (adjust if needed)

        incineration = find_closest_match(search_term_incineration, database_2050, True, False)
        incineration_key = incineration["code"]
        landfill = find_closest_match(search_term_landfill, database_2050, False, True)
        landfill_key = landfill["code"]
        recycling_rate = 0.05

        # Get the material's key and associated activity
        material_key = df_Material_And_EF.loc[df_Material_And_EF["Material"] == i, "Activity_key"].values[0]
        material = database_2050.get(material_key)

        transport = database_bis_2050.get("Transport_European")
        transport_amount = df_Material_and_Distance.loc[df_Material_and_Distance["Material"] == i, "Distance"].values[0]

        similar_process_list = df_Material_And_Process.loc[df_Material_And_Process["Material"] == i, "Process"].values[
            0
        ]
        print("similar_process_list")
        print(similar_process_list)
        time_list = df_Material_And_Process.loc[df_Material_And_Process["Material"] == i, "Time"].values[0]

        # for each process create an activity and a module
        for process, time, base_process in zip(similar_process_list, time_list, process_list):
            print("similar_process_list")
            print(similar_process_list)
            for process_alternative, time_alternative in zip(process, time):
                print("process_alternative")
                print(process_alternative)
                print("time_alternative")
                print(time_alternative)
                name = process_alternative[0] + "_for_" + i + "_process_id_" + str(base_process[7])
                names.append(name)
                output = "Process_for_" + i + "_process_id_" + str(base_process[7])
                existing_process = next((act for act in database_bis_2050 if act["name"] == name), None)
                if existing_process:
                    for exchange in list(existing_process.exchanges()):
                        exchange.delete()  # Remove the exchange
                    existing_process["name"] = name
                    existing_process["code"] = name
                    module_process = existing_process  # Use the existing process object
                    module_process.save()
                if not existing_process:
                    module_process = database_bis_2050.new_node(code=name, name=name, unit="unit")
                    module_process.save()
                process_alternative_act = database_bis_2050.get(process_alternative[0])
                print("process alternative act found!:")
                print(process_alternative_act)
                material_mass = density_similar * material_volume
                amount = time_alternative[0] / material_mass
                module_process.new_exchange(
                    input=process_alternative_act, amount=amount, type="technosphere", unit="min"
                ).save()
                module_process.new_exchange(input=module_process, amount=1, type="production", unit="unit").save()

                my_object = {
                    "name": name,
                    "outputs": [(("database_bis_2050", name), output, 1)],
                    "chain": [("database_bis_2050", name)],
                    "cuts": [],
                    "output_based_scaling": True,
                }

                created_modules.append(my_object)

        print("names")
        print(names)
        # Generate a unique code for each module
        unique_code = str(uuid.uuid4())

        chain = []
        cut = []
        supplier_name = df_Material_and_Distance.loc[df_Material_and_Distance["Material"] == i, "Supplier_name"].values
        if supplier_name.size > 0:
            supplier_name = supplier_name[0]
        else:
            supplier_name = "Unknown Supplier"
        existing_process = next(
            (act for act in database_bis_2050 if act["name"] == f"{i} from {supplier_name} fully processed"), None
        )

        if existing_process:
            for exchange in list(existing_process.exchanges()):
                exchange.delete()  # Remove the exchange
            existing_process["name"] = f"{i} from {supplier_name} fully processed"

            existing_process["code"] = unique_code
            module_material = existing_process  # Use the existing process object
            module_material.save()
        else:
            module_material = database_bis_2050.new_node(
                code=unique_code,
                name=f"{i} from {supplier_name} fully processed",  # Fixed the formatting
                unit="kg",
            )

            module_material.save()
        print("process_list")
        print(process_list)
        for processes_i, process in zip(similar_process_list, process_list):
            print("processes_i[0]")
            print(processes_i[0])

            output = "Process_for_" + i + "_process_id_" + str(process[7])

            code = processes_i[0][0] + "_for_" + i + "_process_id_" + str(process[7])
            print("code")
            print(code)
            chain.append(("database_bis_2050", code))
            cut.append((("database_bis_2050", code), unique_code, output, 1))
            process = database_bis_2050.get(code)
            module_material.new_exchange(input=process, amount=1, type="technosphere", unit="unit").save()

        # Add exchanges to the module
        module_material.new_exchange(input=material, amount=1, type="technosphere", unit="unit").save()
        module_material.new_exchange(input=transport, amount=transport_amount, type="technosphere", unit="unit").save()
        module_material.new_exchange(input=module_material, amount=1, type="production", unit="unit").save()
        landfill_amount = (1 - recycling_rate) * landfill_rate
        incineration_amount = (1 - recycling_rate) * incineration_rate

        module_material.new_exchange(
            input=incineration, amount=incineration_amount, type="technosphere", unit="unit"
        ).save()
        module_material.new_exchange(input=landfill, amount=landfill_amount, type="technosphere", unit="unit").save()
        module_material.new_exchange(
            input=transport_waste, amount=1 / 1000 * 70, type="technosphere", unit="unit"
        ).save()

        module_material.save()
        print("CUT")
        print(cut)
        print("Module saved")
        my_object = {
            "name": module_material["name"],
            "outputs": [(("database_bis_2050", unique_code), material_id, 1)],
            "chain": [("database_bis_2050", unique_code), chain[0]],
            "cuts": [cut[0]],
            "output_based_scaling": True,
        }
        print("object created")

        print(my_object)

        created_modules.append(my_object)
        material_code.append(unique_code)
        print("MATERIAL CODE ADDED:")

    # Return the list of created modules

    return (created_modules, material_code[0])
