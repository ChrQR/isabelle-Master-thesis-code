

from module import Module
import itertools
import numpy as np
import pickle
import networkx as nx

class ModularSystem(object):
    """
    A linked meta-process system holds several interlinked meta-processes. It has methods for:

    * loading / saving linked meta-process systems
    * returning information, e.g. product and process names, the product-process matrix
    * determining all alternatives to produce a given functional unit
    * calculating LCA results for individual meta-processes
    * calculating LCA results for a demand from the linked meta-process system (possibly for all alternatives)

    Meta-processes *cannot* contain:

    * 2 processes with the same name
    * identical names for products and processes (recommendation is to capitalize process names)

    Args:

    * *mp_list* (``[MetaProcess]``): A list of meta-processes
    """

    def __init__(self, mp_list=None):
        print("enter ")
        self.mp_list = []
        print("list odne")
        self.map_name_mp = {} 
        print("map name done ")
        self.map_processes_number = {}
        print("map process number done")
        self.map_products_number = dict()
        self.map_number_processes = dict()
        self.map_number_products = dict()
        self.name_map = {}  # {activity key: output name}
        self.raw_data = []
        self.has_multi_output_processes = False
        self.has_loops = False
        if mp_list:
            self.update(mp_list)
    def __getitem__(self, key):
        attributes = {
            "name": self.name,
            "outputs": self.outputs,
            "chain": self.chain,
            "cuts": self.cuts,
            "output_based_scaling": self.output_based_scaling,
        }
        # Allow access by key
        if isinstance(key, str):
            if key in attributes:
                return attributes[key]
            raise KeyError(f"Invalid key: {key}")

        # Optionally allow access by index (e.g., mp[0])
        elif isinstance(key, int):
            attr_list = list(attributes.values())
            if 0 <= key < len(attr_list):
                return attr_list[key]
            raise IndexError(f"Index {key} out of range.")
        
        # If key is neither int nor str, raise an error
        raise TypeError("Key must be a string or an integer.")

    def update(self, mp_list):
        """
        Updates the linked meta-process system every time processes
        are added, modified, or deleted.
        Errors are thrown in case of:

        * identical names for products and processes
        * identical names of different meta-processes
        * if the input is not of type MetaProcess()
        """
        product_names, process_names = set(), set()
        for mp in mp_list:
            if not isinstance(mp, Module):
                raise ValueError(u"Input must be of MetaProcesses type.")
            try:
                assert mp.name not in process_names  # check if process names are unique
                process_names.add(mp.name)
                product_names.update(self.get_product_names([mp]))
            except AssertionError:
                raise ValueError(u'Meta-Process names must be unique.')
        for product in product_names:
            if product in process_names:
                raise ValueError(u'Product and Process names cannot be identical.')
        self.mp_list = mp_list
        self.map_name_mp = dict([(mp.name, mp) for mp in self.mp_list])
        self.map_processes_number = dict(zip(self.processes, itertools.count()))
        self.map_products_number = dict(zip(self.products, itertools.count()))
        self.map_number_processes = {v: k for k, v in self.map_processes_number.items()}
        self.map_number_products = {v: k for k, v in self.map_products_number.items()}
        self.update_name_map()
        self.raw_data = [mp.mp_data for mp in self.mp_list]
        # multi-output
        self.has_multi_output_processes = False
        for mp in self.mp_list:
            if mp.is_multi_output:
                self.has_multi_output_processes = True
        # check for loops
        G = nx.DiGraph()
        G.add_edges_from(self.edges())
        if [c for c in nx.simple_cycles(G)]:
            self.has_loops = True
        else:
            self.has_loops = False

        print ('\nMeta-process system with', len(self.products), 'products and', len(self.processes), 'processes.')
        print ('Loops:', self.has_loops, ', Multi-output processes:', self.has_multi_output_processes)

    def update_name_map(self):
        """
        Updates the name map, which maps product names (outputs or cuts) to activity keys.
        This is used in the Activity Browser to automatically assign a product name to already known activity keys.
        """
        for mp in self.mp_list:
            for output in mp.outputs:
                self.name_map[output[0]] = self.name_map.get(output[0], set())
                self.name_map[output[0]].add(output[1])
            for cut in mp.cuts:
                self.name_map[cut[0]] = self.name_map.get(cut[0], set())
                self.name_map[cut[0]].add(cut[2])

    # SHORTCUTS

    @ property
    def processes(self):
        """Returns all process names."""
        return sorted([mp.name for mp in self.mp_list])

    @ property
    def products(self):
        for idx, module in enumerate(self.mp_list):
            """Returns all unique product names (the second value in the outputs)."""
            return sorted(
                set(
            itertools.chain.from_iterable(
                [output[1] for output in module.outputs if len(output) > 1] 
                for module in self.mp_list
            )
        )
    )

    # DATABASE METHODS (FILE I/O, LMPS MODIFICATION)



    # Commented out as it was declared multiple times
    # def update(self, mp_list):
    #     """ Updates the system with a new list of meta-processes. """
    #     # Implement logic to update the current system with the new modules
    #     self.mp_list = mp_list  # Replace current modules with new ones.

    def load_from_data(self, raw_data, append=False):
        """
        Loads a list of meta-process data and creates a Module object for each
        meta-process, then adds them to the linked meta-process system.

        Args:
            * raw_data (list): A list of dictionaries, where each dictionary represents a meta-process.
            * append (bool): If True, adds the loaded meta-processes to the existing database.
                              If False, replaces the existing meta-processes with the loaded ones.
        """
        # Ensure the data passed is a list
        if not isinstance(raw_data, list):
            print("is not instance")
            raise ValueError(f"Expected a list, but got {type(raw_data)}.")
        print('IS INSTANCE')
        # Iterate over the raw data and create Module instances
        mp_list = []
        for mp in raw_data:
            print('MP')
            print(mp)
            # Each meta-process dictionary must have 'name', 'outputs', 'chain', 'cuts', and 'output_based_scaling' keys.
            try:
                # Create the Module object for each meta-process
                module = Module(
                    name=mp['name'],
                    outputs=mp['outputs'],
                    chain=mp['chain'],
                    cuts=mp['cuts'],
                    output_based_scaling=mp.get('output_based_scaling', True)  # Default to True if not provided
                )
                print('MP_list creates')
                mp_list.append(module)
            except KeyError as e:
                raise ValueError(f'Missing expected key in meta-process data: {e}')
                print("missing key")
        # Store raw_data for later use
        self.raw_data = raw_data  # Ensure this is set


        # Now decide whether to append or replace the existing meta-processes
        if append:
            self.add_mp(mp_list, rename=True)
        else:
            self.update(mp_list)
            print("update")

    def __str__(self):
        return f"MetaProcessSystem with {len(self.mp_list)} processes."

    def save_to_file(self, filepath):
        """
        Saves data for each meta-process in the meta-process data format using pickle 
        and updates the linked meta-process system.

        Args:
            filepath (str): The path to the file where the data will be saved.
        """
        if not hasattr(self, 'raw_data'):
            raise AttributeError("raw_data attribute is not defined. Load data first using load_from_data.")
    
        with open(filepath, 'wb') as outfile:  # Use 'wb' for binary mode
            pickle.dump(self.raw_data, outfile)



    def add_mp(self, mp_list, rename=False):
        """
        Adds meta-processes to the linked meta-process system.

        *mp_list* can contain meta-process objects or the original data format used to initialize meta-processes.
        """
        print("enter add function")
        new_mp_list = []
        for mp in mp_list:
            # If mp is not an instance of Module, convert it to a Module
            if not isinstance(mp, Module):
                print("it is not an instance")
                mp = Module(**mp)
        
            # Check if the module already exists in the list based on its name
            # This assumes that each Module has a unique 'name' attribute
            if not any(existing_mp.name == mp.name for existing_mp in self.mp_list):
                new_mp_list.append(mp)
            else:
                print(f"Module with name {mp.name} already exists. Skipping addition.")
    
        # Handle renaming if necessary
        if rename:
            for mp in new_mp_list:
                if mp.name in self.processes:
                    mp.name += '__ADDED'
    
        # Update the system with the new modules
        self.update(self.mp_list + new_mp_list)


    def remove_mp(self, mp_list):
        """
        Remove meta-processes from the linked meta-process system.

        *mp_list* can be a list of meta-process objects or meta-process names.
        """
        print("enter function remove")
        for mp in mp_list:
            if not isinstance(mp, Module):  # If it's not already a Module, look it up
                mp_objects = self.get_processes([mp])  # Assuming this returns a list of Module objects
                if mp_objects:
                    mp = mp_objects[0]  # Take the first object if a list is returned
                else:
                    continue  # Skip if no matching process is found
            self.mp_list.remove(mp)
        self.update(self.mp_list)

    # METHODS THAT RETURN DATA FOR A SUBSET OR THE ENTIRE LMPS
    def get_processes(self, mp_list=None):
        """
        Returns a list of meta-processes.

        *mp_list* can be a list of meta-process objects or meta-process names.
        """
        # if empty list return all meta-processes
        if not mp_list:
            return self.mp_list
        else:
            # if name list find corresponding meta-processes
            if not isinstance(mp_list[0], Module):
                self.map_name_mp = self.map_name_mp = dict([(mp.name, mp) for mp in self.mp_list])
                
                return [self.map_name_mp.get(name, None) for name in mp_list if name in self.processes]
            else:
                return mp_list

    def get_process_names(self, mp_list=None):
        """Returns a the names of a list of meta-processes."""
        return sorted([mp.name for mp in self.get_processes(mp_list)])



        
    def get_product_names(self, mp_list=None):
        """Returns all unique product names (the second value in the outputs) for a list of meta-processes.

        *mp_list* can be a list of meta-process objects or meta-process names.
        """
        # If mp_list is not provided, use self.mp_list (all processes)
        if mp_list is None:
            mp_list = self.mp_list
    
        # Extract product names from the outputs of the modules in mp_list
        return sorted(
            set(
                itertools.chain.from_iterable(
                    [output[1] for output in module.outputs if len(output) > 1] 
                    for module in mp_list
                )
            )
        )


    def get_output_names(self, mp_list=None):
        """ Returns output product names for a list of meta-processes."""
        return sorted(list(set([name for mp in self.get_processes(mp_list) for name in mp.output_names])))

    def get_cut_names(self, mp_list=None):
        """ Returns cut/input product names for a list of meta-processes."""
        return sorted(list(set([name for mp in self.get_processes(mp_list) for name in mp.cut_names])))

    def product_process_dict(self, mp_list=None, process_names=None, product_names=None):
        """
        Returns a dictionary that maps meta-processes to produced products (key: product, value: meta-process).
        Optional arguments ``mp_list``, ``process_names``, ``product_names`` can used as filters.
        """
        if not process_names:
            process_names = self.processes
        if not product_names:
            product_names = self.products
        product_processes = {}
        for mp in self.get_processes(mp_list):
            for output in mp.outputs:
                output_name = output[1]
                if output_name in product_names and mp.name in process_names:
                    product_processes[output_name] = product_processes.get(output_name, [])
                    product_processes[output_name].append(mp.name)
        return product_processes

    def edges(self, mp_list=None):
        """
        Returns an edge list for all edges within the linked meta-process system.

        *mp_list* can be a list of meta-process objects or meta-process names.
        """
        edges = []
        for mp in self.get_processes(mp_list):
            for cut in mp.cuts:
                edges.append((cut[2], mp.name))
            for output in mp.outputs:
                edges.append((mp.name, output[1]))
        return edges

    def get_pp_matrix(self, mp_list=None):
        """
        Returns the product-process matrix as well as two dictionaries
        that hold row/col values for each product/process.

        *mp_list* can be used to limit the scope to the contained processes
        """
        
        print(self.get_processes(mp_list))
        mp_list = self.get_processes(mp_list)
        print("mp list", mp_list)
        print("number of rows", len(self.get_product_names(mp_list)))
        print("number of columns",len(mp_list))
        
        matrix = np.zeros((len(self.get_product_names(mp_list)), len(mp_list)))
        map_processes_number = dict(zip(self.get_process_names(mp_list), itertools.count()))
        print("process number",map_processes_number )
        map_products_number = dict(zip(self.get_product_names(mp_list), itertools.count()))
        print("product number",map_products_number )
        
                
        for mp in mp_list:
            for product, amount in mp.pp:
                matrix[map_products_number[product], map_processes_number[mp.name]] += amount
                
        return matrix, map_processes_number, map_products_number

    # ALTERNATIVE PATHWAYS

    def upstream_products_processes(self, product):
        """Returns all upstream products and processes related to a certain product (functional unit)."""
        G = nx.DiGraph()
        G.add_edges_from(self.edges())
        product_ancestors = nx.ancestors(G, product)  # set
        product_ancestors.update([product])  # add product (although not an ancestor in a strict sense)
        # split up into products and processes
        ancestor_processes = [a for a in product_ancestors if a in self.processes]
        ancestor_products = [a for a in product_ancestors if a in self.products]
        return ancestor_processes, ancestor_products

    def all_pathways(self, functional_unit):
        """
        Returns all alternative pathways to produce a given functional unit. Data output is a list of lists.
        Each sublist contains one path made up of products and processes.
        The input Graph may not contain cycles. It may contain multi-output processes.

        Args:

        * *functional_unit*: output product
        """
        print('inside all pathway')
        print('self')
        print(self)
        print('functional_unit')
        print(functional_unit)
        def dfs(current_node, visited, parents, direction_up=True):
            print(f"Predecessors of {current_node}: {list(G.predecessors(current_node))}")
            print(f"Current node: {current_node}, Direction up: {direction_up}")
            print(f"Visited: {visited}")

            if direction_up:
                visited += [current_node]
            if current_node in self.products:
                # go up to all processes if none has been visited previously, else go down
                upstream_processes = list(G.predecessors(current_node))
                if upstream_processes and not [process for process in upstream_processes if process in visited]:
                    
                    parents += [current_node]
                    for process in upstream_processes:
                        
                        dfs(process, visited[:], parents[:])  # needs a real copy due to mutable / immutable
                else:  # GO DOWN or finish
                    if parents:
                        downstream_process = parents.pop()
                        dfs(downstream_process, visited, parents, direction_up=False)
                    else:
                        results.append(visited)
                        print ('Finished')
            else:  #node = process; upstream = product
                # go to one upstream product, if there is one unvisited, else go down
                upstream_products = list(G.predecessors(current_node))
                unvisited = [product for product in upstream_products if product not in visited]
                print ('unvisited:', unvisited)
                if unvisited:  # GO UP
                    parents += [current_node]
                    dfs(unvisited[0], visited, parents)
                else:  # GO DOWN or finish
                    if parents:
                        downstream_product = parents.pop()
                        dfs(downstream_product, visited, parents, direction_up=False)
                    else:
                        print ('Finished @ process, this should not happen if a product was demanded.')
            return results

        results = []
        G = nx.DiGraph()
        print('G done')
        G.add_edges_from(self.edges())
        print('all edges added')
        print(G)
        print("Nodes in the graph:", list(G.nodes))

        print(dfs(functional_unit, [], []))
        

        return dfs(functional_unit, [], [])

    # LCA

    def scaling_vector_foreground_demand(self, mp_list, demand):
        """
        Returns a scaling dictionary for a given demand and matrix defined by a list of processes (or names).
        Keys: process names. Values: scaling vector values.

        Args:

        * *mp_list*: meta-process objects or names
        * *demand* (dict): keys: product names, values: amount
        """
        # matrix
        matrix, map_processes, map_products = self.get_pp_matrix(mp_list)
        try:
            # TODO: define conditions that must be met (e.g. square, single-output); Processes can still have multiple outputs (system expansion)
            assert matrix.shape[0] == matrix.shape[1]  # matrix needs to be square to be invertable!
            # demand vector
            demand_vector = np.zeros((len(matrix),))
            for name, amount in demand.items():
                demand_vector[map_products[name]] = amount
            # scaling vector
            scaling_vector = np.linalg.solve(matrix, demand_vector).tolist()
            scaling_dict = dict([(name, scaling_vector[index]) for name, index in map_processes.items()])
            # # foreground product demand (can be different from scaling vector if diagonal values are not 1)
            # foreground_demand = {}
            # for name, amount in scaling_dict.items():
            #     number_in_matrix = map_processes[name]
            #     product = [name for name, number in map_products.items() if number == number_in_matrix][0]
            #     foreground_demand.update({
            #         product: amount*matrix[number_in_matrix, number_in_matrix]
            #     })
            return scaling_dict  # , foreground_demand
        except AssertionError:
            print ("Product-Process Matrix must be square! Currently", matrix.shape[0], 'products and', matrix.shape[1], 'processes.')

    # def lca_processes(self, method, process_names=None, factorize=False):
    #     """Returns a dictionary where *keys* = meta-process name, *value* = LCA score
    #     """
    #     print('METHOD USED IN LCA_PROCESSES IS ')
    #     print(method)
    #     print('SELF OBJECT IS ')
    #     print(self)
    #     print('REULTS ')
    #     for mp in self.get_processes(process_names):
    #         print(mp)
    #         print(mp.name)
    #         print(method)
    #         print(mp.lca(method, factorize=factorize))
            
            
    #     print(dict([(mp.name, mp.lca(method, factorize=factorize))
    #                  for mp in self.get_processes(process_names)]))
    #     return dict([(mp.name, mp.lca(method, factorize=factorize))
    #                  for mp in self.get_processes(process_names)])
    def lca_processes(self, method, process_names=None, factorize=False):
        """Returns a dictionary where *keys* = meta-process name, *value* = fresh LCA score"""
    
        print(f"\n🔄 Running fresh LCA calculations for method: {method}")

        results = {}
        for mp in self.get_processes(process_names):
            print(f"Processing: {mp.name} with method: {method}")

            # Force recalculation: If mp caches results, clear them first
            if hasattr(mp, "clear_cache"):
                mp.clear_cache()  # Try clearing cache if the function exists
        
            if hasattr(mp, "cached_lca"):
                mp.cached_lca = {}  # Reset any cached LCA results

            # Ensure method is correctly passed
            assert method is not None, "❌ ERROR: LCIA method is None!"
        
            # Explicitly check if method is used inside `mp.lca()`
            print(f"Calling mp.lca() with method {method}")
            score = mp.lca(method, factorize=factorize)

            # Log unexpected identical results
            if mp.name in results and results[mp.name] == score:
                print(f"⚠️ Warning: {mp.name} has the same LCA score as before! Check mp.lca() implementation.")

            results[mp.name] = score

            print(f"✅ {mp.name} → LCA Score: {results[mp.name]}")

        print(f"📊 Final LCA results for method {method}: {results}")
        return results



    # def lca_linked_processes(self, method, process_names, demand):
    #     """
    #     Performs LCA for a given demand from a linked meta-process system.
    #     Works only for square matrices (see scaling_vector_foreground_demand).

    #     Returns a dictionary with the following keys:

    #     * *path*: involved process names
    #     * *demand*: product demand
    #     * *scaling vector*: result of the demand
    #     * *LCIA method*: method used
    #     * *process contribution*: contribution of each process
    #     * *relative process contribution*: relative contribution
    #     * *LCIA score*: LCA result

    #     Args:

    #     * *method*: LCIA method
    #     * *process_names*: selection of processes from the linked meta-process system (that yields a square matrix)
    #     * *demand* (dict): keys: product names, values: amount
    #     """
    #     scaling_dict = self.scaling_vector_foreground_demand(process_names, demand)
    #     if not scaling_dict:
    #         return
    #     lca_scores = self.lca_processes(method, process_names)
    #     # multiply scaling vector with process LCA scores
    #     path_lca_score = 0.0
    #     process_contribution = {}
    #     for process, amount in scaling_dict.items():
    #         process_contribution.update({process: amount*lca_scores[process]})
    #         path_lca_score = path_lca_score + amount*lca_scores[process]
    #     process_contribution_relative = {}
    #     for process, amount in scaling_dict.items():
    #         process_contribution_relative.update({process: amount*lca_scores[process]/path_lca_score})

    #     output = {
    #         'path': process_names,
    #         'demand': demand,
    #         'scaling vector': scaling_dict,
    #         'LCIA method': method,
    #         'process contribution': process_contribution,
    #         'relative process contribution': process_contribution_relative,
    #         'LCA score': path_lca_score,
    #     }
    #     return output

    def lca_linked_processes(self, method, process_names, demand):
    

        print(f"Running LCA for method: {method} and process: {process_names}")

        # Force fresh calculations for each call
        scaling_dict = self.scaling_vector_foreground_demand(process_names, demand)
        if not scaling_dict:
            print("⚠️ Warning: Empty scaling vector. No calculation performed.")
            return None  # Explicitly return None to indicate failure

        # Ensure fresh LCA scores per method
        lca_scores = self.lca_processes(method, process_names)
        print('-------LCA SCORE---------')
        print(lca_scores)
        if not lca_scores:
            print("⚠️ Warning: No LCA scores returned.")
            return None

        # Multiply scaling vector with process LCA scores
        path_lca_score = 0.0
        process_contribution = {}
        for process, amount in scaling_dict.items():
            score = amount * lca_scores.get(process, 0)  # Default to 0 if missing
            process_contribution[process] = score
            path_lca_score += score  # Accumulate total LCA score

        # Prevent division by zero
        process_contribution_relative = {}
        if path_lca_score != 0:
            for process, amount in scaling_dict.items():
                process_contribution_relative[process] = process_contribution[process] / path_lca_score
        else:
            print("⚠️ Warning: Path LCA score is zero. Setting relative contributions to zero.")
            process_contribution_relative = {proc: 0 for proc in process_contribution}

        # Output dictionary
        output = {
            'path': process_names[:],  # Ensure a copy of the list
            'demand': demand.copy(),   # Ensure a copy of the dictionary
            'scaling vector': scaling_dict.copy(),
            'LCIA method': method,
            'process contribution': process_contribution.copy(),
            'relative process contribution': process_contribution_relative.copy(),
            'LCA score': path_lca_score,
        }

        print(f"LCA Score for {process_names}: {path_lca_score}")
        return output


    def lca_alternatives(self, method, demand):
        print(self)
        print('LCA ALTERNATIVE')
        """
        Calculation of LCA results for all alternatives in a linked meta-process system that yield a certain demand.
        Results are stored in a list of dictionaries as described in 'lca_linked_processes'.

        Args:

        * *method*: LCIA method
        * *demand* (dict): keys: product names, values: amount
        """
        if self.has_multi_output_processes:
            print ('\nCannot calculate LCAs for alternatives as system contains ' \
                  'loops (', self.has_loops, ') / multi-output processes (', self.has_multi_output_processes, ').')
        else:
            # assume that only one product is demanded for now (functional unit)
            path_lca_data = []
            print ('looking for all pathwats')
            print('list(demand.keys())[0]')
            print(list(demand.keys())[0])
            for path in self.all_pathways(list(demand.keys())[0]):
                print('PATH')
                print(path)
                path_lca_data.append(self.lca_linked_processes(method, path, demand))
            return path_lca_data
