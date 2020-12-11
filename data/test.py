from build_graph import *
import seaborn as sb

# 1- Start by defining our favorite regular structure

width_basis = 15
nbTrials = 20


################################### EXAMPLE TO BUILD A SIMPLE REGULAR STRUCTURE ##########
## REGULAR STRUCTURE: the most simple structure:  basis + n small patterns of a single type

### 1. Choose the basis (cycle, torus or chain)
basis_type = "cycle"

### 2. Add the shapes
nb_shapes = 5  ## numbers of shapes to add
#shape = ["fan",6] ## shapes and their associated required parameters  (nb of edges for the star, etc)
#shape = ["star",6]
list_shapes = [["house"]] * nb_shapes

### 3. Give a name to the graph
identifier = 'AA'  ## just a name to distinguish between different trials
name_graph = 'houses' + identifier
sb.set_style('white')

### 4. Pass all these parameters to the Graph Structure
add_edges = 4 ## nb of edges to add anywhere in the structure
del_edges  =0

G, communities, plugins, role_id = build_structure(width_basis, basis_type, list_shapes, start=0,
                            rdm_basis_plugins =False, add_random_edges=0,
                            plot=True, savefig=True)