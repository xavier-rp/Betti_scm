import gudhi
import numpy as np
import matplotlib.pyplot as plt
import itertools
import time

site_facet_list = []
site_facet_string = []
with open("/home/xavier/Documents/Projet/scm/datasets/facet_list_c1_as_simplices.txt") as f:
    for l in f:
        site_facet_list.append([int(x) for x in l.strip().split()])
        site_facet_string.append([str(x) for x in l.strip().split()])
# G, site_node_list = facet_list_to_bipartite(site_facet_list)

st = gudhi.SimplexTree()
i = 0
for facet in site_facet_list:
    print(facet)
    #if len(facet) > 3:
    #    for face in itertools.combinations(facet, 3):
    #        st.insert(face)
    #        st.assign_filtration(face, 3)
    st.insert(facet)
    st.assign_filtration(facet, len(facet))
    #else:
    #    st.insert(facet)
    print(i)
    i += 1

print('_____________________________________________________________')

save_pers = st.persistence()
#st.write_persistence_diagram('test2')
print(len(save_pers))
print(len(st.get_filtration()))
#print(save_pers)

gudhi.plot_persistence_barcode(persistence_file='test2', max_barcodes=0, inf_delta=0.1)
plt.show()

