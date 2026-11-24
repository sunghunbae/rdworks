
## Graph.add_node(node_for_adding, **attr)[source]

    Add a single node node_for_adding and update node attributes.

    Parameters:

        node_for_adding: node
            A node can be any hashable Python object except None.
        attr : keyword arguments, optional
            Set or change node attributes using key=value.

    Examples:

        G = nx.Graph()  # or DiGraph, MultiGraph, MultiDiGraph, etc
        G.add_node(1)
        G.add_node("Hello")
        K3 = nx.Graph([(0, 1), (1, 2), (2, 0)])
        G.add_node(K3)
        G.number_of_nodes()
        >>> 3

        G.add_node(1, size=10)
        G.add_node(3, weight=0.4, UTM=("13S", 382871, 3972649))


## Graph.add_edge(u_of_edge, v_of_edge, **attr)[source]

    Add an edge between u and v.
    The nodes u and v will be automatically added if they are not already in the graph.
    Edge attributes can be specified with keywords or by directly accessing the edge's attribute dictionary.

    Parameters:

        u_of_edge, v_of_edge: nodes
        Nodes can be, for example, strings or numbers. Nodes must be hashable (and not None) Python objects.

        attr: keyword arguments, optional
        Edge data (or labels or objects) can be assigned using keyword arguments.

    Examples:

        G = nx.Graph()  # or DiGraph, MultiGraph, MultiDiGraph, etc
        e = (1, 2)
        G.add_edge(1, 2)  # explicit two-node form
        G.add_edge(*e)  # single edge as tuple of two nodes
        G.add_edges_from([(1, 2)])  # add edges from iterable container

        G.add_edge(1, 2, weight=3)
        G.add_edge(1, 3, weight=7, capacity=15, length=342.7)

        G.add_edge(1, 2)
        G[1][2].update({0: 5})
        G.edges[1, 2].update({0: 5})

## Isomorphism

    In NetworkX, the node_label parameter in the vf2pp_all_isomorphisms function is the name of a node attribute 
    that is used to compare nodes when checking for isomorphism between two graphs. 
    The default value for node_label is None, which means that node attributes are not considered in the comparison. 
    If a node doesn't have the node_label attribute, the default_label value is used instead.
    


## API

::: rdworks.utils
    handler: python

::: rdworks.units
    handler: python
    
::: rdworks.readers
    handler: python

::: rdworks.writers
    handler: python

::: rdworks.view
    handler: python

::: rdworks.descriptor
    handler: python

::: rdworks.xml
    handler: python