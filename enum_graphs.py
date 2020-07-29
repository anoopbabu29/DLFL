''' Finds Unique Graphs '''
# Modules
from typing import Dict, Tuple, List, Iterable, Set, Any
import itertools
import math
import time
import copy
import json

# Local Modules
import calc_helper


# Classes
class Node:
    ''' Node class used to create a tree '''
    def __init__(self, data: int, depth: int, children: List[Any]) -> None:
        self.data: int = data
        self.depth: int = depth
        if children == []:
            self.children: List[Any] = [None] * (depth + 1)
        else:
            self.children = children

    def add_node(self, ind: int) -> None:
        ''' Adds node to children '''
        if ind > self.depth:
            return

        self.children[ind] = Node(ind, self.depth + 1, [])

    def get_node(self, ind: int):
        ''' Gets node from children '''
        if ind > self.depth:
            return None

        return self.children[ind]


def node_to_dict(node: Node) -> Dict[str, Any]:
    ''' Converts node to dict '''
    node.children = [node_to_dict(child)
                     if child is not None and not isinstance(child, dict)
                     else child for child in node.children]

    return node.__dict__


def dict_to_node(node_dict: Dict[str, Any]) -> Node:
    ''' Convertes dict to node '''
    node_dict['children'] = [dict_to_node(child) if child is not None else None
                             for child in node_dict['children']]

    return Node(node_dict['data'], node_dict['depth'], node_dict['children'])


def flatten_nodes(node: Node) -> List[Any]:
    ''' Turn Nodes to a collection of lists  '''
    return [(node.data, flatten_nodes(child))
            for child in node.children if child is not None]


# Globals
GRAPH_TREE_DICT: Dict[int, Dict[str, Any]] = json.loads(open('graph_tree.txt')
                                                        .read())
GRAPH_TREE: Dict[int, Node] = {edge: dict_to_node(GRAPH_TREE_DICT[edge])
                               for edge in GRAPH_TREE_DICT}
TREE_REP: Dict[int, List[int]] = json.loads(open('tree_rep.txt').read())


def save_trees() -> None:
    ''' Saves Trees to a file '''
    with open('tree_rep.txt', 'w') as tr_file:
        tr_file.write(json.dumps(TREE_REP))

    with open('graph_tree.txt', 'w') as gt_file:
        graph_tree_dict: Dict[int, Dict[str, Any]] = {
            edge: node_to_dict(GRAPH_TREE[edge]) for edge in GRAPH_TREE}
        gt_file.write(json.dumps(graph_tree_dict))


def update_tree(vertices: int, edge: int,
                unique_graphs: List[Tuple[int, ...]]) -> None:
    ''' Updates Global variables after traversal '''
    if edge not in GRAPH_TREE:
        GRAPH_TREE[edge] = Node(-1, 1, [])

    if edge not in TREE_REP:
        TREE_REP[edge] = []

    if vertices < 2:
        return

    for graph in unique_graphs:
        current_node: Node = GRAPH_TREE[edge]

        for next_vertex in graph:
            if current_node.get_node(next_vertex) is None:
                current_node.add_node(next_vertex)

                if len(TREE_REP[edge]) < current_node.depth:
                    TREE_REP[edge].append(0)
                TREE_REP[edge][current_node.depth - 1] += 1

            current_node = current_node.get_node(next_vertex)

            if current_node.depth > vertices:
                break

    # print(TREE_REP[edge])
    # print(node_to_dict(GRAPH_TREE[edge]))
    save_trees()


def swap_modes(delta: Dict[Tuple[int, str], int], sigma: List[str],
               mode_a: int, mode_b: int) -> Dict[Tuple[int, str], int]:
    ''' Returns a delta where mode_a swaps with mode_b '''
    new_delta: Dict[Tuple[int, str], int] = delta

    temp_delta: Dict[Tuple[int, str], int] = {edge_info: new_delta[edge_info]
                                              for edge_info in new_delta
                                              if edge_info[0] == mode_a}

    for symb in sigma:
        new_delta[(mode_a, symb)] = new_delta[(mode_b, symb)]

    for edge_info in temp_delta:
        new_delta[(mode_b, edge_info[1])] = temp_delta[edge_info]

    for edge_info in new_delta:
        # print(f'{edge_info = }, {new_delta[edge_info] = }')
        if new_delta[edge_info] == mode_a:
            new_delta[edge_info] = mode_b
        elif new_delta[edge_info] == mode_b:
            new_delta[edge_info] = mode_a

    return new_delta


def test_enum_unique_graph(num_modes: int, sigma: List[str]) -> int:
    ''' Enumerate over nonsimple, directed graphs '''
    # Creates set of all graphs with num_modes vertices and sigma edges/vertex
    out_mode_set: List[List[int]] = [list(range(num_modes))
                                     for _ in range(len(sigma) * num_modes)]
    graph_set: Iterable[Tuple[int, ...]] = itertools.product(*out_mode_set)

    # Inits helper variables
    skip_set: Set[Tuple[int, ...]] = set()
    unique_graphs: List[Tuple[int, ...]] = []
    duplicates: List[int] = [0] * calc_helper.factorial(num_modes)
    dups: Dict[int, List[int]] = {}
    total_graphs: int = int(math.pow(num_modes, num_modes * len(sigma)))

    print(chr(27) + "[2J")
    print(f'Number of Vertices, Edges: {num_modes}, {len(sigma)}')
    print(f'Number of Graphs: {total_graphs}')

    curr_time: float = time.time()

    # Goes through all types of graphs
    for i, graph in enumerate(graph_set):
        if graph in skip_set:
            continue

        # Adds graph to skip_set and unique set
        skip_set.add(graph)
        unique_graphs.append(graph)

        num_dups: int = 0
        delta: Dict[Tuple[int, str], int] = {}

        # Fills delta based on graph
        for edge_num, out_mode in enumerate(graph):
            mode: int = int(edge_num / len(sigma))
            symb: str = sigma[edge_num % len(sigma)]

            delta[(mode, symb)] = out_mode

        # For all permutations, check for different versions of graph
        for perm in calc_helper.gen_perms_wo_rep(list(range(num_modes))):
            swapped_delta: Dict[Tuple[int, str], int] = copy.copy(delta)

            # Swaps the vertices to each permutation
            for j, _ in enumerate(perm):
                while j != perm[j]:
                    temp: int = perm[j]
                    perm[j] = perm[temp]
                    perm[temp] = temp

                    swapped_delta = swap_modes(swapped_delta, sigma, j, temp)

            # If it is a new version of the graph, add to skip_set
            if tuple(swapped_delta.values()) not in skip_set:
                skip_set.add(tuple(swapped_delta.values()))
                num_dups += 1

        duplicates[num_dups] += 1
        dups[num_dups + 1] = calc_helper.prime_fact(num_dups + 1)

        # Print status
        if (i + 1) % 1 == 0:
            time_taken: float = time.time() - curr_time
            count: int = sum([(k+1) * dup for k, dup in enumerate(duplicates)])
            est_time: float = time_taken * ((total_graphs - count) / count)

            print(chr(27) + "[2J")
            print(f'Number of Vertices, Edges: {num_modes}, {len(sigma)}')
            print(f'Count: {count}/{total_graphs} | ' +
                  f'Time Taken: {time_taken:.2f}/{time_taken+est_time:.2f}s' +
                  f' {100* time_taken/(time_taken+est_time):.2f}%')

    # Print Final report
    print()
    print(f'Number of Unique Graphs: {len(unique_graphs)}')
    # print(f', {duplicates = }')
    for graph in unique_graphs:
        print(graph)
    print()
    for num in dups:
        print(f'{num}, {duplicates[num-1]}: {dups[num]}')

    # update_tree(num_modes - 2, len(sigma), unique_graphs)

    return len(unique_graphs)


def main() -> None:
    ''' Main Function used for testing '''
    test_enum_unique_graph(5, ['_'])


if __name__ == '__main__':
    main()
