''' Modules '''
import math
from typing import Dict, Tuple, List
import time


def append_combo(list_to_app: List[List[int]],
                 curr_list: List[int],
                 num_to_app: int, 
                 max_len: int) -> List[List[int]]:
    '''Appends to list at every step in init_graph'''
    res_list: List[List[int]] = [combo + [num_to_app] for combo in list_to_app]
    if num_to_app != max_len - 1:
        res_list.append(curr_list + [num_to_app + 1])
    
    return res_list


def init_graph(num_modes: int, num_sigma: int, num_dir: int) -> Dict[Tuple[int, ...], List[List[int]]]:
    ''' Returns the graph '''
    graph: Dict[Tuple[int, ...], List[List[int]]] = {}
    num_1_combo: int = num_modes * num_sigma * num_dir

    s_time: float = time.time()
    for a in range(num_1_combo):
        list_to_app: List[List[int]] = append_combo([], [], a, num_1_combo)
        for b in range(num_1_combo):
            list_to_app = append_combo(list_to_app, [a], b, num_1_combo)
            for c in range(num_1_combo):
                list_to_app = append_combo(list_to_app, [a, b], c, num_1_combo)
                for d in range(num_1_combo):
                    list_to_app = append_combo(list_to_app, [a, b, c], d, num_1_combo)
                    for e in range(num_1_combo):
                        list_to_app = append_combo(list_to_app, [a, b, c, d], e, num_1_combo)
                        for f in range(num_1_combo):
                            graph[(a, b, c, d, e, f)] = append_combo(list_to_app, [a, b, c, d, e], f, num_1_combo)

    print(f'Init time: {time.time() - s_time}s')
    return graph
                            


def main():
    ''' Main Function that will be tested '''
    pass


if __name__ == '__main__':
    main()