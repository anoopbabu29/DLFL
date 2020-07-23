''' Modules '''
from typing import Dict, Tuple, List, Set, Any
import copy
import itertools
import time
import math
import threading

# Local Modules
import setup
import calc_helper
import tm_sim

# Defined types
DeltaOut = Tuple[int, str, str]
Delta = Dict[Tuple[int, str], DeltaOut]
DeltaMod = Dict[Tuple[int, str], bool]
Comb = Tuple[List[List[int]], List[List[int]]]

# Globals
CURR_TIME: float = 0
NUM_PERMS: int = 0
COUNT: int = 0
NUM_SKIPPED: int = 0
DELTAS: List[Delta] = []

# Lines 29-122 intends to replace get_mode_perms
# May not be beneficial


def get_combinations(pool: List[int], num_r: int, taken: List[int],
                     forg_pool: List[int]) -> Comb:
    ''' Returns all combinations of numbers '''
    combinations: List[List[int]] = []
    rem_combinations: List[List[int]] = []

    if len(taken) == num_r:
        return [taken], [forg_pool + pool]

    for i in range(len(pool) - num_r + len(taken) + 1):
        all_combinations: Comb = get_combinations(
            pool[i+1:], num_r, taken + [pool[i]], forg_pool + pool[:i])
        combinations += all_combinations[0]
        rem_combinations += all_combinations[1]

    return combinations, rem_combinations


def extract_perm(sigma: List[str], num_modes: int, ind: int) -> DeltaOut:
    ''' Get permutation from possible transitions via index '''
    move_dir: List[str] = ['L', 'R']

    if ind == 2 * len(sigma) * num_modes:
        return (-1, '-1', '-1')

    symb: str = sigma[int(ind/(2*num_modes))]
    ind -= int(ind/(2*num_modes)) * 2 * num_modes

    return int(ind/2), symb, move_dir[ind % 2]


def gen_mode_perm(sigma: List[str], num_modes: int,
                  ind: int) -> Dict[str, DeltaOut]:
    ''' Get permutation of possible modes via index '''
    perm: Dict[str, DeltaOut] = {}
    num_perm: int = 2 * len(sigma) * num_modes + 1

    for i, symb in enumerate(sigma):
        div_num: int = int(math.pow(num_perm, len(sigma) - i - 1))
        out_comb_ind: int = int(ind/div_num)

        perm[symb] = extract_perm(sigma, num_modes, out_comb_ind)

        ind -= out_comb_ind * div_num

    return perm


def gen_lim_mode_perm(sigma: List[str], num_modes: int,
                      ind: int) -> Dict[str, DeltaOut]:
    ''' Get permutation of possible modes
        containing Halt transition via index '''
    perm: Dict[str, DeltaOut] = {}
    num_perm: int = 2 * len(sigma) * num_modes

    poss_ranges: List[int] = []
    for i in range(len(sigma)):
        poss_ranges.append(0)
        if poss_ranges != []:
            poss_ranges[i] += poss_ranges[i-1]

        poss_ranges[i] += int(calc_helper.num_combinations(len(sigma), i + 1) *
                              math.pow(num_perm, len(sigma) - 1 - i))

    i_range: int = 0
    for poss_range in poss_ranges:
        low_range: int = poss_ranges[i_range-1] if i_range != 0 else 0
        if low_range <= ind < poss_range:
            break
        i_range += 1

    offset: int = int((poss_ranges[i_range] - low_range) /
                      calc_helper.num_combinations(len(sigma), i_range + 1))

    pot_combinations: Comb = get_combinations(list(range(len(sigma))),
                                              i_range + 1, [], [])

    off_ind: int = int((ind - low_range)/offset)

    halt_combs: List[int] = pot_combinations[0][off_ind]
    other_combs: List[int] = pot_combinations[1][off_ind]

    ind = ind - low_range - off_ind * offset

    for comb in halt_combs:
        perm[sigma[comb]] = (-1, '-1', '-1')

    for i, comb in enumerate(other_combs):
        div_num: int = int(math.pow(num_perm, len(other_combs) - i - 1))

        perm[sigma[comb]] = extract_perm(sigma, num_modes, int(ind/div_num))
        ind -= int(ind/div_num) * div_num

    return perm


def get_mode_perms(sigma: List[str],
                   num_modes: int) -> Tuple[List[Dict[str, DeltaOut]],
                                            List[Dict[str, DeltaOut]]]:
    ''' Generates the possible output combinations for a state '''
    halt: DeltaOut = (-1, '-1', '-1')

    output_set: List[List[Any]] = [list(range(num_modes)), sigma, ['L', 'R']]
    output_combinations: List[DeltaOut] = list(itertools.product(*output_set))
    output_combinations.append(halt)

    mode_set: List[List[DeltaOut]] = [output_combinations for _ in sigma]

    mode_combinations: List[Dict[str, DeltaOut]] = [
        {sigma[i]: delta_out for i, delta_out in enumerate(list(comb))}
        for comb in list(itertools.product(*mode_set))]

    lim_combinations: List[Dict[str, DeltaOut]] = list(filter(
        lambda comb: halt in comb.values(), mode_combinations))

    return mode_combinations, lim_combinations


def swap_modes(delta: Dict[Tuple[int, str], int], sigma: List[str],
               mode_a: int, mode_b: int) -> Dict[Tuple[int, str], int]:
    ''' Returns a delta where mode_a swaps with mode_b '''
    new_delta: Dict[Tuple[int, str], int] = copy.copy(delta)

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
    out_mode_set: List[List[int]] = [list(range(num_modes))
                                     for _ in range(len(sigma) * num_modes)]
    graph_set: List[Tuple[int, ...]] = list(itertools.product(*out_mode_set))
    num_unique: int = 0

    print(f'Number of Vertices, Edges: {num_modes}, {len(sigma)}')
    print(f'Number of Graphs: {len(graph_set)}, ' +
          f'Calc: {int(math.pow(num_modes, num_modes * len(sigma)))}')

    for i, graph in enumerate(graph_set):
        # print(f'Graph #{i+1}')

        delta: Dict[Tuple[int, str], int] = {}
        for edge_num, out_mode in enumerate(graph):
            mode: int = int(edge_num / len(sigma))
            symb: str = sigma[edge_num % len(sigma)]

            delta[(mode, symb)] = out_mode

        for mode_a in range(num_modes):
            for mode_b in range(mode_a + 1, num_modes):
                # print(f'{mode_a = }, {mode_b = }')
                # print(f'Original: {tuple(delta.values())}')
                swapped_delta: Tuple[int, ...] = tuple(swap_modes(
                    delta, sigma, mode_a, mode_b).values())
                # print(f'Swapped: {swapped_delta}')
                # print()

                if swapped_delta in graph_set and graph != swapped_delta:
                    graph_set.remove(swapped_delta)

                if graph == swapped_delta:
                    num_unique += 1

    print()
    print(f'Number of Unique Graphs: {len(graph_set)}, {num_unique = }')

    return len(graph_set)


def check_delta(delta: Delta,
                data_train: List[List[str]],
                data_valid: List[List[str]],
                num_modes: int,
                sigma: List[str],
                max_steps: int = 1000,
                debug_mode: bool = False) -> Tuple[bool, DeltaMod]:
    ''' Checks whether delta function satisfies data_train '''
    global NUM_SKIPPED

    if not tm_sim.djikstra(delta, num_modes, sigma):
        NUM_SKIPPED += 1
        return False, {}

    modified: DeltaMod = {(mode, symb): False for symb in sigma
                          for mode in range(num_modes)}

    for ind, tape in enumerate(data_train):
        output, modified, steps, is_success = tm_sim.run_tm(
            delta, copy.copy(tape), max_steps, modified, debug_mode)

        if debug_mode:
            print()
            print(f'Episode {ind+1}')
            print(f'Tape Input: {tape}')
            print(f'Target Output: {data_valid[ind]}')
            print(f'Tape Output: {output}')
            print(f'steps: {steps}')
            print()

        if not is_success or output != data_valid[ind]:
            return False, modified

    return True, modified


def copy_perms(perms1: List[Dict[str, DeltaOut]],
               perms2: List[Dict[str, DeltaOut]],
               perm: Dict[str, DeltaOut],
               is_first_lim: bool) -> Tuple[List[Dict[str, DeltaOut]],
                                            List[Dict[str, DeltaOut]]]:
    ''' Copies the permatations, and removes at index '''
    perms1_copy = copy.copy(perms1)
    perms2_copy = copy.copy(perms2)

    if is_first_lim:
        perms1_copy.remove(perm)
        if perm in perms2:
            perms2_copy.remove(perm)
    else:
        perms2_copy.remove(perm)
        if perm in perms1:
            perms1_copy.remove(perm)

    return perms1_copy, perms2_copy


def trav_factorial_perms(total_modes: int,
                         sigma: List[str],
                         data_train: List[List[str]],
                         data_valid: List[List[str]],
                         mode_perms: List[Dict[str, DeltaOut]],
                         lim_mode_perms: List[Dict[str, DeltaOut]],
                         lim_mode: int,
                         delta: Dict[Tuple[int, str], DeltaOut],
                         mode: int = 0) -> Tuple[Delta, DeltaMod, int]:
    ''' Method that goes through all perms in factorial
        Complexity based on (((s*q)^(s)!) '''
    global CURR_TIME, COUNT, NUM_SKIPPED, DELTAS

    if mode != total_modes:
        num_skip: int = 0
        has_mod_mode: bool = False
        modified: DeltaMod = {}
        prev_mode: Dict[str, DeltaOut] = {}
        is_lim_mode: bool = mode == lim_mode
        iter_perms: List[Dict[str, DeltaOut]] = (lim_mode_perms if is_lim_mode
                                                 else mode_perms)

        if iter_perms == {}:
            return {}, {}, 0

        for i, perm in enumerate(iter_perms):
            # print(f'{mode}: {is_lim_mode}')
            if has_mod_mode and modified != {}:
                for symb in sigma:
                    if modified[(mode, symb)] and (prev_mode[symb]
                                                   != perm[symb]):
                        has_mod_mode = False

                        # TODO: Skip to next transition
                        break

                num_skip = calc_helper.calc_num_skipped(
                    1, total_modes, len(sigma), mode, lim_mode)
                COUNT += num_skip
                NUM_SKIPPED += num_skip

                if has_mod_mode:
                    continue

            delta.update({(mode, sym): perm[sym] for sym in perm})

            mode_perms_copy, lim_perms = copy_perms(mode_perms, lim_mode_perms,
                                                    perm, not is_lim_mode)

            delta_res, modified, skipped = trav_factorial_perms(
                total_modes, sigma, data_train, data_valid, mode_perms_copy,
                lim_perms, lim_mode, delta, mode+1)

            if delta_res != {}:
                return delta_res, modified, 0

            if modified == {}:
                continue

            has_mod_mode = False
            for symb in sigma:
                if modified[(mode, symb)]:
                    has_mod_mode = True

            if has_mod_mode:
                # print(f'{skipped = }')
                # print()
                COUNT += skipped
                NUM_SKIPPED += skipped
                prev_mode = perm
            else:
                num_skip = calc_helper.calc_num_skipped(
                    len(iter_perms) - i - 1, total_modes,
                    len(sigma), mode, lim_mode) + skipped
                # print(f'{num_skip = }, {len(iter_perms) = }, {i = }, {mode = }')
                break

        return {}, modified, num_skip

    COUNT += 1

    is_valid, modified = check_delta(
        delta, data_train, data_valid, total_modes, sigma)

    if COUNT % 1000 == 0:
        print(chr(27) + "[2J")
        print(f'{COUNT}/{NUM_PERMS} - {COUNT*100/NUM_PERMS:.2f}% | ' +
              # f'{COUNT - NUM_SKIPPED} Checked, {NUM_SKIPPED} Skipped | ' +
              f'{time.time() - CURR_TIME:.2f}s')
        # print(f'{modified}')

    return delta if is_valid else {}, modified, 0


def pprint_perms(perms: List[Dict[str, DeltaOut]]) -> None:
    ''' Prints the permutations completely '''
    print()
    print(f'{len(perms) = }')
    for i, perm in enumerate(perms):
        print(i)
        for symb in perm:
            print(f'{symb}: {perm[symb]}')
        print()
    print()


def factorial_perms_method(num_modes: int, sigma: List[str],
                           data_train: List[List[str]],
                           data_valid: List[List[str]],
                           start_mode: int, end_mode: int,
                           is_thread: bool = False) -> Delta:
    ''' Method that calls the the traversal '''
    global CURR_TIME, DELTAS

    CURR_TIME = time.time()
    mode_perms, lim_mode_perms = get_mode_perms(sigma, num_modes)

    # print(f'{len(mode_perms) = }')
    # pprint_perms(mode_perms)

    # for i in range(int(math.pow(num_modes * len(sigma) * 2 + 1,
    #                             len(sigma)))):
    #     test_mode_perms.append(gen_mode_perm(sigma, num_modes, i))
    # pprint_perms(test_mode_perms)

    # print(test_mode_perms == mode_perms)

    # print(f'{len(lim_mode_perms) = }')
    # pprint_perms(lim_mode_perms)

    for mode in range(start_mode, end_mode):
        delta, _, _ = trav_factorial_perms(
            num_modes, sigma, data_train, data_valid, mode_perms,
            lim_mode_perms, mode, {})

        if delta != {}:
            if is_thread:
                DELTAS.append(delta)
            return delta

    if is_thread:
        DELTAS.append({})

    return {}


def thread_perms_method(num_modes: int, sigma: List[str],
                        data_train: List[List[str]],
                        data_valid: List[List[str]],
                        num_threads: int = 10) -> Delta:
    ''' Does factorial_perms_method threaded '''
    global DELTAS

    num_threads = num_threads if num_threads < num_modes else num_modes
    threads: List[threading.Thread] = []

    mode_ranges: List[Tuple[int, int]] = [(0, 0) for _ in range(num_threads)]

    for i in range(num_threads):
        prev_mode: int = 0 if i == 0 else mode_ranges[i-1][1]
        mode_ranges[i] = (prev_mode, (prev_mode + int(num_modes/num_threads) +
                                      (1 if num_modes % num_threads > i
                                       else 0)))

    for mode_range in mode_ranges:
        thread: threading.Thread = threading.Thread(
            target=factorial_perms_method,
            args=(num_modes, sigma, data_train, data_valid,
                  mode_range[0], mode_range[1], True))

        thread.start()
        threads.append(thread)

    # for thread in threads:
    #     thread.join()

    while (len(DELTAS) < num_threads and
           filter(lambda d: d != {}, DELTAS) == []):
        pass

    if filter(lambda d: d != {}, DELTAS) != []:
        return list(filter(lambda d: d != {}, DELTAS))[0]

    return {}


def main() -> None:
    ''' Main Function used for testing '''
    global NUM_PERMS

    test_enum_unique_graph(3, ['_', 's0'])

    quit()

    # Init data
    data_train, data_valid, action_set, state_set = setup.init_mult_data()

    # Generates the phi function and applies it to Data
    phi: Dict[str, str] = setup.gen_phi(action_set, state_set)
    setup.conv_data_to_tm(data_train, phi)
    setup.conv_data_to_tm(data_valid, phi)

    sigma: List[str] = list(phi.values())
    num_modes: int = 10
    sigma.append('s1')

    NUM_PERMS = calc_helper.calc_num_perms(num_modes, len(sigma))

    res_delta: Delta = factorial_perms_method(
        num_modes, sigma, data_train, data_valid, 0, num_modes)

    # res_delta: Delta = thread_perms_method(num_modes, sigma,
    #                                        data_train, data_valid)

    print(chr(27) + "[2J")
    print(f'{COUNT}/{NUM_PERMS} - {COUNT*100/NUM_PERMS:.2f}% | ' +
          f'{COUNT - NUM_SKIPPED} Checked, {NUM_SKIPPED} Skipped | ' +
          f'{time.time() - CURR_TIME:.2f}s')
    print(res_delta)


if __name__ == '__main__':
    main()
