''' Modules '''
from typing import Dict, Tuple, List, Any
import copy
import itertools
import time
import math
import threading

# Local Modules
import driver

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


def factorial(num: int) -> int:
    ''' Calculates the factorial of a number'''
    return num * factorial(num - 1) if num > 1 else 1


def fact_range(num_a: int, num_b: int) -> int:
    ''' Calculates the factorial of num_a until it reaches num_b '''
    return num_a * fact_range(num_a - 1, num_b) if num_a > num_b else 1


def num_combinations(num_n: int, num_r: int) -> int:
    ''' Calculates the number of combinations of a number '''
    return int(fact_range(num_n, num_r) / factorial(num_n - num_r))


def calc_num_lim_perms(num_modes: int, num_sigma: int) -> int:
    ''' Calculates the number of permatations of limited modes '''
    num_perm: int = 2 * num_modes * num_sigma + 1

    return sum([int(num_combinations(num_sigma, i + 1) *
                    math.pow(num_perm - 1, num_sigma - (i + 1)))
                for i in range(num_sigma)])


def calc_num_comm_perms(num_modes: int, num_sigma: int) -> int:
    ''' Calculates the number of permutations of common modes '''
    num_perm: int = 2 * num_modes * num_sigma + 1

    return int(math.pow(num_perm, num_sigma) - 1)


def calc_num_perms(num_modes: int, num_sigma: int) -> int:
    ''' Calculates the number of permutations '''
    num_lim: int = num_modes * calc_num_lim_perms(num_modes, num_sigma)

    fact_fract: int = fact_range(calc_num_comm_perms(num_modes, num_sigma),
                                 calc_num_comm_perms(num_modes, num_sigma) -
                                 num_modes + 1)

    print(f'{num_lim = } {fact_fract = }')

    return int(num_lim * fact_fract)


# Lines 71-163 intends to replace get_mode_perms
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

        poss_ranges[i] += int(num_combinations(len(sigma), i + 1) *
                              math.pow(num_perm, len(sigma) - 1 - i))

    i_range: int = 0
    for poss_range in poss_ranges:
        low_range: int = poss_ranges[i_range-1] if i_range != 0 else 0
        if low_range <= ind < poss_range:
            break
        i_range += 1

    offset: int = int((poss_ranges[i_range] - low_range) /
                      num_combinations(len(sigma), i_range + 1))

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


def clean(tape: List[str]) -> List[str]:
    ''' Removes blanks from both sides of the tape '''
    output_reverse: List[str] = []
    front: bool = True
    tape.reverse()

    for val in tape:
        if front:
            if val != '_':
                output_reverse.append(val)
                front = False
        else:
            output_reverse.append(val)

    output: List[str] = []
    front = True
    output_reverse.reverse()

    for val in output_reverse:
        if front:
            if val != '_':
                output.append(val)
                front = False
        else:
            output.append(val)

    return output


def check_inf_tape(trail: List[Tuple[Tuple[int, str], DeltaOut]],
                   current_mode: int, move: str) -> bool:
    ''' Check if trail leads to infinite '''
    return (trail != [] and trail[-1][0][1] == '_' and trail[-1][1][2] == move
            and trail[-1][0][0] == current_mode)


def run_tm(delta: Delta, tape: List[str], max_steps: int, modified: DeltaMod,
           debug_mode: bool = False) -> Tuple[List[str], DeltaMod, int, bool]:
    ''' Runs the TM and checks if it works '''
    global NUM_SKIPPED

    current_mode: int = 0
    tape_idx: int = 0
    steps: int = 0
    is_success: bool = False

    trail: List[Tuple[Tuple[int, str], DeltaOut]] = []

    while steps < max_steps:
        # Step the TM
        # READ
        read: str = str(tape[tape_idx])

        # INSTRUCTION
        instruction: DeltaOut = delta[(current_mode, read)]

        modified[(current_mode, read)] = True

        if debug_mode:
            print(f'Instruction: {instruction}')
            print(f'Tape: {tape}, mode: {current_mode}, ind: {tape_idx}, ' +
                  f'curr_val: {tape[tape_idx]}')
            print()

        if instruction == (-1, '-1', '-1'):
            is_success = True
            break

        # EXECUTE
        # change mode
        current_mode = instruction[0]

        # write
        tape[tape_idx] = instruction[1]

        # move
        if instruction[2] == "L":
            tape_idx = tape_idx - 1
        else:
            tape_idx = tape_idx + 1

        # expand tape
        if tape_idx >= len(tape):
            tape.append("_")

            if check_inf_tape(trail, current_mode, 'R'):
                is_success = False
                NUM_SKIPPED += 1
                break

        if tape_idx < 0:
            tape.insert(0, "_")
            tape_idx = 0

            if check_inf_tape(trail, current_mode, 'L'):
                is_success = False
                NUM_SKIPPED += 1
                break

        trail.append(((current_mode, read), instruction))

        # increment steps
        steps += 1

    output: List[str] = clean(tape)

    return output, modified, steps, is_success


def djikstra(delta: Delta, modes: int, sigma: List[str],
             q_in: int = 0) -> bool:
    ''' Djikstra's Algorithm finding Halt Transition from q_in '''
    dist: List[int] = [-1 for _ in range(modes)]
    dist[q_in] = 0
    count: int = 0

    while True:
        pot_dots: List[int] = list(filter(lambda x: x > -1, dist))
        if pot_dots == []:
            break

        mode: int = dist.index(min(pot_dots))

        for symb in sigma:
            if (-1, '-1', '-1') == delta[(mode, symb)]:
                return True

        neighbors: List[int] = []
        for symb in sigma:
            if delta[(mode, symb)][0] not in neighbors:
                neighbors.append(delta[(mode, symb)][0])

        new_dist: int = dist[mode] + 1
        for neighbor in neighbors:
            if ((dist[neighbor] == -1
                 and dist[neighbor] != -2) or new_dist < dist[neighbor]):
                dist[neighbor] = new_dist

        dist[mode] = -2
        count += 1
        if count >= 10:
            break

    return False


def check_delta(delta: Delta,
                data_train: List[List[str]],
                data_valid: List[List[str]],
                num_modes: int,
                sigma: List[str],
                max_steps: int = 1000,
                debug_mode: bool = False) -> Tuple[bool, DeltaMod]:
    ''' Checks whether delta function satisfies data_train '''
    global NUM_SKIPPED

    # if not djikstra(delta, num_modes, sigma):
    #     NUM_SKIPPED += 1
    #     return False

    modified: DeltaMod = {(mode, symb): False for symb in sigma
                          for mode in range(num_modes)}

    for ind, tape in enumerate(data_train):
        output, modified, steps, is_success = run_tm(
            delta, copy.copy(tape), max_steps, modified)

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


def calc_num_skipped(num_left: int, num_modes: int, num_sigma: int,
                     mode: int, lim_mode: int) -> int:
    ''' Find number of permutations skipped by modify procedure '''
    num_skip: int = num_left
    if mode != (num_modes - 1):
        num_skip *= (
            calc_num_lim_perms(num_modes - 1, num_sigma)
            if mode < lim_mode else
            calc_num_comm_perms(num_modes, num_sigma))
        num_skip *= int(math.pow(
            calc_num_comm_perms(num_modes, num_sigma) - mode - 1,
            num_modes - mode - 2))

    return num_skip


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
                        break

                num_skip = calc_num_skipped(
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
                num_skip = calc_num_skipped(
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


def test_valid_delta() -> None:
    ''' Tests valid delta '''
    # Init data
    data_train, data_valid, action_set, state_set = driver.init_dumb_add_data()

    # Generates the phi function and applies it to Data
    phi: Dict[str, str] = driver.gen_phi(action_set, state_set)
    driver.conv_data_to_tm(data_train, phi)
    driver.conv_data_to_tm(data_valid, phi)

    sigma: List[str] = list(phi.values())
    num_modes: int = 3

    delta_correct: Delta = {
        (0, '_'): (1, 's0', 'R'), (0, 's0'): (0, 's0', 'R'),
        (1, '_'): (2, '_', 'L'), (1, 's0'): (1, 's0', 'R'),
        (2, 's0'): (2, '_', 'R'), (2, '_'): (-1, '-1', '-1')
    }

    check_delta(delta_correct, data_train, data_valid,
                num_modes, sigma, debug_mode=True)


def main() -> None:
    ''' Main Function used for testing '''
    global NUM_PERMS

    # Init data
    data_train, data_valid, action_set, state_set = driver.init_dumb_add_data()

    # Generates the phi function and applies it to Data
    phi: Dict[str, str] = driver.gen_phi(action_set, state_set)
    driver.conv_data_to_tm(data_train, phi)
    driver.conv_data_to_tm(data_valid, phi)

    sigma: List[str] = list(phi.values())
    num_modes: int = 3

    NUM_PERMS = calc_num_perms(num_modes, len(sigma))

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
