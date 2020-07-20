''' Modules '''
from typing import Dict, Tuple, List, Any
import copy
import itertools
import time
import math

# Local Modules
import driver

# Globals
CURR_TIME: float = 0
NUM_PERMS: int = 0
COUNT: int = 0
NUM_SKIPPED: int = 0

# Defined types
DeltaOut = Tuple[int, str, str]
Delta = Dict[Tuple[int, str], DeltaOut]


def filter_lim_combs(comb: Dict[str, DeltaOut]) -> bool:
    ''' Method checking if list is valid '''
    halt: DeltaOut = (-1, '-1', '-1')
    vals: List[DeltaOut] = list(comb.values())

    return halt in vals


def gen_mode_perms(sigma: List[str],
                   num_modes: int) -> Tuple[List[Dict[str, DeltaOut]],
                                            List[Dict[str, DeltaOut]]]:
    ''' Generates the possible output combinations for a state '''
    halt = (-1, '-1', '-1')

    output_set: List[List[Any]] = [list(range(num_modes)), sigma, ['L', 'R']]
    output_combinations: List[DeltaOut] = list(itertools.product(*output_set))
    output_combinations.append(halt)

    mode_set: List[List[DeltaOut]] = [output_combinations for _ in sigma]

    mode_combinations: List[Dict[str, DeltaOut]] = [
        {sigma[i]: delta_out for i, delta_out in enumerate(list(comb))}
        for comb in list(itertools.product(*mode_set))]

    lim_combinations: List[Dict[str, DeltaOut]] = list(filter(
        filter_lim_combs, mode_combinations))

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

    output = []
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


def run_tm(delta: Delta, tape: List[str], max_steps: int,
           modified: Dict[Tuple[int, str], bool]) -> Tuple[List[str],
                                                           int, bool]:
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

        # print(f'Instruction: {instruction}')
        # print(f'Tape: {tape}, mode: {current_mode}, ind: {tape_idx}, '
        #       f'curr_val: {tape[tape_idx]}')
        # print()

        if instruction == (-1, '-1', '-1'):
            is_success = True
            break

        modified[(current_mode, read)] = True

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

            # if check_inf_tape(trail, current_mode, 'R'):
            #     is_success = False
            #     NUM_SKIPPED += 1
            #     break

        if tape_idx < 0:
            tape.insert(0, "_")
            tape_idx = 0

            # if check_inf_tape(trail, current_mode, 'L'):
            #     is_success = False
            #     NUM_SKIPPED += 1
            #     break

        trail.append(((current_mode, read), instruction))

        # increment steps
        steps += 1

    output: List[str] = clean(tape)

    return output, steps, is_success


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
                debug_mode: bool = False) -> bool:
    ''' Checks whether delta function satisfies data_train '''
    global NUM_SKIPPED

    if not djikstra(delta, num_modes, sigma):
        NUM_SKIPPED += 1

        return False

    for ind, tape in enumerate(data_train):
        output, steps, is_success = run_tm(delta, copy.copy(tape),
                                           max_steps, {})
        if debug_mode:
            print()
            print(f'Episode {ind+1}')
            print(f'Tape Input: {tape}')
            print(f'Target Output: {data_valid[ind]}')
            print(f'Tape Output: {output}')
            print(f'steps: {steps}')
            print()

        if not is_success or output != data_valid[ind]:
            return False

    return True


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
                         mode: int = 0) -> Delta:
    ''' Method that goes through all perms in factorial
        Complexity based on (((s*q)^(s)!) '''
    global CURR_TIME, COUNT

    if mode != total_modes:
        is_lim_mode: bool = mode == lim_mode
        iter_perms: List[Dict[str, DeltaOut]] = (lim_mode_perms if is_lim_mode
                                                 else mode_perms)

        # print(f'{mode_perms =}\n')
        # print(f'{lim_mode_perms =}\n')

        if iter_perms == {}:
            return {}

        for perm in iter_perms:
            # print(f'{mode}: {is_lim_mode}')
            delta.update({(mode, sym): perm[sym] for sym in perm})

            mode_perms_copy, lim_perms = copy_perms(mode_perms, lim_mode_perms,
                                                    perm, not is_lim_mode)

            delta_res = trav_factorial_perms(total_modes, sigma,
                                             data_train, data_valid,
                                             mode_perms_copy, lim_perms,
                                             lim_mode, delta, mode+1)
            if delta_res != {}:
                return delta_res

        return {}

    COUNT += 1

    if COUNT % 100 == 0:
        print(chr(27) + "[2J")
        print(f'{COUNT}/{NUM_PERMS} - {COUNT*100/NUM_PERMS:.2f}% | ' +
              f'Skipped {NUM_SKIPPED} | ' +
              f'{time.time() - CURR_TIME:.2f}s')

    return (delta
            if check_delta(delta, data_train, data_valid, total_modes, sigma)
            else {})


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
                           data_valid: List[List[str]]) -> Delta:
    ''' Method that calls the the traversal '''
    global CURR_TIME

    CURR_TIME = time.time()
    mode_perms, lim_mode_perms = gen_mode_perms(sigma, num_modes)
    print(f'{len(mode_perms) = }')
    # pprint_perms(mode_perms)
    print(f'{len(lim_mode_perms) = }')
    # pprint_perms(lim_mode_perms)
    print()

    for mode in range(num_modes):
        delta: Delta = trav_factorial_perms(num_modes, sigma,
                                            data_train, data_valid,
                                            mode_perms, lim_mode_perms,
                                            mode, {})
        if delta != {}:
            return delta

    return {}


def factorial(num: int) -> int:
    ''' Calculates the factorial of a number'''
    return num * factorial(num - 1) if num > 1 else 1


def calc_num_fact_perms(num_modes: int, num_sigma: int) -> int:
    ''' Calculates the number of permutations '''
    num_perm: int = 2 * num_modes * num_sigma + 1

    num_lim: int = num_modes * int((num_sigma *
                                    math.pow(num_perm, num_sigma - 1)) -
                                   (math.pow(num_sigma, num_sigma - 1) - 1))

    numerator: int = factorial(int(
        math.pow(num_perm, num_sigma) - 1))
    denominator: float = factorial(int(
        math.pow(num_perm, num_sigma) - num_modes))
    fact_fract: int = int(numerator / denominator)

    print(f'{num_lim = } {fact_fract = }')

    return int(num_lim * fact_fract)


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

    NUM_PERMS = calc_num_fact_perms(num_modes, len(sigma))

    res_delta: Delta = factorial_perms_method(num_modes, sigma,
                                              data_train, data_valid)

    print(chr(27) + "[2J")
    print(f'{COUNT}/{NUM_PERMS} - {COUNT*100/NUM_PERMS:.2f}% | ' +
          f'Skipped {NUM_SKIPPED} | ' +
          f'{time.time() - CURR_TIME:.2f}s')
    print(res_delta)


def test_valid_delta():
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
                num_modes, sigma, 1000, True)


if __name__ == '__main__':
    main()
