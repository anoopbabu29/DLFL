''' Modules '''
import itertools
import copy
import time
from typing import List, Dict, Tuple, Callable, Iterable, Any
import random
from os import system

num_skipped: List[int] = [0, 0, 0]
curr_time: float
time_run: int = 10
is_finite_time: bool = False

random.seed(3)


def algorithm(D: List[List[Tuple[str, str, str]]],
              state_set: List[str],
              action_set: List[str]):
    global is_TM_success, num_skipped, curr_time, time_run, is_finite_time

    TM: Dict[str, Any] = {
        "Q": None,
        "sigma": None,
        "delta": None,
        "q0": 0,
        "q_accept": 1,
        "q_reject": 2
    }  # < Q, ∑, ∂, q0, qaccept, qreject >

    # Define Q = {...} finite
    # Define ∑ = { all states, all actions, 1, 0, _ } finite
    # Define q0, qaccept, qreject Semantically means start mode, can do, and can’t do

    num_extra_modes: int = 1
    num_discrete_states: int = 2
    num_discrete_actions: int = 3

    assert num_extra_modes >= 0
    assert num_discrete_states > 0
    assert num_discrete_actions > 0

    discrete_states: List[str] = [f's{i}' for i in range(num_discrete_states)]
    discrete_actions: List[str] = [f'a{i}'
                                   for i in range(num_discrete_actions)]

    TM["Q"] = [0, 1, 2] + [i + 3 for i in range(num_extra_modes)]
    TM["sigma"] = discrete_states + discrete_actions + ["b"]

    print(f'TM: {TM}')

    # // Remainder of program is to find: ∂ = ?  (Q x ∑ -> Q x ∑ x {left, right})

    # Collect data D = { (s_i, a_i, s’_i)... | for i in N } # *Taken in as argument*

    # ??? for (datapoint) d in D “train model”: F_θ(ø(s), a) => ø(s’) and  I_θ(ø(s), ø(s’)) => a

    if True:

        F: Dict[Tuple[str, str], Tuple[int, str, str]] = {
            i: None for i in itertools.product(state_set, action_set)
        }
        I: Dict[Tuple[str, str], Tuple[int, str, str]] = {
            i: None for i in itertools.product(state_set, state_set)
        }

        for episode in D:
            for d in episode:
                F[(d[0], d[1])] = d[2]
                I[(d[0], d[2])] = d[1]

        print(F)
        print(I)
    

    # Enumerate all possible programs ∂ | ∂ := (Q x ∑ -> Q x ∑ x {left, right})
    ## N_∂ = (|Q| x (|S| + |A| + 3) x 2) ^ (|Q| x (|S| + |A| + 3))

    input_set: List[List[Any]] = [list(filter(
        lambda x: x != 1 and x != 2, TM["Q"])), TM["sigma"]]
    input_set = [s for s in input_set if (None == random.shuffle(s))]  # Random shuffle
    input_combinations: List[Tuple[int, str]] = list(itertools.product(
        *input_set))

    output_set: List[List[Any]] = [TM["Q"], TM["sigma"], ["L", "R"]]
    output_combinations: List[Tuple[int, str, str]] = list(itertools.product(
        *output_set))

    # print("input")
    # print(input_combinations)
    # print()
    # print('output')
    # print(output_combinations)

    num_programs: int = len(output_combinations) ** len(input_combinations)

    print(f'# of programs: {num_programs}')
    print()
    
    
    #assert(num_programs < 1e12)

    # TODO not random order of programs
    program_set: List[List[Tuple[int, str, str]]] = (
        [output_combinations for _ in input_combinations
         if random.shuffle(output_combinations) is None])


    # TODO make this generator smarter
    program_combinations: Iterable[Tuple[Any, ...]] = itertools.product(
        *program_set)
    program_combinations = filter(is_valid_combination(input_combinations,
                                  TM['Q'], TM['sigma']), program_combinations)

    count: int = 0
    
    prev_modified: Dict[Tuple[int, str], bool] = {i: False
                                                  for i in input_combinations}
    prev_delta: Dict[Tuple[int, str], Tuple[int, str, str]] = {
        i: None for i in input_combinations}
    
    curr_time = time.time()
    for program in program_combinations:
        if is_finite_time and time.time() - curr_time >= time_run:
            print(f'{time_run}s passed')
            quit()

        # print('Program')
        # print(program)
        # quit()

        print(f'{count + sum(num_skipped)}/{num_programs}')
        print(f'num_skipped: {num_skipped}, {sum(num_skipped)}')
        print("#"*30)

        delta: Dict[Tuple[int, str], Tuple[int, str, str]] = {
            i: None for i in input_combinations}
        for ind, x in enumerate(input_combinations):
            delta[x] = program[ind]
            # print("(",q_in,",",symbol_in,") -> (",q_out,",",symbol_out,",",direction,")")

        true_mod_prev: Dict[Tuple[int, str], bool] = {
            i: x for i, x in prev_modified.items() if x}

        # print(true_mod_prev)
        # print(prev_delta)
        is_diff: bool = False
        for i in true_mod_prev:
            if delta[i] != prev_delta[i]:
                is_diff = True
                break

        if not is_diff and true_mod_prev != {}:
            num_skipped[1] += 1
            continue

        prev_delta = copy.deepcopy(delta)
        for i in prev_modified:
            prev_modified[i] = False
        

        #print("delta:", delta)

        # For each ∂_i find/check consistency on D
        ## Map actual states and actions -> ∑’s states and actions (semantic grounding)
        # TODO Define a mapping random
        phi_state: Dict[str, str] = {".": "s0", "-": "s1"}
        phi_action: Dict[str, str] = {"_": "a0", "p": "a1", "q": "a2"}

        if check_TM(TM, delta, prev_modified, F, D, phi_state, phi_action):
            break
        #system('clear')

        ## This means running TM on s_i and s_f should output the path [s_i, a_i, …. a_f-1, s_f] or terminate with q_reject after max time T
        ## Where the path should exist in D if trajectory s_i to s_f exists in D (or) F_θ(ø(s), a) = ø(s’) for all states in path
        ## ??? I want to say this allows for generalization if we restrict the size of |Q|, |∑|, and |T|

        # Run TM on cs_i and s_f for max steps T
    return TM

    # Notes:
    ## Complexity:
    ### O(T x B x (|Q| x (|S| + |A| + 3) x 2) ^ (|Q| x (|S| + |A| + 3))):  where |B| <= |D|

    # IDEA:
    ## Anoop: Can add a heuristic to reduce the search for ∂
    ## Check consistency during and not after validating all actions with the model as they are written not after the tape halts. Therefore if not getting to the final state from initial state in the model within max Time T is a failure.
    ## Have function over the tape that enforces structure of compute (either tape should or  shouldn’t look like this) OR forward model and inverse model function on tape  (with different tape)

    # Questions:
    ## How many states are enough?
    ## Is a finite amount of states able to replicate any universal TM?

def check_TM(TM, delta, prev_modified, F,
             Data: List[List[Tuple[str, str, str]]],
             phi_state: Dict[str, str],
             phi_action: Dict[str, str]) -> bool:

    for ind, episode in enumerate(Data):
        T = 1000

        sym_ep: List[Tuple[str, str, str]] = episode_state_to_sym(
            episode, phi_state, phi_action)
        tape: List[str] = init_tape(sym_ep, False)
        final_tape: List[str] = init_tape(sym_ep, True)


        TM["delta"] = copy.deepcopy(delta)
        print(f'Episode {ind+1}')
        print(f'Tape Input: {tape}')
        output, steps, is_reject = run_TM(TM, tape, T, prev_modified)
        print(f'Target Output: {final_tape}')
        print(f'Tape Output: {output}')
        print(f'steps: {steps}')

        if is_reject:
            print('Went to reject state')
            return False

        # Validate tape output with forward model
        valid = validate(output, final_tape, F)

        print(valid)
        if valid:
            print("MAGICAL")
            print(TM["delta"])
            print()
        else:
            # TODO Change TM to fit data????
            return False
    system('clear')

    return True


def episode_state_to_sym(episode: List[Tuple[str, str, str]],
                         phi_state: Dict[str, str],
                         phi_action: Dict[str, str]) -> List[Tuple[str, str, str]]:
    sym_ep: List[Tuple[str, str, str]] = []

    for part in episode:
        sym_ep.append((phi_state[part[0]], phi_action[part[1]], phi_state[part[2]]))

    return sym_ep


def init_tape(episode: List[Tuple[str, str, str]], is_final: bool) -> List[str]:
    tape: List[str] = []

    if episode != []:
        tape.append(episode[0][0])

    for part in episode:
        tape.append(part[1])
        if not is_final and part[1] == 'a2':
            break
        tape.append(part[2])

    return tape


# !!! IF ERROR THIS IS PROABLY MESSED UP !!! #
def validate(output, final_tape, F):
    valid = False

    if len(output) == len(final_tape):
        if not (False in
                [output[i] == final_tape[i] for i in range(len(final_tape))]):
            valid = True

    return valid


def clean(tape: List[str]):
    output_reverse = []
    front = True
    tape.reverse()

    for val in tape:
        if front:
            if val != 'b':
                output_reverse.append(val)
                front = False
        else:
            output_reverse.append(val)

    output = []
    front = True
    output_reverse.reverse()

    for val in output_reverse:
        if front:
            if val != 'b':
                output.append(val)
                front = False
        else:
            output.append(val)

    return output


def run_TM(TM, tape: List[str], T, modified):
    global num_skipped
    current_mode = TM["q0"]
    tape_idx = 0
    steps = 0
    is_reject: bool = False

    trail: List[Tuple[Tuple[int, str], Tuple[int, str, str]]] = []

    while current_mode not in [TM["q_accept"], TM["q_reject"]] and steps < T:
        # Step the TM
        # READ
        read = str(tape[tape_idx])

        # INSTRUCTION
        instruction = TM["delta"][(current_mode, read)]
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
            tape.append("b")

            if trail != [] and trail[-1][0][1] == 'b' and trail[-1][1][2] == 'R' and trail[-1][0][0] == current_mode:
                num_skipped[2] += 1
                is_reject = True
                print('Infinite Write on tape')
                break
        
        if tape_idx < 0:
            tape.insert(0, "b")

            if trail != [] and trail[-1][0][1] == 'b' and trail[-1][1][2] == 'L' and trail[-1][0][0] == current_mode:
                num_skipped[2] += 1
                is_reject = True
                print('Infinite Write on tape')
                break
        
        trail.append(((current_mode, read), instruction))

        # increment steps
        steps = steps + 1


    # TODO clean tape
    output = clean(tape)

    is_reject = is_reject or current_mode == TM["q_reject"]

    return output, steps, is_reject



def djikstra(delta, modes: List[int], sigma: List[str],
             q_in: int = 0, q_accept: int = 1, q_reject: int = 2) -> bool:
    ''' Djikstra's Algorithm finding q_accept from q_in '''
    dist: List[int] = [-1 for _ in modes]
    dist[q_in] = 0
    dist[q_reject] = -2
    # num_visited: int = 1
    # visited: List[int] = [False] * len(modes)
    count = 0

    while True:
        pot_dots = list(filter(lambda x: x > -1, dist))
        if pot_dots == []:
            break

        mode: int = dist.index(min(pot_dots))

        if mode == q_accept:
            return True

        neighbors: List[int] = []
        for s in sigma:
            if delta[(mode, s)][0] not in neighbors:
                neighbors.append(delta[(mode, s)][0])

        new_dist: int = dist[mode] + 1
        for neighbor in neighbors:
            if ((dist[neighbor] == -1
                 and dist[neighbor] != -2) or new_dist < dist[neighbor]):
                dist[neighbor] = new_dist

        dist[mode] = -2
        count += 1
        if count >= 10:
            break

        # print(mode)
        # print(num_visited)
        # print(dist)

    return False


def is_valid_combination(input_combinations: List[Tuple[int, str]],
                         modes: List[int],
                         sigma: List[str]) -> Callable[
                             [List[Tuple[int, str, str]]], bool]:
    delta: Dict[Tuple[int, str], Tuple[int, str, str]] = {
        i: None for i in input_combinations}

    def filter_combination_func(val: List[Tuple[int, str, str]]) -> bool:
        global num_skipped, curr_time, time_run, is_finite_time

        if is_finite_time and time.time() - curr_time >= time_run:
            print(f'{time_run}s passed')
            quit()

        for i, x in enumerate(input_combinations):
            delta[x] = val[i]

        return_val: bool = djikstra(delta, modes, sigma)
        if not return_val:
            num_skipped[0] += 1

        print(f'Skipped: {num_skipped}, {sum(num_skipped)}')
        system('clear')

        return return_val

    return filter_combination_func


if __name__ == '__main__':
    # D = [[("s0","a0","s0"),("s0","a1","s1"),("s1","a1","s0")],
    # [("s0","a1","s1"),("s1","a0","s0"),("s0","a1","s1")],
    # [("s0","a0","s0"),("s0","a0","s0"),("s0","a1","s1")]]

    action_set: List[str] = ["_","p","q"]
    state_set: List[str] = ["-"]

    D: List[List[Tuple[str, str, str]]] = []
    # Random data 1-10
    for i in range(1, 11):
        for j in range(1, 11):
            episode = " _ ".join(["-" + random.choice([" _ .", ""]) for _ in range(i)]) + " p " + " _ ".join(["-" + random.choice([" _ .",""]) for _ in range(j)]) + " q " + " _ ".join(["-" for _ in range(i+j)])
            D.append([(episode.split()[i], episode.split()[i+1], episode.split()[i+2]) for i in range(0,len(episode.split())-2,2)])

    algorithm(D, state_set, action_set)
