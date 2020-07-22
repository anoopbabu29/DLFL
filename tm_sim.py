''' Modules '''
from typing import Tuple, List, Dict

# Defined Types
DeltaOut = Tuple[int, str, str]
Delta = Dict[Tuple[int, str], DeltaOut]
DeltaMod = Dict[Tuple[int, str], bool]


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
                  f'steps: {steps}, curr_val: {tape[tape_idx]}')
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
                break

        if tape_idx < 0:
            tape.insert(0, "_")
            tape_idx = 0

            if check_inf_tape(trail, current_mode, 'L'):
                is_success = False
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
