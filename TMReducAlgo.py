''' Modules '''
import time
from typing import List, Dict, Tuple, Any

SIGMA: List[str] = []

DeltaOut = Tuple[int, str, str]


def init_tm() -> Dict[str, Any]:
    ''' Initializes the Turing Machine '''
    turing_mach: Dict[str, Any] = {
        "Q": 1,
        "sigma": SIGMA,
        "delta": {(0, sigma): [] for sigma in SIGMA},
        "q0": 0,
    }  # < Q, ∑, ∂, q0 >

    return turing_mach


def gen_tm_from_episode(episode: List[str]) -> Dict[str, Any]:
    ''' Generates a stupid turing_mach from a single episode '''
    turing_mach: Dict[str, Any] = init_tm()

    turing_mach['Q'] = len(episode)
    turing_mach['delta'] = {(mode, sigma): [] for sigma in SIGMA
                            for mode in range(turing_mach['Q'])}

    current_mode: int = 0
    for symbol in episode:
        next_mode: int = (current_mode + 1
                          if current_mode + 1 != turing_mach['Q'] else -1)

        turing_mach['delta'][(current_mode, symbol)] = [
            (next_mode, symbol, 'R')]

        current_mode += 1

    return turing_mach


def adjust_dout(delta_res: List[DeltaOut],
                delta_res_start: int,
                is_start: bool = False) -> List[DeltaOut]:
    ''' Helper Function that adds num to DeltaOut state '''
    if delta_res == []:
        return []

    delta_res[0] = (delta_res[0][0], delta_res[0][1],
                    'e' if is_start else delta_res[0][2])

    new_mode: int = delta_res[0][0]
    new_mode += 0 if delta_res[0][0] == -1 else delta_res_start

    return [(new_mode, delta_res[0][1], delta_res[0][2])]


def comb_tms_to_ntm(orig_tm: Dict[str, Any],
                    gen_tm: Dict[str, Any]) -> Dict[str, Any]:
    ''' Combines two turing_machs into a single NTM '''
    ntm: Dict[str, Any] = init_tm()
    ntm['Q'] = orig_tm['Q'] + gen_tm['Q'] + 1

    # Initial state goes to both turing machines
    ntm['delta'] = {(0, sigma): (adjust_dout(
        orig_tm['delta'][(0, sigma)], 1, True) + adjust_dout(
            gen_tm['delta'][(0, sigma)], orig_tm['Q'], True))
                    for sigma in SIGMA}

    # Set all deltas to empty
    ntm['delta'].update({(mode, sigma): [] for sigma in SIGMA
                         for mode in range(1, ntm['Q'])})

    # Add original TM to the new NTM
    ntm['delta'].update({(key[0] + 1, key[1]):
                         adjust_dout(orig_tm['delta'][key], 1)
                         for key in orig_tm['delta']})

    # Add generated TM in NTM
    ntm['delta'].update({(key[0] + orig_tm['Q'] + 1, key[1]): adjust_dout(
        gen_tm['delta'][key], orig_tm['Q'] + 1)
                         for key in gen_tm['delta']})

    return ntm


def conv_ntm_to_dtm(ntm: Dict[str, Any]) -> Dict[str, Any]:
    ''' Converts a Nondeterministic TM to a Deterministic TM '''
    det_tm: Dict[str, Any] = init_tm()

    print(ntm)
    # TODO: convert NTM to DTM

    return det_tm


def minimize_tm(turing_mach: Dict[str, Any]) -> Dict[str, Any]:
    ''' Minimizes the turing_mach '''
    min_tm: Dict[str, Any] = init_tm()

    # FIXME: Below, replace with Minimize the turing_mach
    # Note: Probably not NP Complete
    min_tm = turing_mach

    return min_tm


def search_tm_reduce_algo(d_train: List[List[str]],
                          check_time: bool = True) -> Dict[str, Any]:
    ''' Function that finds the correct turing_mach '''
    turing_mach: Dict[str, Any] = init_tm()

    # num_extra_modes: int = 2
    # num_discrete_states: int = 2
    # num_discrete_actions: int = 3

    # assert num_discrete_states > 0
    # assert num_discrete_actions > 0
    # assert num_extra_modes >= 0

    assert d_train != []
    # assert len(D_train) == len(D_valid)

    if check_time:
        curr_time: float = time.time()

    for i, episode in enumerate(d_train):
        turing_mach = minimize_tm(
            conv_ntm_to_dtm(
                comb_tms_to_ntm(turing_mach, gen_tm_from_episode(episode))))
        print(f'{i+1}/{len(d_train)} | {time.time() - curr_time}s')

    print(f'Reduce Algo Time: {time.time() - curr_time}')

    return turing_mach


def gen_phi(action_set: List[str], state_set: List[str]) -> Dict[str, str]:
    ''' Gen mapping '''
    phi: Dict[str, str] = {}
    num_states: int = 0
    num_actions: int = 0

    phi['_'] = '_'

    for state in state_set:
        phi[state] = f's{num_states}'
        num_states += 1

    for action in action_set:
        phi[action] = f's{num_actions}'
        num_actions += 1

    return phi


def init_dumb_add_data() -> Tuple[List[List[str]], List[List[str]]]:
    ''' Sets up training and resulting data for the dumb addition '''
    d_train: List[List[str]] = []
    d_valid: List[List[str]] = []

    # Random data 1-10
    for i in range(1, 11):
        for j in range(1, 11):
            episode_train: List[str] = (['-' for _ in range(i)] + ['_'] +
                                        ['-' for _ in range(j)] + ['_'])
            episode_valid: List[str] = (['-' for _ in range(i)] +
                                        ['-' for _ in range(j)] + ['_'])

            d_train.append(episode_train)
            d_valid.append(episode_valid)

    return (d_train, d_valid)


def conv_data_to_tm(data: List[List[str]], phi: Dict[str, str]) -> None:
    ''' Converts the data into a form acceptable by turing_mach'''
    for ep_ind, episode in enumerate(data):
        for inp_ind, inp in enumerate(episode):
            data[ep_ind][inp_ind] = phi[inp]


def validate_tm(turing_mach: Dict[str, Any],
                d_train: List[List[str]],
                d_valid: List[List[str]]) -> bool:
    ''' Check if resulting Turing Machine works'''
    # TODO: Run turing machine to check whether valid

    return True


def search_tm(d_train: List[List[str]],
              d_valid: List[List[str]]) -> Dict[str, Any]:
    ''' Searches for turing_mach that fits D_train and optimizes D_valid '''
    # Choose which search method to test/use
    turing_mach: Dict[str, Any] = search_tm_reduce_algo(d_train)

    # Returns turng
    if validate_tm(turing_mach, d_train, d_valid):
        return turing_mach

    return {}


def main() -> None:
    ''' Main Function '''
    global SIGMA

    # Init action and state sets
    action_set: List[str] = []
    state_set: List[str] = ['-']

    # Init data
    d_train, d_valid = init_dumb_add_data()

    # Generates the phi function and applies it to Data
    phi: Dict[str, str] = gen_phi(action_set, state_set)
    conv_data_to_tm(d_train, phi)
    conv_data_to_tm(d_valid, phi)

    SIGMA = list(phi.values())

    # Search for turing_mach
    print(search_tm(d_train, d_valid))


if __name__ == '__main__':
    main()
