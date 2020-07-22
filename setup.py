''' Modules '''
from typing import List, Dict, Tuple


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


def init_dumb_add_data() -> Tuple[List[List[str]],
                                  List[List[str]],
                                  List[str], List[str]]:
    ''' Sets up training and resulting data for the dumb addition '''
    d_train: List[List[str]] = []
    d_valid: List[List[str]] = []
    action_set: List[str] = []
    state_set: List[str] = ['-']

    # Random data 1-10
    for i in range(1, 11):
        for j in range(1, 11):
            episode_train: List[str] = (['-' for _ in range(i)] + ['_'] +
                                        ['-' for _ in range(j)])
            episode_valid: List[str] = (['-' for _ in range(i)] +
                                        ['-' for _ in range(j)])
            d_train.append(episode_train)
            d_valid.append(episode_valid)

    return d_train, d_valid, action_set, state_set


def init_mult_data() -> Tuple[List[List[str]],
                              List[List[str]],
                              List[str], List[str]]:
    ''' Sets up training and resulting data for multiplication
        10 modes, 3 sigma, or 13 modes , 2 sigma'''
    d_train: List[List[str]] = []
    d_valid: List[List[str]] = []
    action_set: List[str] = []
    state_set: List[str] = ['-']

    start_ind: int = 1
    end_ind: int = 5

    # Random data 1-10
    for i in range(start_ind, end_ind):
        for j in range(start_ind, end_ind):
            episode_train: List[str] = (['-' for _ in range(i)] + ['_'] +
                                        ['-' for _ in range(j)])
            d_train.append(episode_train)

    for i in range(start_ind, end_ind):
        for j in range(start_ind, end_ind):
            episode_valid: List[str] = ['-' for _ in range(i)
                                        for _ in range(j)]
            d_valid.append(episode_valid)

    return d_train, d_valid, action_set, state_set


def init_subt_data() -> Tuple[List[List[str]],
                              List[List[str]],
                              List[str], List[str]]:
    ''' Sets up training and resulting data for multiplication
        10 modes, 3 sigma, or 13 modes , 2 sigma'''
    d_train: List[List[str]] = []
    d_valid: List[List[str]] = []
    action_set: List[str] = []
    state_set: List[str] = ['-']

    start_ind: int = 1
    end_ind: int = 5

    # Random data 1-10
    for i in range(start_ind, end_ind):
        for j in range(start_ind, end_ind):
            episode_train: List[str] = (['-' for _ in range(i)] + ['_'] +
                                        ['-' for _ in range(j)])
            d_train.append(episode_train)

    for i in range(start_ind, end_ind):
        for j in range(start_ind, end_ind):
            episode_valid: List[str] = ['-' for _ in range(i-j)]
            d_valid.append(episode_valid)

    return d_train, d_valid, action_set, state_set


def conv_data_to_tm(data: List[List[str]], phi: Dict[str, str]) -> None:
    ''' Converts the data into a form acceptable by TM'''
    for ep_ind, episode in enumerate(data):
        for inp_ind, inp in enumerate(episode):
            data[ep_ind][inp_ind] = phi[inp]


def main() -> None:
    ''' Main Function '''
    # Init data
    d_train, d_valid, action_set, state_set = init_mult_data()

    # Generates the phi function and applies it to Data
    phi: Dict[str, str] = gen_phi(action_set, state_set)
    conv_data_to_tm(d_train, phi)
    conv_data_to_tm(d_valid, phi)

    print(d_train)
    print(d_valid)
    print(phi)


if __name__ == '__main__':
    main()
