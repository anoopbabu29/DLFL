''' Modules '''
import random
import time
from typing import List, Dict, Tuple, Any

''' Algorithms '''
import TMReducAlgo


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


def init_dumb_add_data() -> Tuple[List[List[str]], List[List[int]]]:
    ''' Sets up training and resulting data for the dumb addition '''
    D_train: List[List[str]] = []
    D_valid: List[List[str]] = []

    # Random data 1-10
    for i in range(1, 11):
        for j in range(1, 11):
            episode_train: List[str] = ['-' for _ in range(i)] + ['_'] + ['-' for _ in range(j)]
            episode_valid: List[str] = ['-' for _ in range(i)] + ['-' for _ in range(j)]
            D_train.append(episode_train)
            D_valid.append(episode_valid)
    
    return D_train, D_valid


def conv_data_to_TM(data: List[List[str]], phi: Dict[str, str]) -> None:
    ''' Converts the data into a form acceptable by TM'''
    for ep_ind, episode in enumerate(data):
        for inp_ind, inp in enumerate(episode):
            data[ep_ind][inp_ind] = phi[inp]
    

def search_TM(D_train: List[List[str]], 
              D_valid: List[List[str]]) -> Dict[str, Any]:
    ''' Searches for TM that fits D_train and optimizes D_valid '''
    # Choose which search method to test/use
    return TMReducAlgo.search_TM_reduce_algo(D_train)


def main() -> None:
    ''' Main Function '''
    # Init action and state sets
    action_set: List[str] = []
    state_set: List[str] = ['-']

    # Init data
    D_train, D_valid = init_dumb_add_data()
    
    # Generates the phi function and applies it to Data
    phi: Dict[str, str] = gen_phi(action_set, state_set)
    conv_data_to_TM(D_train)
    conv_data_to_TM(D_valid)

    # Search for TM
    print(search_TM(D_train, D_valid))


if __name__ == '__main__':
    main()
