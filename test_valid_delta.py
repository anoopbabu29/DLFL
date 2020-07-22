''' Modules '''
from typing import Dict, Tuple, List, Callable

# Local Imports
import setup
import lim_tm_search

# Defined Types
DeltaOut = Tuple[int, str, str]
Delta = Dict[Tuple[int, str], DeltaOut]


def test_valid_delta(
        obt_data: Callable[[], Tuple[List[List[str]], List[List[str]],
                                     List[str], List[str]]],
        delta_valid: Delta, num_modes: int, num_extra_sigma: int = 0) -> None:
    ''' Tests valid delta '''
    # Init data
    data_train, data_valid, action_set, state_set = obt_data()

    # Generates the phi function and applies it to Data
    phi: Dict[str, str] = setup.gen_phi(action_set, state_set)
    setup.conv_data_to_tm(data_train, phi)
    setup.conv_data_to_tm(data_valid, phi)

    sigma: List[str] = list(phi.values())
    for extra_sigma in range(num_extra_sigma):
        sigma.append(f'e{extra_sigma}')

    print(lim_tm_search.check_delta(delta_valid, data_train, data_valid,
                                    num_modes, sigma, debug_mode=True))


def test_valid_delta_add() -> None:
    ''' Tests valid delta for addition | 3 modes, 2 sigma'''
    num_modes: int = 3

    delta_valid: Delta = {
        (0, '_'): (1, 's0', 'R'), (0, 's0'): (0, 's0', 'R'),
        (1, '_'): (2, '_', 'L'), (1, 's0'): (1, 's0', 'R'),
        (2, 's0'): (2, '_', 'R'), (2, '_'): (-1, '-1', '-1')
    }

    test_valid_delta(setup.init_dumb_add_data, delta_valid, num_modes,
                     num_extra_sigma=0)


def test_valid_delta_mult() -> None:
    ''' Tests valid delta for multiplication | 10 modes, 3 sigma '''
    num_modes: int = 10

    delta_valid: Delta = {
        (0, '_'): (9, '_', 'L'), (0, 's0'): (1, 's1', 'R'),
        (0, 's1'): (6, '_', 'R'), (1, '_'): (2, '_', 'R'),
        (1, 's0'): (1, 's0', 'R'), (1, 's1'): (-1, '-1', '-1'),
        (2, '_'): (7, '_', 'L'), (2, 's0'): (3, 's1', 'R'),
        (2, 's1'): (-1, '-1', '-1'), (3, '_'): (4, '_', 'R'),
        (3, 's0'): (3, 's0', 'R'), (3, 's1'): (-1, '-1', '-1'),
        (4, '_'): (5, 's0', 'L'), (4, 's0'): (4, 's0', 'R'),
        (4, 's1'): (-1, '-1', '-1'), (5, '_'): (6, '_', 'L'),
        (5, 's0'): (5, 's0', 'L'), (5, 's1'): (-1, '-1', '-1'),
        (6, '_'): (-1, '-1', '-1'), (6, 's0'): (6, 's0', 'L'),
        (6, 's1'): (2, 's1', 'R'), (7, '_'): (8, '_', 'L'),
        (7, 's0'): (7, '_', 'R'), (7, 's1'): (7, 's0', 'L'),
        (8, '_'): (-1, '-1', '-1'), (8, 's0'): (8, 's0', 'L'),
        (8, 's1'): (0, 's1', 'R'), (9, '_'): (9, '_', 'R'),
        (9, 's0'): (7, '_', 'R'), (9, 's1'): (9, '_', 'L'),
    }

    test_valid_delta(setup.init_mult_data, delta_valid, num_modes,
                     num_extra_sigma=1)


def test_valid_delta_subt() -> None:
    ''' Tests valid delta for multiplication | 6 modes, 2 sigma '''
    num_modes: int = 6

    delta_valid: Delta = {
        (0, '_'): (1, '_', 'R'), (0, 's0'): (0, 's0', 'R'),
        (1, '_'): (2, '_', 'L'), (1, 's0'): (1, 's0', 'R'),
        (2, '_'): (-1, '-1', '-1'), (2, 's0'): (3, '_', 'L'),
        (3, '_'): (4, '_', 'L'), (3, 's0'): (3, 's0', 'L'),
        (4, '_'): (5, '_', 'R'), (4, 's0'): (4, 's0', 'L'),
        (5, '_'): (1, '_', 'R'), (5, 's0'): (0, '_', 'R'),
    }

    test_valid_delta(setup.init_subt_data, delta_valid, num_modes,
                     num_extra_sigma=0)


def main() -> None:
    ''' Main Method '''
    test_valid_delta_subt()


if __name__ == '__main__':
    main()
