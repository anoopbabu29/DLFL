''' Modules '''
from typing import List
import math
import sympy


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

    return int(num_lim * fact_fract)


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


def prime_fact(num: int) -> List[int]:
    ''' Finds the prime factorization of a number '''
    if sympy.isprime(num):
        return [num]

    for prime in sympy.sieve.primerange(0, num):
        if num % prime == 0:
            return [prime] + prime_fact(int(num/prime))

    return []


def gen_perms_wo_rep(inp_set: List[int]) -> List[List[int]]:
    ''' Generates all permutations w/o repetition'''
    if len(inp_set) == []:
        return []
    if len(inp_set) == 1:
        return [[inp_set[0]]]
    perms: List[List[int]] = []
    for i, curr_inp in enumerate(inp_set):
        perms += [[curr_inp] + prev_perm for prev_perm in
                  gen_perms_wo_rep(inp_set[:i] + inp_set[i+1:])]
    return perms
