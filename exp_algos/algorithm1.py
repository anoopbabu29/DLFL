import itertools
import copy

def algorithm(D):

    TM  = {"Q": None, "sigma": None, "delta": None, "q0": 0, "q_accept": 1, "q_reject": 2} # < Q, ∑, ∂, q0, qaccept, qreject >

    # Define Q = {...} finite
    # Define ∑ = { all states, all actions, 1, 0, _ } finite
    # Define q0, qaccept, qreject Semantically means start mode, can do, and can’t do

    num_extra_modes = 0
    num_discrete_states = 2
    num_discrete_actions = 2

    assert(num_extra_modes >= 0)
    assert(num_discrete_states > 0)
    assert(num_discrete_actions > 0)

    discrete_states = ["s" + str(i) for i in range(num_discrete_states)]
    discrete_actions = ["a" + str(i) for i in range(num_discrete_actions)]

    TM["Q"] = [0,1,2] + [i + 3 for i in range(num_extra_modes)]
    TM["sigma"] = discrete_states + discrete_actions + ["1","0","b"]

    print("TM:",TM)

    # // Remainder of program is to find: ∂ = ?  (Q x ∑ -> Q x ∑ x {left, right})

    # Collect data D = { (s_i, a_i, s’_i)... | for i in N } # *Taken in as argument*

    # ??? for (datapoint) d in D “train model”: F_θ(ø(s), a) => ø(s’) and  I_θ(ø(s), ø(s’)) => a

    if True:

        F = { i : None for i in itertools.product(discrete_states, discrete_actions)}
        I = { i : None for i in itertools.product(discrete_states, discrete_states)}

        for episode in D:
            for d in episode:
                F[(d[0],d[1])] = d[2]
                I[(d[0],d[2])] = d[1]

        print(F)
        print(I)

    # Enumerate all possible programs ∂ | ∂ := (Q x ∑ -> Q x ∑ x {left, right})
    ## N_∂ = (|Q| x (|S| + |A| + 3) x 2) ^ (|Q| x (|S| + |A| + 3))

    input_set = [TM["Q"], TM["sigma"]]
    input_combinations = list(itertools.product(*input_set))

    output_set = [TM["Q"], TM["sigma"], ["L","R"]]
    output_combinations = list(itertools.product(*output_set))

    num_programs = (len(output_combinations)) ** (len(input_combinations))

    print("# of programs:", num_programs)
    print()
    
    #assert(num_programs < 1e12)

    program_set = [output_combinations for _ in input_combinations]

    # TODO make this generator smarter
    program_combinations = itertools.product(*program_set)

    count = 0

    for program in program_combinations:
        count = count + 1

        print(count,"/",num_programs)

        print("#"*30)
        delta = { i : None for i in itertools.product(TM["Q"], TM["sigma"])}

        for i, x in enumerate(input_combinations):
                delta[x] = program[i]
                #print("(",q_in,",",symbol_in,") -> (",q_out,",",symbol_out,",",direction,")")

        print("delta:", delta)

        # For each ∂_i check consistency on D
        ## This means running TM on s_i and s_f should output the path [s_i, a_i, …. a_f-1, s_f] or terminate with q_reject after max time T
        ## Where the path should exist in D if trajectory s_i to s_f exists in D (or) F_θ(ø(s), a) = ø(s’) for all states in path
        ## ??? I want to say this allows for generalization if we restrict the size of |Q|, |∑|, and |T|

        # Run TM on s_i and s_f for max steps T
        T = 1000
        s_i = "s0"
        s_f = "s1"
        tape = [s_i, s_f]

        TM["delta"] = copy.deepcopy(delta)
        output, steps = run_TM(TM, tape, T)
        print("Tape Output:", output)
        print("steps:",steps)

        #if steps == T:
            #quit()

        # Validate tape output with forward model
        valid = validate(output, s_i, s_f, discrete_states, discrete_actions, F)

        print(valid)
        if valid:
            print("MAGICAL")
            print(TM["delta"])
            quit()

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

# !!! IF ERROR THIS IS PROABLY MESSED UP !!! #
def validate(output, s_i, s_f, discrete_states, discrete_actions, F):
    valid = False

    if output[0] == s_i and output[-1] == s_f:
        for i in range(0,len(output) - 2,2):
            valid = True
            print("THIS HAPPENED")
            print(output[i])
            print(output[i+1])


            if output[i] not in discrete_states:
                valid = False
                break

            if output[i+1] not in discrete_actions:
                valid = False
                break

            if F[(output[i],output[i+1])] != output[i+2]:
                valid = False
                break

    return valid

def clean(tape):
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

def run_TM(TM, tape, T):
    current_mode = TM["q0"]
    tape_idx = 0
    steps = 0

    while current_mode not in [TM["q_accept"], TM["q_reject"]] and steps < T:
        # Step the TM
        # READ
        read = str(tape[tape_idx])

        # INSTRUCTION
        instruction = TM["delta"][(current_mode, read)]

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
        
        if tape_idx < 0:
            tape.insert(0,"b")

        # increment steps
        steps = steps + 1

    # TODO clean tape
    output = clean(tape)

    return output, steps

D = [[("s0","a0","s0"),("s0","a1","s1"),("s1","a1","s0")],[("s0","a1","s1"),("s1","a0","s0"),("s0","a1","s1")],[("s0","a0","s0"),("s0","a0","s0"),("s0","a1","s1")]]
algorithm(D)