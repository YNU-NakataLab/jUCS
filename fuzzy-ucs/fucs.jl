include("fclassifier.jl")

"Main structure for the Fuzzy-UCS"
mutable struct FUCS
    env::Environment                # The environment the system interacts with
    parameters::Parameters          # System parameters
    population::Vector{FClassifier} # Population of fuzzy classifiers
    time_stamp::Int64               # Current time step
    covering_occur_num::Int64       # Number of times covering occurred
    subsumption_occur_num::Int64    # Number of times subsumption occurred
    global_id::Int64                # Global ID counter for classifiers
end

"Constructor for Fuzzy-UCS"
function FUCS(env, parameters)
    return FUCS(env, parameters, [], 0, 0, 0, 0)
end

"Main experiment loop for Fuzzy-UCS"
function run_experiment(self::FUCS)
    # Get current state and correct answer from environment
    curr_state::Vector{Union{Float64, String}} = state(self.env)
    curr_answer::Int64 = answer(self.env, curr_state)
    
    # Generate match set based on current state
    match_set = generate_match_set(self, curr_state, curr_answer)
    
    # Generate correct set from match set
    correct_set = generate_correct_set(match_set, curr_answer)
    
    # Update parameters of classifiers in match and correct sets
    update_set!(self, match_set, correct_set, curr_answer)
    
    # Run GA on correct set
    run_ga!(self, correct_set, curr_answer)
    
    # Increment time step
    self.time_stamp += 1
end

"Generate the Pignistic Probability (BetP) set from the match set"
function generate_BetP_set(self::FUCS, match_set::Vector{FClassifier})::Vector{Float64}
    # Filter out classifiers with experience below the exploitation threshold
    experienced_match_set::Vector{FClassifier} = filter(clas -> clas.experience > self.parameters.theta_exploit, match_set)
    
    # Initialize mass array for all actions plus frame of discernment
    mass_all::Vector{Float64} = ones(self.env.num_actions + 1)
    
    # Initialize BetP set (Pignistic Probability) for each action
    BetP_set::Vector{Float64} = zeros(self.env.num_actions)
    previous_K::Float64 = 0.0
    
    if !isempty(experienced_match_set)
        # Set mass for each classifier in the experienced match set
        set_mass(experienced_match_set)
        
        # Combine masses of all classifiers
        for clas in experienced_match_set
            for i in 1:clas.numerosity
                mass_all, previous_K = combine_masses(mass_all, clas.mass_array)
                # If complete conflict, return zero probabilities (Yager's rule)
                if previous_K == 1.0 
                    return zeros(self.env.num_actions)
                end
            end
        end 
        
        # Calculate BetP (add distributed mass of frame of discernment)
        BetP_set = mass_all[1:end-1] .+ mass_all[end]/self.env.num_actions
        
        # Sanity check: BetP should sum to 1
        if round(sum(BetP_set), digits=3) != 1.0
            println("The sum of BetP set is not 1.")
            println(mass_all, BetP_set)
            exit(1)
        end
    end
    return BetP_set
end

"Combine mass functions using Dempster's rule of combination"
function combine_masses(mass_all::Vector{Float64}, mass_clas::Vector{Float64})::Tuple{Vector{Float64}, Float64}
    combined_masses::Vector{Float64} = zeros(length(mass_all))
    previous_K::Float64 = 0.0
    
    if sum(mass_all) == length(mass_all)
        # For the first classifier, just copy its mass
        combined_masses = deepcopy(mass_clas)
    else
        # Combine masses for each class and frame of discernment
        for i in 1:(length(combined_masses)-1)
            combined_masses[i] = mass_all[i] * mass_clas[end] + mass_all[end] * mass_clas[i] + mass_all[i] * mass_clas[i]
        end
        combined_masses[end] = mass_all[end] * mass_clas[end]
        
        # Sanity check: combined mass should be <= 1
        if round(sum(combined_masses), digits=3) > 1.0
            println("The sum of combined mass should be lower than 1.0.")
            println(mass_all, mass_clas, combined_masses)
            exit(1)
        end 

        # Calculate conflict and normalize
        previous_K = 1 - sum(combined_masses)
        combined_masses = combined_masses / sum(combined_masses)
    end
    
    return combined_masses, previous_K
end

"Set mass values for each classifier in the experienced match set"
function set_mass(experienced_match_set::Vector{FClassifier})
    @simd for clas in experienced_match_set
        # Calculate mass for each class
        clas.mass_array[1:end-1] = clas.weight_array * clas.matching_degree
        # Assign remaining mass to frame of discernment
        clas.mass_array[end] = max(1.0 - sum(clas.mass_array[1:end-1]), 0.0)
        
        # Sanity check: mass should sum to 1
        if round(sum(clas.mass_array), digits=3) != 1.0
            println("The sum of mass_array is not 1.")
            println(clas.weight_array, clas.matching_degree, clas.mass_array)
            exit(1)
        end
    end
end

"Generate the match set for a given state and answer"
function generate_match_set(self::FUCS, state::Vector{Union{Float64, String}}, answer::Int64, do_exploit=false)::Vector{FClassifier}
    # Calculate matching degrees for all classifiers
    set_matching_degree(self, state)
    
    # Filter classifiers with non-zero matching degree
    match_set::Vector{FClassifier} = filter(clas -> clas.matching_degree > 0.0, self.population)
    
    if !do_exploit
        # Check if covering is needed
        if do_covering(self, match_set, answer)
            # Covering not needed, do nothing
        else
            # Generate a covering classifier
            clas::FClassifier = generate_covering_classifier(self, state, answer)
            # Add the new classifier to the population
            push!(self.population, clas)
            # Remove excess classifiers if necessary
            delete_from_population!(self)
            # Add the new classifier to the match set
            push!(match_set, clas)
        end
    end
    return match_set
end

"Set the matching degree for all classifiers in the population"
function set_matching_degree(self::FUCS, state::Vector{Union{Float64, String}})
    @simd for clas in self.population
        clas.matching_degree = get_matching_degree(clas, state)
    end
end

"Calculate the matching degree of a classifier for a given state"
function get_matching_degree(clas::FClassifier, state::Vector{Union{Float64, String}})::Float64
    matching_degree::Float64 = 1.0
    @simd for i in 1:length(state)
        # Multiply membership values for each attribute
        matching_degree *= get_membership_value(clas.condition[i], state[i])
        # Early exit if matching degree becomes zero
        if matching_degree == 0.0
            return 0.0
        end
    end
    # Sanity check: matching degree should be between 0 and 1
    if !(0.0 <= matching_degree <= 1.0)
        error("Matching degree should be in [0,1] : mu_A=$(matching_degree)")
    end
    return matching_degree
end

"Check if covering is needed for the current state and answer"
function do_covering(self::FUCS, match_set::Vector{FClassifier}, answer::Int64)::Bool
    weight_array::Vector{Float64} = zeros(Float64, self.env.num_actions)
    @simd for clas in match_set
        # Accumulate matching degrees for each action
        weight_array[argmax(clas.weight_array)] += clas.matching_degree
        # If the correct action has sufficient weight, no covering is needed
        if weight_array[answer + 1] >= 1
            return true
        end
    end
    # Covering is needed
    return false
end

"Generate a covering classifier for the current state and answer"
function generate_covering_classifier(self::FUCS, state::Vector{Union{Float64, String}}, answer::Int64)::FClassifier
    # Create a new fuzzy classifier
    clas::FClassifier = FClassifier(self.parameters, self.env, state)
    # Set the weight for the correct action to 1
    clas.weight_array[answer + 1] = 1
    # Set the timestamp and ID
    clas.time_stamp = self.time_stamp
    clas.id = self.global_id
    self.global_id += 1
    # Increment the covering counter
    self.covering_occur_num += 1
    return clas
end

"Update the match set and correct set classifiers"
function update_set!(self::FUCS, match_set::Vector{FClassifier}, correct_set::Vector{FClassifier}, answer::Int64)
    # Calculate the total numerosity of the correct set
    set_numerosity::Int64 = mapreduce(clas -> clas.numerosity, +, correct_set)

    @simd for clas in match_set
        # Update experience
        clas.experience = clas.experience + clas.matching_degree

        # Update correct matching array and weights
        @simd for i in 1:self.env.num_actions
            if i == answer + 1
                clas.correct_matching_array[i] += clas.matching_degree
            end
            clas.weight_array[i] = clas.correct_matching_array[i] / clas.experience
        end
        
        # Sanity check: weights should sum to 1
        if round(sum(clas.weight_array), digits=3) != 1.0
            println("The sum of all the weights is not 1.")
            println(clas.weight_array, [clas.experience], [clas.matching_degree], clas.correct_matching_array)
            exit(1)
        end

        # Update fitness
        clas.fitness = 2 * maximum(clas.weight_array) - sum(clas.weight_array)
    end

    # Update correct set size for classifiers in the correct set
    @simd for clas in correct_set
        push!(clas.correct_set_size_array, set_numerosity)
        clas.correct_set_size = sum(clas.correct_set_size_array) / length(clas.correct_set_size_array)
    end

    # Perform correct set subsumption if enabled
    if self.parameters.do_correct_set_subsumption
        do_correct_set_subsumption!(self, correct_set)
    end
end

"Run the GA on the correct set"
function run_ga!(self::FUCS, correct_set::Vector{FClassifier}, answer::Int64)
    if isempty(correct_set)
        return
    end

    # Filter out classifiers with negative fitness
    positive_correct_set::Vector{FClassifier} = filter(clas -> clas.fitness >= 0, correct_set)
    if isempty(positive_correct_set)
        return
    end

    # Check if it's time to run GA based on the average timestamp of the correct set
    if self.time_stamp - mapreduce(clas -> clas.time_stamp * clas.numerosity, +, correct_set) / mapreduce(clas -> clas.numerosity, +, correct_set) > self.parameters.theta_GA
        # Update timestamps
        @simd for clas in correct_set
            clas.time_stamp = self.time_stamp
        end

        # Select parents and create offspring
        parent_1::FClassifier = select_offspring(self, positive_correct_set)
        parent_2::FClassifier = select_offspring(self, positive_correct_set)
        child_1::FClassifier = deepcopy(parent_1)
        child_2::FClassifier = deepcopy(parent_2)

        # Initialize children properties
        child_1.id, child_2.id = self.global_id, self.global_id + 1
        self.global_id += 2
        @simd for child in (child_1, child_2)
            child.numerosity = 1
            child.experience = 0
            child.correct_matching_array = zeros(Float64, self.env.num_actions)
            empty!(child.correct_set_size_array)
        end

        # Apply crossover with probability chi
        if rand() < self.parameters.chi
            apply_crossover!(child_1, child_2)
        end

        # Apply mutation and handle subsumption for each child
        @simd for child in (child_1, child_2)
            apply_mutation!(child, self.parameters.mu)
            if self.parameters.do_GA_subsumption
                if does_subsume(parent_1, child, self.parameters.theta_sub, self.parameters.F0)
                    self.subsumption_occur_num += 1
                    parent_1.numerosity += 1
                elseif does_subsume(parent_2, child, self.parameters.theta_sub, self.parameters.F0)
                    self.subsumption_occur_num += 1
                    parent_2.numerosity += 1
                else
                    insert_in_population!(self, child)
                end
            else
                insert_in_population!(self, child)
            end
            delete_from_population!(self)
        end
    end
end

"Select an offspring for reproduction"
function select_offspring(self::FUCS, positive_correct_set::Vector{FClassifier})::FClassifier
    if self.parameters.tau == 0.
        # Roulette-Wheel Selection
        fitness_sum = sum([(clas.fitness ^ self.parameters.nu) * clas.matching_degree for clas in positive_correct_set])
        choice_point = rand() * fitness_sum

        fitness_sum = 0.
        for clas in positive_correct_set
            fitness_sum += (clas.fitness ^ self.parameters.nu) * clas.matching_degree
            if fitness_sum > choice_point
                return clas
            end
        end
    else
        # Tournament Selection
        parent = nothing
        for clas in positive_correct_set
            if parent == nothing || (parent.fitness ^ self.parameters.nu) * parent.matching_degree / parent.numerosity < (clas.fitness ^ self.parameters.nu) * clas.matching_degree / clas.numerosity
                for i in 1:clas.numerosity
                    if rand() < self.parameters.tau
                        parent = clas
                        break
                    end
                end
            end
        end
        if parent == nothing
            parent = rand(positive_correct_set)
        end
        return parent
    end
end

"Insert a classifier into the population or increase numerosity if it already exists"
function insert_in_population!(self::FUCS, clas::FClassifier)
    for c in self.population
        if is_equal_condition(c, clas)
            c.numerosity += 1
            return
        end
    end
    push!(self.population, clas)
end

"Delete classifiers from the population to maintain the population size"
function delete_from_population!(self::FUCS)
    # Calculate the total numerosity of the population
    numerosity_sum::Float64 = mapreduce(clas -> clas.numerosity, +, self.population)
    # If the population size is within limits, do nothing
    if numerosity_sum <= self.parameters.N
        return
    end

    # Calculate average fitness of the population, using the power of nu
    average_fitness::Float64 = mapreduce(clas -> clas.fitness^self.parameters.nu, +, self.population) / numerosity_sum
    # Calculate the sum of deletion votes for all classifiers
    vote_sum::Float64 = mapreduce(clas -> deletion_vote(clas, average_fitness, self.parameters.theta_del, self.parameters.delta, self.parameters.nu), +, self.population)

    # Select a classifier for deletion using roulette wheel selection
    choice_point::Float64 = rand() * vote_sum
    vote_sum = 0.

    for clas in self.population
        vote_sum += deletion_vote(clas, average_fitness, self.parameters.theta_del, self.parameters.delta, self.parameters.nu)
        if vote_sum > choice_point
            # Decrease numerosity of the selected classifier
            clas.numerosity -= 1
            # If numerosity reaches zero, remove the classifier from the population
            if clas.numerosity == 0
                @views filter!(e -> e != clas, self.population)
            end
            return
        end
    end
end

"Perform subsumption in the correct set"
function do_correct_set_subsumption!(self::FUCS, correct_set::Vector{FClassifier})
    # Find the most general subsumer in the correct set
    cl::Any = nothing
    for c in correct_set
        if could_subsume(c, self.parameters.theta_sub, self.parameters.F0)
            if cl == nothing || is_more_general(c, cl)
                cl = c
            end
        end
    end
    # If a subsumer is found, subsume more specific classifiers
    if cl != nothing
        for c in correct_set
            if is_more_general(cl, c)
                self.subsumption_occur_num += 1
                cl.numerosity = cl.numerosity + c.numerosity
                # Remove subsumed classifier from correct set and population
                @views filter!(e->e!=c, correct_set)
                @views filter!(e->e!=c, self.population)
            end
        end
    end
end

"Check if a classifier could potentially subsume others"
could_subsume(clas::FClassifier, theta_sub::Int, F0::Float64)::Bool = clas.experience > theta_sub && clas.fitness > F0

"Check if one classifier subsumes another"
does_subsume(clas::FClassifier, tos::FClassifier, theta_sub::Int, F0::Float64)::Bool = could_subsume(clas, theta_sub, F0) && is_more_general(clas, tos)

"Calculate the deletion vote for a classifier"
function deletion_vote(clas::FClassifier, average_fitness::Float64, theta_del::Int, delta::Float64, nu::Float64)::Float64
    # Initial vote is based on the classifier's correct set size and numerosity
    vote::Float64 = clas.correct_set_size * clas.numerosity
    
    # If the classifier is experienced enough and its fitness is below average
    if clas.experience > theta_del && clas.fitness^nu < delta * average_fitness
        # Increase the vote, making it more likely to be deleted
        # The vote is scaled by the ratio of average fitness to the classifier's fitness
        vote *= average_fitness / max(clas.fitness^nu, 1e-12)
    end
    return vote
end

"Generate the correct set from the match set"
function generate_correct_set(match_set::Vector{FClassifier}, answer::Int)::Vector{FClassifier}
    # Filter the match set to include only classifiers whose highest weight corresponds to the correct answer
    return filter(clas -> argmax(clas.weight_array) == answer + 1, match_set)
end