include("classifier.jl")

"Define the main UCS structure"
mutable struct UCS
    env::Environment                # The environment the UCS interacts with
    parameters::Parameters          # System parameters
    population::Vector{Classifier}  # Population of classifiers
    time_stamp::Int64               # Current time step
    covering_occur_num::Int64       # Number of times covering occurred
    subsumption_occur_num::Int64    # Number of times subsumption occurred
    global_id::Int64                # Global ID counter for classifiers
end

"Constructor for UCS"
function UCS(env, parameters)
    return UCS(env, parameters, [], 0, 0, 0, 0)
end

"Main experiment loop for UCS"
function run_experiment(self::UCS)
    curr_state::Vector{Union{Float64, String}} = state(self.env)  # Get current state from environment
    curr_answer::Int64 = answer(self.env, curr_state)  # Get correct answer for current state
    match_set = generate_match_set(self, curr_state, curr_answer)  # Generate match set
    correct_set = generate_correct_set(match_set, curr_answer)  # Generate correct set
    
    # Update correct track for classifiers in correct set
    @simd for clas in correct_set
        clas.correct_track += 1
    end
    
    update_set!(self, match_set, correct_set)  # Update classifier parameters
    run_ga!(self, correct_set)  # Run GA
    self.time_stamp += 1  # Increment time step
end

"Generate match set for given state and answer"
function generate_match_set(self::UCS, state::Vector{Union{Float64, String}}, answer::Int64, do_exploit=false)::Vector{Classifier}
    match_set::Vector{Classifier} = []
    if !do_exploit
        # Find matching classifiers
        match_set = filter(clas -> does_match(clas.condition, state), self.population)
        selected_actions::Set{Int64} = Set(clas.action for clas in match_set)
        
        # If correct action not in match set, generate covering classifier
        if !(answer in selected_actions)
            clas::Classifier = generate_covering_classifier(self, state, answer)
            push!(self.population, clas)
            delete_from_population!(self)
            push!(match_set, clas)
        end
    elseif do_exploit
        # Only find matching classifiers without covering
        match_set = filter(clas -> does_match(clas.condition, state), self.population)
    else
        error("Invalid argument for do_exploit.")
    end
    return match_set
end

"Generate a correct set from the match set based on the given answer"
function generate_correct_set(match_set::Vector{Classifier}, answer::Int64)::Vector{Classifier}
    n::Int64 = count(clas -> clas.action == answer, match_set)
    correct_set::Vector{Classifier} = similar(match_set, n)
    i::Int64 = 1
    @simd for clas in filter(clas -> clas.action == answer, match_set)
        correct_set[i] = clas
        i += 1
    end
    return correct_set
end

"Generate a covering classifier for a given state and answer"
function generate_covering_classifier(self::UCS, state::Vector{Union{Float64, String}}, answer::Int64)::Classifier
    # Create a new classifier based on the current state
    clas::Classifier = Classifier(self.parameters, self.env, state)
    
    # Set the classifier's action to the correct answer
    clas.action = answer
    
    # Set the classifier's time stamp to the current time
    clas.time_stamp = self.time_stamp
    
    # Assign a unique ID to the classifier
    clas.id = self.global_id
    self.global_id += 1
    
    # Increment the covering occurrence counter
    self.covering_occur_num += 1
    
    return clas
end

"Generate a fitness sum array for the match set"
function generate_fitness_sum_array(self::UCS, match_set::Vector{Classifier})::Vector{Union{Float64, String}}
    # Initialize an array to hold fitness sums for each action
    FSA::Vector{Float64} = zeros(Float64, self.env.num_actions)
    
    # Calculate the fitness sum for each action
    @simd for clas in match_set
        FSA[clas.action + 1] += clas.fitness * clas.numerosity
    end
    
    return FSA
end

"Update the match set and correct set classifiers"
function update_set!(self::UCS, match_set::Vector{Classifier}, correct_set::Vector{Classifier})
    # Calculate the total numerosity of the correct set
    set_numerosity::Float64 = mapreduce(clas -> clas.numerosity, +, correct_set)
    
    # Update experience, accuracy, and fitness for all classifiers in the match set
    match_set = map(clas -> begin
        clas.experience += 1
        clas.accuracy = clas.correct_track / clas.experience
        clas.fitness = clas.accuracy ^ self.parameters.nu
        clas
    end, match_set)

    # Update correct set size for classifiers in the correct set
    @simd for clas in correct_set
        if clas.experience < 1. / self.parameters.beta && self.parameters.use_MAM
            # Use MAM (Moyenne Adaptive Modifee) for young classifiers
            clas.correct_set_size += (set_numerosity - clas.correct_set_size) / clas.experience
        else
            # Use standard update for mature classifiers
            clas.correct_set_size += self.parameters.beta * (set_numerosity - clas.correct_set_size)
        end
    end

    # Perform correct set subsumption if enabled
    if self.parameters.do_correct_set_subsumption
        do_correct_set_subsumption!(self, correct_set)
    end
end

"Run the GA on the correct set"
function run_ga!(self::UCS, correct_set::Vector{Classifier})
    if isempty(correct_set)
        return
    end

    # Check if it's time to run the GA
    if self.time_stamp - mapreduce(clas -> clas.time_stamp * clas.numerosity, +, correct_set) / mapreduce(clas -> clas.numerosity, +, correct_set) > self.parameters.theta_GA
        # Update time stamps
        for clas in correct_set
            clas.time_stamp = self.time_stamp
        end

        # Select parents and create offspring
        parent_1::Classifier = select_offspring(self, correct_set)
        parent_2::Classifier = select_offspring(self, correct_set)
        child_1::Classifier = deepcopy(parent_1)
        child_2::Classifier = deepcopy(parent_2)
        
        # Initialize children properties
        child_1.id, child_2.id = self.global_id, self.global_id + 1
        self.global_id += 2
        for child in (child_1, child_2)
            child.fitness = 0.01
            child.numerosity = 1
            child.experience = 0
            child.correct_track = 0
        end

        # Apply crossover
        if rand() < self.parameters.chi
            apply_crossover!(child_1, child_2)
        end

        # Apply mutation and handle subsumption for each child
        @simd for child in (child_1, child_2)
            apply_mutation!(child, self.parameters.m0, self.parameters.mu, self.env.num_actions)
            if self.parameters.do_GA_subsumption
                if does_subsume(parent_1, child, self.parameters.theta_sub, self.parameters.acc0)
                    self.subsumption_occur_num += 1
                    parent_1.numerosity += 1
                elseif does_subsume(parent_2, child, self.parameters.theta_sub, self.parameters.acc0)
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
function select_offspring(self::UCS, correct_set::Vector{Classifier})::Classifier
    if self.parameters.tau == 0.
        # Roulette-Wheel Selection
        fitness_sum::Float64 = sum([clas.fitness for clas in correct_set])
        choice_point::Float64 = rand() * fitness_sum

        fitness_sum = 0.
        for clas in correct_set
            fitness_sum = fitness_sum + clas.fitness
            if fitness_sum > choice_point
                return clas
            end
        end
    else
        # Tournament Selection
        parent::Any = nothing
        for clas in correct_set
            if parent == nothing || parent.fitness / parent.numerosity < clas.fitness / clas.numerosity
                for i in 1:clas.numerosity
                    if rand() < self.parameters.tau
                        parent = clas
                        break
                    end
                end
            end
        end
        if parent == nothing
            parent = rand(correct_set)
        end
        return parent
    end
end

"Insert a classifier into the population or increase numerosity if it already exists"
function insert_in_population!(self::UCS, clas::Classifier)
    # Check if a classifier with the same condition and action already exists
    for c in self.population
        if is_equal_condition(c, clas) && c.action == clas.action
            # If found, increase its numerosity and return
            c.numerosity += 1
            return
        end
    end
    # If not found, add the new classifier to the population
    push!(self.population, clas)
end

"Delete classifiers from the population to maintain the population size"
function delete_from_population!(self::UCS)
    # Calculate the total numerosity of the population
    numerosity_sum::Float64 = mapreduce(clas -> clas.numerosity, +, self.population)
    # If the population size is within limits, do nothing
    if numerosity_sum <= self.parameters.N
        return
    end

    # Calculate average fitness of the population
    average_fitness::Float64 = mapreduce(clas -> clas.fitness, +, self.population) / numerosity_sum
    # Calculate the sum of deletion votes for all classifiers
    vote_sum::Float64 = mapreduce(clas -> deletion_vote(clas, average_fitness, self.parameters.theta_del, self.parameters.delta), +, self.population)

    # Select a classifier for deletion using roulette wheel selection
    choice_point::Float64 = rand() * vote_sum
    vote_sum = 0.

    for clas in self.population
        vote_sum += deletion_vote(clas, average_fitness, self.parameters.theta_del, self.parameters.delta)
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
function do_correct_set_subsumption!(self::UCS, correct_set::Vector{Classifier})
    # Find the most general subsumer in the correct set
    cl::Any = nothing
    for c in correct_set
        if could_subsume(c, self.parameters.theta_sub, self.parameters.acc0)
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

"Determine if a classifier is eligible to subsume other classifiers"
could_subsume(self::Classifier, theta_sub::Int64, acc0::Float64)::Bool = self.experience > theta_sub && self.accuracy > acc0

"Check if one classifier can subsume another"
does_subsume(self::Classifier, tos::Classifier, theta_sub::Int64, acc0::Float64)::Bool = self.action == tos.action && could_subsume(self, theta_sub, acc0) && is_more_general(self, tos)

"Calculate a vote for deleting a classifier from the population"
function deletion_vote(self::Classifier, average_fitness::Float64, theta_del::Int64, delta::Float64)::Float64
    vote::Float64 = self.correct_set_size * self.numerosity
    if self.experience > theta_del && self.fitness / self.numerosity < delta * average_fitness
        vote *= average_fitness / max(self.fitness, 1e-12)
    end
    return vote
end