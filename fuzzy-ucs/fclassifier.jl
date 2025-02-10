include("fcondition.jl")

using Random

"Represents a Fuzzy Classifier in the Fuzzy-UCS system"
mutable struct FClassifier
    id::Int64                               # Unique identifier for the classifier
    condition::Vector{CNF}                  # Condition represented as a vector of CNFs
    weight_array::Vector{Float64}           # Weights for each action
    fitness::Float64                        # Fitness of the classifier
    correct_matching_array::Vector{Float64} # Correct matches for each action
    experience::Float64                     # Number of times the classifier has been involved in matching
    time_stamp::Int64                       # Last time the classifier was involved in the GA
    correct_set_size::Float64               # Average size of correct sets this classifier has been part of
    correct_set_size_array::Vector{Float64} # History of correct set sizes
    numerosity::Int64                       # Number of micro-classifiers this macro-classifier represents
    matching_degree::Float64                # Degree of match with the current state
    mass_array::Vector{Float64}             # Mass array for Dempster-Shafer theory
end

"Constructor function for creating a new FClassifier"
function FClassifier(parameters, env, state)
    condition::Vector{CNF} = Vector{CNF}(undef, env.state_length)
    # Generate condition based on the state and parameters
    @simd for i in 1:env.state_length
        if rand() < parameters.P_hash || state[i] == "?"
            condition[i] = CNF(true, true, true, true, true)  # Wildcard condition
        else
            s = state[i]
            # Create a specific condition based on the state value
            if s == 0.0
                condition[i] = CNF(true, false, false, false, false)
            elseif s < 0.25
                condition[i] = CNF(true, s != 0.0, false, false, false)
            elseif s < 0.5
                condition[i] = CNF(false, true, s != 0.25, false, false)
            elseif s < 0.75
                condition[i] = CNF(false, false, true, s != 0.5, false)
            elseif s < 1.0
                condition[i] = CNF(false, false, false, true, s != 0.75)
            elseif s == 1.0
                condition[i] = CNF(false, false, false, false, true)
            else
                println("Invalid input value.")
                exit(0)
            end
        end
    end

    # Create and return a new FClassifier instance
    clas::FClassifier = FClassifier(0, condition, zeros(Float64, env.num_actions), 1, zeros(Float64, env.num_actions), 0, 0, 1, [], 1, 1, zeros(Float64, env.num_actions+1))
    return clas
end

"Apply crossover operation to two child classifiers"
function apply_crossover!(child_1::FClassifier, child_2::FClassifier)
    # Uniform Crossover
    @simd for i in 1:length(child_1.condition)
        # For each CNF in the condition, swap fields with 50% probability
        if rand() < 0.5
            child_1.condition[i].vs, child_2.condition[i].vs = child_2.condition[i].vs, child_1.condition[i].vs
        end
        if rand() < 0.5
            child_1.condition[i].s, child_2.condition[i].s = child_2.condition[i].s, child_1.condition[i].s
        end
        if rand() < 0.5
            child_1.condition[i].m, child_2.condition[i].m = child_2.condition[i].m, child_1.condition[i].m
        end
        if rand() < 0.5
            child_1.condition[i].l, child_2.condition[i].l = child_2.condition[i].l, child_1.condition[i].l
        end
        if rand() < 0.5
            child_1.condition[i].vl, child_2.condition[i].vl = child_2.condition[i].vl, child_1.condition[i].vl
        end
        # Ensure that no CNF becomes all false after crossover
        for child in (child_1, child_2)
            if are_all_false(child.condition[i])
                expansion!(child.condition[i])
            end
        end
    end
end

"Apply mutation to a fuzzy classifier's condition"
function apply_mutation!(self::FClassifier, mu::Float64)
    @simd for i in 1:length(self.condition)
        if rand() < mu  # Mutate with probability mu
            if are_all_true(self.condition[i])
                contraction!(self.condition[i])  # If all true, contract
            elseif is_only_one_true(self.condition[i])
                if rand() < 0.5
                    expansion!(self.condition[i])  # 50% chance to expand
                else
                    shift!(self.condition[i])  # 50% chance to shift
                end
            else
                r = rand()
                if r < 1/3
                    expansion!(self.condition[i])  # 1/3 chance to expand
                elseif r < 2/3
                    contraction!(self.condition[i])  # 1/3 chance to contract
                else
                    shift!(self.condition[i])  # 1/3 chance to shift
                end
            end
        end
    end
end

"Check if one fuzzy classifier is more general than another"
function is_more_general(self::FClassifier, spec::FClassifier)::Bool
    k::Int64 = 0
    for i in 1:length(self.condition)
        for fname in fieldnames(CNF)
            # If self has a false where spec has a true, self is not more general
            if getfield(self.condition[i], fname) == false && getfield(spec.condition[i], fname) == true
                return false
            end
        end

        # Count how many conditions are equal
        if is_equal(self.condition[i], spec.condition[i])
            k += 1
        end
    end
    # If all conditions are equal, self is not more general
    if k == length(self.condition)
        return false
    end
    return true
end

"Check if two fuzzy classifiers are the same (based on their ID)"
is_equal_classifier(self::FClassifier, other::FClassifier)::Bool = self.id == other.id

"Check if two fuzzy classifiers have equal conditions"
function is_equal_condition(self::FClassifier, other::FClassifier)::Bool
    @simd for i in eachindex(self.condition)
        !is_equal(self.condition[i], other.condition[i]) && return false
    end
    return true
end

"Check if two fuzzy classifiers are the same (based on their unique ID)"
is_equal_classifier(self::FClassifier, other::FClassifier)::Bool = self.id == other.id