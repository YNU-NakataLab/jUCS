include("condition.jl")

using Random

"Represents a classifier in the UCS system"
mutable struct Classifier
    id::Int64                  # Unique identifier for the classifier
    condition::Vector{UBR}     # Condition represented as a vector of Upper-Lower Bounds
    action::Int64              # Action associated with this classifier
    accuracy::Float64          # Accuracy of the classifier
    fitness::Float64           # Fitness of the classifier
    correct_track::Int64       # Number of correct predictions
    experience::Int64          # Number of times the classifier has been involved in matching
    time_stamp::Int64          # Last time the classifier was involved in the GA
    correct_set_size::Float64  # Average size of correct sets this classifier has been part of
    numerosity::Int64          # Number of micro-classifiers this macro-classifier represents
end

"Constructor function for creating a new Classifier"
function Classifier(parameters, env, state)
    condition = Vector{UBR}(undef, env.state_length)
    @simd for i in 1:env.state_length
        if rand() < parameters.P_hash || state[i] == "?"
            # Create a wildcard condition
            if rand() < 0.5
                condition[i] = UBR(0.0, 1.0)
            else
                condition[i] = UBR(1.0, 0.0)
            end
        else
            # Create a specific condition based on the state
            if rand() < 0.5
                condition[i] = UBR(min(max(state[i] - rand() * parameters.r0, 0.0), 1.0), max(min(state[i] + rand() * parameters.r0, 1.0), 0.0))
            else
                condition[i] = UBR(max(min(state[i] + rand() * parameters.r0, 1.0), 0.0), min(max(state[i] - rand() * parameters.r0, 0.0), 1.0))
            end
        end
    end
    clas::Classifier = Classifier(0, condition, rand(0:env.num_actions-1), 0, 0, 0, 0, 0, 0, 1)
    return clas
end

"Check if a classifier's condition matches a given state"
function does_match(condition::Vector{UBR}, state::Vector{Union{Float64, Int64, String}})::Bool
    for i in 1:length(state)
        if state[i] == "?"
            continue
        end
        if !(get_lower_bound(condition[i]) <= state[i] <= get_upper_bound(condition[i]))
            return false
        end
    end
    return true
end

"Apply crossover operation to two child classifiers"
function apply_crossover!(child_1::Classifier, child_2::Classifier)
    # Uniform Crossover
    @simd for i in 1:length(child_1.condition)
        if rand() < 0.5
            child_1.condition[i].p, child_2.condition[i].p = child_2.condition[i].p, child_1.condition[i].p
        end
        if rand() < 0.5
            child_1.condition[i].q, child_2.condition[i].q = child_2.condition[i].q, child_1.condition[i].q
        end
    end
end

"Apply mutation to a classifier"
function apply_mutation!(self::Classifier, m0::Float64, mu::Float64, num_actions::Int64)
    # Condition Mutation
    @simd for i in 1:length(self.condition)
        if rand() < mu
            self.condition[i].p += 2. * m0 * rand() - m0
            self.condition[i].p = min(max(0.0, self.condition[i].p), 1.0)

            self.condition[i].q += 2. * m0 * rand() - m0
            self.condition[i].q = min(max(0.0, self.condition[i].q), 1.0)
        end
    end

    # Action Mutation
    if rand() < mu
        used_actions = [self.action]
        available_actions = setdiff(Set(Vector(0:num_actions-1)), Set(used_actions))
        self.action = rand(available_actions)
    end

end

"Check if one classifier (self) is more general than another (spec)"
function is_more_general(self::Classifier, spec::Classifier)::Bool
    k::Int64 = 0
    for i in 1:length(self.condition)
        l_gen = get_lower_bound(self.condition[i])
        u_gen = get_upper_bound(self.condition[i])
        l_spec = get_lower_bound(spec.condition[i])
        u_spec = get_upper_bound(spec.condition[i])

        if !(l_gen <= l_spec && u_spec <= u_gen)
            return false
        end

        if l_spec == l_gen && u_gen == u_spec
            k += 1
        end
    end
    if k == length(self.condition)
        return false
    end
    return true
end

"Check if two classifiers have equal conditions"
function is_equal_condition(self::Classifier, other::Classifier)::Bool
    @simd for i in eachindex(self.condition)
        !is_equal(self.condition[i], other.condition[i]) && return false
    end
    return true
end

"Check if two classifiers are the same (based on their unique ID)"
is_equal_classifier(self::Classifier, other::Classifier)::Bool = self.id == other.id
