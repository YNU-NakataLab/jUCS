"Represents a Conjunctive Normal Form (CNF) for fuzzy membership functions"
mutable struct CNF
    vs::Bool  # Very Small
    s::Bool   # Small
    m::Bool   # Medium
    l::Bool   # Large
    vl::Bool  # Very Large
end

"Constructor for CNF"
function CNF(vs, s, m, l, vl)
    return CNF(vs, s, m, l, vl)
end

"Calculate the membership value for a given input"
function get_membership_value(self::CNF, x::Union{Float64, Int64, String})::Float64
    vs, s, m, l, vl = self.vs, self.s, self.m, self.l, self.vl

    # If input is "?", return full membership
    x == "?" && return 1.0
    
    # Check if input is within valid range
    if !(0.0 <= x <= 1.0)
        println("Invalid input value.")
        exit(0)
    end

    # Determine which segment the input falls into
    segment = min(floor(Int, x * 4), 3)
    lower_bound = segment * 0.25
    upper_bound = (segment + 1) * 0.25

    # Get the flags for the lower and upper bounds of the segment
    lower_flag = (vs, s, m, l)[segment + 1]
    upper_flag = (s, m, l, vl)[segment + 1]

    # Calculate membership value based on flags
    if lower_flag && upper_flag
        return 1.0  # Full membership
    elseif lower_flag
        return (upper_bound - x) / 0.25  # Decreasing membership
    elseif upper_flag
        return (x - lower_bound) / 0.25  # Increasing membership
    else
        return 0.0  # No membership
    end
end

"Expand the CNF by setting a random false field to true"
function expansion!(cnf::CNF)
    false_fields = [fname for fname in fieldnames(CNF) if !getfield(cnf, fname)]

    # If there are any fields that are false, randomly select one and set it to true
    if !isempty(false_fields)
        chosen_field = rand(false_fields)
        setfield!(cnf, chosen_field, true)
    end
end

"Contract the CNF by setting a random true field to false"
function contraction!(cnf::CNF)
    true_fields = [fname for fname in fieldnames(CNF) if getfield(cnf, fname)]
    if !isempty(true_fields)
        chosen_field = rand(true_fields)
        setfield!(cnf, chosen_field, false)
    end
end

"Shift the CNF by moving a true value to an adjacent field"
function shift!(cnf::CNF)
    field_names = fieldnames(CNF)
    true_fields = [fname for fname in field_names if getfield(cnf, fname)]
    if !isempty(true_fields)
        chosen_field_idx = rand(1:length(true_fields))
        chosen_field = true_fields[chosen_field_idx]
        setfield!(cnf, chosen_field, false)

        # Find the immediately preceding or following element and set it to true.
        adjacent_idx = chosen_field_idx > 1 ? chosen_field_idx - 1 : chosen_field_idx + 1
        if adjacent_idx <= length(field_names)
            adjacent_field = field_names[adjacent_idx]
            setfield!(cnf, adjacent_field, true)
        end
    end
end

"Check if all fields in the CNF are true"
function are_all_true(cnf::CNF)
    for fname in fieldnames(CNF)
        if !getfield(cnf, fname)
            return false
        end
    end
    return true
end

"Check if all fields in the CNF are false"
function are_all_false(cnf::CNF)
    for fname in fieldnames(CNF)
        if getfield(cnf, fname)
            return false
        end
    end
    return true
end

"Check if exactly one field in the CNF is true"
function is_only_one_true(cnf::CNF)
    true_count = 0
    for fname in fieldnames(CNF)
        if getfield(cnf, fname)
            true_count += 1
        end
    end
    return true_count == 1
end

"Check if two CNF structures are equal"
function is_equal(self::CNF, other::CNF)::Bool
    if self.vs == other.vs && self.s == other.s && self.m == other.m && self.l == other.l && self.vl == other.vl
        return true
    else
        return false
    end
end


# for i in 1:100
#     x = 0.01*i
#     for a in (true, false)
#         for b in (true, false)
#             for c in (true, false)
#                 for d in (true, false)
#                     for e in (true, false)
#                         print("2")
#                         if get_membership_value(CNF(a,b,c,d,e), x) != get_membership_value_modify(CNF(a,b,c,d,e), x)
#                             print("error")
#                         end
#                     end
#                 end
#             end
#         end
#     end
# end