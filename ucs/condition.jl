```
Classifier Condition: the Unordered Bound Hyperrectangular Representation
```

mutable struct UBR
    p::Float64
    q::Float64
end

function UBR(p, q)
    return UBR(p, q)
end

function get_lower_bound(self::UBR)::Float64
    return min(self.p, self.q)
end

function get_upper_bound(self::UBR)::Float64
    return max(self.p, self.q)
end

function get_lower_upper_bounds(self::UBR)::Tuple{Float64, Float64}
    p, q = self.p, self.q
    if p <= q
        return p, q
    else
        return q, p
    end
end

function is_equal(self::UBR, other::UBR)::Bool
    if self.p == other.p && self.q == other.q
        return true
    else
        return false
    end
end

