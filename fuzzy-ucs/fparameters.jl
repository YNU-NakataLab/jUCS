"""
Fuzzy-UCS Hyperparameters
(for further details: julia ./fuzzy-ucs/main.jl --help)
"""
mutable struct Parameters
    N::Int
    F0::Float64
    nu::Float64
    theta_GA::Int
    chi::Float64
    mu::Float64
    theta_del::Int
    delta::Float64
    theta_sub::Int
    tau::Float64
    do_GA_subsumption::Bool
    do_correct_set_subsumption::Bool
    P_hash::Float64
    theta_exploit::Float64
end

function Parameters(args)
    return Parameters(
        args["N"], 
        args["F0"], 
        args["nu"], 
        args["theta_GA"],
        args["chi"], 
        args["mu"], 
        args["theta_del"], 
        args["delta"], 
        args["theta_sub"],
        args["tau"], 
        args["do_GA_subsumption"], 
        args["do_correct_set_subsumption"], 
        args["P_hash"], 
        args["theta_exploit"]
    )
end