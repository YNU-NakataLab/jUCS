using ArgParse
using Random
using CSV
using DataFrames
using Dates

function parse_commandline()
    s = ArgParseSettings(description="The Fuzzy-UCS classifier system")
    @add_arg_table s begin
        "--num_trials"
            help = "The number of trials"
            arg_type = Int64
            default = 30
        "--epoch", "-e"
            help = "The number of epochs"
            arg_type = Int64
            default = 50
        "--csv"
            help = "A CSV file for classification"
            arg_type = String
            default = nothing
        "-a", "--all"
            help = "Whether to conduct experiments using all CSV files"
            arg_type = Bool
            default = false
        "-N"
            help = "The maximum size of the population"
            arg_type = Int64
            default = 2000
        "--F0"
            help = "The fitness threshold"
            arg_type = Float64
            default = 0.99
        "--nu"
            help = "The exponent in the power function for parent selection"
            arg_type = Float64
            default = 1.0
        "--theta_GA"
            help = "The threshold for the GA application in a correct set"
            arg_type = Int64
            default = 50
        "--chi"
            help = "The probability of applying crossover"
            arg_type = Float64
            default = 0.8
        "--mu"
            help = "The probability of mutating one allele"
            arg_type = Float64
            default = 0.04
        "--theta_del"
            help = "The experience threshold over which the fitness of a classifier may be considered in its deletion probability"
            arg_type = Int64
            default = 50
        "--delta"
            help = "The fraction of the mean fitness of the population below which the fitness of a classifier may be considered in its vote for deletion"
            arg_type = Float64
            default = 0.1
        "--theta_sub"
            help = "The experience of a classifier required to be a subsumer"
            arg_type = Int64
            default = 50
        "--tau"
            help = "The tournament size for selection [Butz et al., 2003] (set 0 to use the roulette-wheel selection)"
            arg_type = Float64
            default = 0.4
        "--do_GA_subsumption"
            help = "Whether offspring are to be tested for possible logical subsumption by parents"
            arg_type = Bool
            default = true
        "--do_correct_set_subsumption"
            help = "Whether correct sets are to be tested for subsuming classifiers"
            arg_type = Bool
            default = true
        "--P_hash"
            help = "The probability of Don't Care in covering"
            arg_type = Float64
            default = 0.33
        "--theta_exploit"
            help = "The experience threshold for class inference"
            arg_type = Float64
            default = 10.0
        "--inference", "-i"
            help = "Class inference scheme (vote, swin, ds)"
            arg_type = String
            default = "vote"
        "--idk"
            help = "Whether to save the I don't know belief mass"
            arg_type = Bool
            default = false
        "--all_metrics"
            help = "Whether to output all metrics (if false, Precision, Recall, and F1 are not shown in the log)"
            arg_type = Bool
            default = true
    end
    return parse_args(s)
end

function main_csv(args; now_str=Dates.format(Dates.now(), "Y-m-d-H-M-S"))
    env = Environment(args)
    param = Parameters(args)
    helper = Helper(env)

    println("[ Settings ]")
    println("    Environment = ", get_environment_name(env))
    println("         #Epoch = ", args["epoch"])
    println("          #Inst = ", size(env.all_data, 1))
    println("           #Fea = ", env.state_length)
    println("            #Cl = ", env.num_actions)
    println("      Inference = ", args["inference"])

    println("\n[ Fuzzy-UCS General Parameters ]")
    println("              N = ", param.N)
    println("             F0 = ", param.F0)
    println("              ν = ", param.nu)
    println("           θ_GA = ", param.theta_GA)
    println("              χ = ", param.chi)
    println("              μ = ", param.mu)
    println("          θ_del = ", param.theta_del)
    println("              δ = ", param.delta)
    println("          θ_sub = ", param.theta_sub)
    println("      θ_exploit = ", param.theta_exploit)
    println("            P_# = ", param.P_hash)
    println("doGASubsumption = ", Bool(param.do_GA_subsumption))
    println("doCSSubsumption = ", Bool(param.do_correct_set_subsumption))

    println("\n[ Fuzzy-UCS Optional Settings ]")
    if param.tau != 0.
        println("              τ = ", param.tau)
    end
    if args["inference"] == "ds" && args["idk"]
        println("        SaveIDK = ", Bool(args["idk"]))
    end

    @time for n in 1:args["num_trials"]
        start_time = time()
        fucs::FUCS = FUCS(env, param)
        env.seed = n
        Random.seed!(env.seed)
        shuffle_train_and_test_data!(env)
        println("\n[ Seed $(env.seed) / $(args["num_trials"]) ]\n")
        time_list = Vector{Float64}(undef, 1)
        summary_list = Array{Any}(undef, args["epoch"] + 1, 13)

        if args["inference"] == "ds" && args["idk"]
            train_idk_list = Vector{Float64}(undef, args["epoch"])
            test_idk_list = Vector{Float64}(undef, args["epoch"])
        end

        for e in 1:args["epoch"]
            # Training Phase
            env.is_exploit = false
            num_iter = size(env.train_data, 1)
            for i in 1:num_iter
                run_experiment(fucs)
            end

            # Test Phase
            env.is_exploit = true
            train_accuracy::Float64, train_precision::Float64, train_recall::Float64, train_f1::Float64 = get_scores_per_epoch(env.seed, fucs, env.train_data, args["inference"])
            test_accuracy::Float64, test_precision::Float64, test_recall::Float64, test_f1::Float64 = get_scores_per_epoch(env.seed, fucs, env.test_data, args["inference"])
            if args["inference"] == "ds" && args["idk"]
                train_idk::Float64 = get_idk_per_epoch(fucs, env.train_data)
                test_idk::Float64 = get_idk_per_epoch(fucs, env.test_data)
                train_idk_list[e] = train_idk
                test_idk_list[e] = test_idk
            end
            popsize::Int64 = length(fucs.population)

            # Output log
            summary_list = output_full_score_for_csv(e, args["epoch"], env, train_accuracy, test_accuracy, train_precision, test_precision, train_recall, test_recall, train_f1, test_f1, popsize, fucs.covering_occur_num, fucs.subsumption_occur_num, summary_list, args["all_metrics"])
            fucs.covering_occur_num = fucs.subsumption_occur_num = 0.
        end
        elapsed_time = time() - start_time
        time_list[1] = elapsed_time

        if !args["all"]
            dir_path = "./fuzzy-ucs/result/" * args["inference"] * "/" * get_environment_name(env) * "/" * now_str * "/seed" * string(env.seed)
        else
            dir_path = "./all_" * now_str * "/fuzzy-ucs/" * args["inference"]  * "/" *  basename(get_environment_name(env)) * "/seed" * string(env.seed)
        end
        mkpath(dir_path)
        make_one_column_csv(joinpath(dir_path, "time.csv"), time_list)
        make_matrix_csv(joinpath(dir_path, "classifier.csv"), make_classifier_list(helper, fucs))
        make_matrix_csv(joinpath(dir_path, "summary.csv"), summary_list)
        make_matrix_csv(joinpath(dir_path, "parameter.csv"), make_parameter_list(args))

        if args["inference"] == "ds" && args["idk"]
            make_one_column_csv(joinpath(dir_path, "train_idk.csv"), train_idk_list)
            make_one_column_csv(joinpath(dir_path, "test_idk.csv"), test_idk_list)
        end
    end
end

function main_all_csv(args)
    now = Dates.now()
    now_str = Dates.format(now, "Y-m-d")
    dir_path::String = "./dataset/"
    csv_list_array::Vector{String} = [
    
        # 30 datasets used in the main article
        "banknote", 
        "cancer",
        "car",
        "column_3C_weka",
        "diabetes",
        "ecoli",
        "fruit",
        "glass",
        "heart",
        "hepatitis",
        "horse-colic",
        "iris",
        "mine",
        "mammographic_masses",
        "Paddy-Leaf",
        "pistachio",
        "pharyngitis",
        "engagement",
        "pumpkin",
        "raisin",
        "segment",
        "sirtuin",
        "smoker",
        "tae",
        "travel",
        "titanic",
        "wine",
        "breast-cancer-wisconsin",
        "wpbc",
        "yeast",

        # 7 datasets used in the supplementary material
        "12carry",
        "11mop",
        "11mux",
        "sonar",
        "movement_libras",
        "myocardial_infarction",
        "phishing"
    ]

    for csv::String in csv_list_array
        if csv == "movement_libras" || csv == "myocardial_infarction" || csv == "sonar"
            args["P_hash"] = 0.8
        else
            args["P_hash"] = 0.33
        end
        args["csv"] = dir_path * csv * ".csv"
        main_csv(args; now_str)
    end
end


args = parse_commandline()
args = parse_commandline()
include("../environment/real_world.jl")
include("fparameters.jl")
include("fhelper.jl")
include("fucs.jl")

if args["csv"] == nothing && args["all"]
    main_all_csv(args)
elseif args["csv"] != nothing && !args["all"]
    main_csv(args)
else
    error("Set either args[\"csv\"] or args[\"all\"]")
end