include("fucs.jl")

using DataFrames, CSV, Printf, MLJ, Statistics, Suppressor, CategoricalArrays

mutable struct Helper
    env::Environment
end

function Helper(env)
    return Helper(env)
end

function make_one_column_csv(filename::String, list)
    dataframe = DataFrame(x=list)
    CSV.write(filename, dataframe, delim=',', writeheader=false)
end

function make_matrix_csv(filename::String, list)
    tbl = Tables.table(list)
    CSV.write(filename, tbl, header=false)
end

function make_classifier_list(self::Helper, fucs::FUCS)::Array{Any, 2}
    classifier_list = Array{Any}(undef, length(fucs.population) + 1, 11)
    classifier_list[1, 1] = "Classifier"
    classifier_list[1, 2] = "Antecedent"
    classifier_list[1, 3] = "Consequent"
    classifier_list[1, 4] = "Weight"
    classifier_list[1, 5] = "Weight Vector"
    classifier_list[1, 6] = "Fitness"
    classifier_list[1, 7] = "Experience"
    classifier_list[1, 8] = "Correct Matching Vector"
    classifier_list[1, 9] = "Correct Set Size"
    classifier_list[1, 10] = "Time Stamp"
    classifier_list[1, 11] = "Numerosity"
   
    i::Int64 = 2
    for clas in fucs.population
        condition::String = ""
        for j in 1:length(clas.condition)
            condition *= "{"
            if are_all_true(clas.condition[j])
                condition *= "#, "
            else
                if clas.condition[j].vs
                    condition *= "vS, "
                end
                if clas.condition[j].s 
                    condition *= "S, "
                end
                if clas.condition[j].m
                    condition *= "M, "
                end
                if clas.condition[j].l
                    condition *= "L, "
                end
                if clas.condition[j].vl
                    condition *= "vL, "
                end
            end
            condition = chop(condition, tail=2) * "}, "
        end
        condition = chop(condition, tail=2)
        classifier_list[i, 1] = clas.id
        classifier_list[i, 2] = condition
        classifier_list[i, 3] = argmax(clas.weight_array) - 1
        classifier_list[i, 4] = maximum(clas.weight_array)
        classifier_list[i, 5] = clas.weight_array
        classifier_list[i, 6] = clas.fitness
        classifier_list[i, 7] = clas.experience
        classifier_list[i, 8] = clas.correct_matching_array
        classifier_list[i, 9] = clas.correct_set_size
        classifier_list[i, 10] = clas.time_stamp
        classifier_list[i, 11] = clas.numerosity
        i += 1
    end
    return classifier_list
end

function make_parameter_list(args)::Array{Any, 2}
    parameter_list = Array{Any}(undef, 14, 2)

    parameter_list[1,:] = ["N", args["N"]]
    parameter_list[2,:] = ["F0", args["F0"]]
    parameter_list[3,:] = ["nu", args["nu"]]
    parameter_list[4,:] = ["chi", args["chi"]] 
    parameter_list[5,:] = ["mu", args["mu"]]
    parameter_list[6,:] = ["delta", args["delta"]]
    parameter_list[7,:] = ["theta_GA", args["theta_GA"]]
    parameter_list[8,:] = ["theta_del", args["theta_del"]]
    parameter_list[9,:] = ["theta_sub", args["theta_sub"]]
    parameter_list[10,:] = ["theta_exploit", args["theta_exploit"]]
    parameter_list[11,:] = ["tau", args["tau"]]
    parameter_list[12,:] = ["P_hash", args["P_hash"]]
    parameter_list[13,:] = ["doGASubsumption", args["do_GA_subsumption"]]
    parameter_list[14,:] = ["doCSSubsumption", args["do_correct_set_subsumption"]]

    return parameter_list
end

function val_with_spaces(val::Any)
    s = ""
    str_val = string(val)
    @inbounds for i in 1:12-length(str_val)
        s *= " "
    end
    return s * str_val * " "
end

function output_full_score_for_csv(current_epoch::Int64, num_epoch::Int64, env::Environment, train_accuracy::Float64, test_accuracy::Float64, train_precision::Float64, test_precision::Float64, train_recall::Float64, test_recall::Float64, train_f1::Float64, test_f1::Float64, popsize::Int64, covering_occur_num::Int64, subsumption_occur_num::Int64, summary_list::Array{Any, 2}, all_metric::Bool)::Array{Any, 2}
    if current_epoch == 1
        if all_metric
            println("       Epoch    Iteration     TrainAcc      TestAcc     TrainPre      TestPre     TrainRec      TestRec      TrainF1       TestF1      PopSize    %Covering #Subsumption")
            println("============ ============ ============ ============ ============ ============ ============ ============ ============ ============ ============ ============ ============")
        else
            println("       Epoch    Iteration     TrainAcc      TestAcc      PopSize    %Covering #Subsumption")
            println("============ ============ ============ ============ ============ ============ ============")
        end
        
        summary_list[1,:] = [
            "Epoch", 
            "Iteration", 
            "Training Accuracy", 
            "Testing Accuracy", 
            "Training Precision",
            "Testing Precision",
            "Training Recall",
            "Testing Recall",
            "Training F1",
            "Testing F1",
            "Population Size", 
            "CoveringOccurRate", 
            "#SubsumptionOccur"]
    end

    train_accuracy_str = @sprintf("%.3f", train_accuracy)
    test_accuracy_str = @sprintf("%.3f", test_accuracy)
    if all_metric
        train_precision_str = @sprintf("%.3f", train_precision)
        test_precision_str = @sprintf("%.3f", test_precision)
        train_recall_str = @sprintf("%.3f", train_recall)
        test_recall_str = @sprintf("%.3f", test_recall)
        train_f1_str = @sprintf("%.3f", train_f1)
        test_f1_str = @sprintf("%.3f", test_f1)
        println(
            val_with_spaces(current_epoch), 
            val_with_spaces(current_epoch * size(env.train_data, 1)),
            val_with_spaces(train_accuracy_str), 
            val_with_spaces(test_accuracy_str), 
            val_with_spaces(train_precision_str), 
            val_with_spaces(test_precision_str), 
            val_with_spaces(train_recall_str), 
            val_with_spaces(test_recall_str), 
            val_with_spaces(train_f1_str), 
            val_with_spaces(test_f1_str), 
            val_with_spaces(popsize),
            val_with_spaces(round(covering_occur_num / size(env.train_data, 1), digits=3)), 
            val_with_spaces(subsumption_occur_num))
    else
        println(
            val_with_spaces(current_epoch), 
            val_with_spaces(current_epoch * size(env.train_data, 1)),
            val_with_spaces(train_accuracy_str), 
            val_with_spaces(test_accuracy_str), 
            val_with_spaces(popsize),
            val_with_spaces(round(covering_occur_num / size(env.train_data, 1), digits=3)), 
            val_with_spaces(subsumption_occur_num))
    end

    summary_list[round(Int, current_epoch) + 1,:] = [
        current_epoch, 
        current_epoch * size(env.train_data, 1), 
        train_accuracy, 
        test_accuracy, 
        train_precision,
        test_precision,
        train_recall,
        test_recall,
        train_f1,
        test_f1,
        popsize, 
        covering_occur_num / size(env.train_data, 1), 
        round(Int, subsumption_occur_num)]

    return summary_list
end

"Get Accuracy, Precision, Recall, and F1"
function get_scores_per_epoch(seed::Int64, fucs::FUCS, data::Array{Union{Float64, Int64, String}, 2}, inference::String)::Tuple{Float64, Float64, Float64, Float64}
    true_labels::Vector{Int64} = [Int64(data[row_index, size(data, 2)]) for row_index in 1:size(data, 1)]
    predicted_labels::Vector{Int64} = [class_inference(seed, fucs, data[row_index, 1:(size(data, 2)-1)], inference) for row_index in 1:size(data, 1)]

    # Determine the common levels
    all_levels = unique([true_labels; predicted_labels])

    cat_true_labels = categorical(true_labels, levels=all_levels, ordered=true)
    cat_predicted_labels = categorical(predicted_labels, levels=all_levels, ordered=true)
   
    # Step 1: Calculate confusion matrix
    cm = confusion_matrix(cat_predicted_labels, cat_true_labels)

    # Step 2: Calculate precision, recall, and F1 for each class
    precision_per_class = Dict{Int64, Float64}()
    recall_per_class = Dict{Int64, Float64}()
    f1_per_class = Dict{Int64, Float64}()

    for label in all_levels
        c = findfirst(==(label), all_levels)
        TP = cm[c, c]  # True Positives for class c
        FP = sum(cm[:, c]) - TP  # False Positives for class c
        FN = sum(cm[c, :]) - TP  # False Negatives for class c
        
        # Precision
        precision = TP / (TP + FP)
        precision_per_class[c] = isnan(precision) ? 0 : precision
        
        # Recall
        recall = TP / (TP + FN)
        recall_per_class[c] = isnan(recall) ? 0 : recall
        
        # F1 Score
        f1 = 2 * (precision * recall) / (precision + recall)
        f1_per_class[c] = isnan(f1) ? 0 : f1
    end

    # Step 3: Calculate Accuracy, Precision, Recall, and F1
    overall_accuracy = accuracy(true_labels, predicted_labels) * 100
    macro_precision = mean(values(precision_per_class)) * 100
    macro_recall = mean(values(recall_per_class)) * 100
    macro_f1 = mean(values(f1_per_class)) * 100

    return overall_accuracy, macro_precision, macro_recall, macro_f1
end

function get_idk_per_epoch(fucs::FUCS, data::Array{Union{Float64, Int64, String}, 2})::Float64
    idk_array::Vector{Float64} = [calculate_idk(fucs, data[row_index, 1:(size(data, 2)-1)]) for row_index in 1:size(data, 1)]
    return mean(idk_array)
end

function calculate_idk(fucs::FUCS, state::Vector{Union{Float64, Int64, String}})::Float64
    match_set::Vector{FClassifier} = @views generate_match_set(fucs, state, -1, true)
    css = @views generate_conjunctive_sum_set(fucs, match_set)
    if css[end-1] != 1
        return css[end] / (1 - css[end-1])
    else
        return 1
    end
end

function generate_conjunctive_sum_set(fucs::FUCS, match_set::Vector{FClassifier})
    experienced_match_set::Vector{FClassifier} = filter(clas -> clas.experience > fucs.parameters.theta_exploit, match_set)
    mass_all = ones(fucs.env.num_actions + 1) # Number of class and frame of discernment
    conjunctive_sum_set = zeros(fucs.env.num_actions + 2)

    if isempty(experienced_match_set)
        conjunctive_sum_set[end-1] = 1
        mass_all = zeros(fucs.env.num_actions + 1)
        mass_all[end] = 1
    else
        set_mass(experienced_match_set)
        for clas in experienced_match_set
            for i in 1:clas.numerosity
                mass_all = conjunctive_masses(mass_all, clas.mass_array)
            end
        end

        for i in 1:(length(conjunctive_sum_set)-2)
            conjunctive_sum_set[i] = mass_all[i]
        end
        conjunctive_sum_set[end-1] = 1 - sum(mass_all) # mass of conflict
        conjunctive_sum_set[end] = mass_all[end] # I don't know belief mass

        if round(sum(conjunctive_sum_set), digits=3) != 1.0 
            println("The sum of conjunctive sum set is not 1.")
            println(mass_all, conjunctive_sum_set)
            exit(1)
        end
    end

   return conjunctive_sum_set
end

function conjunctive_masses(mass_all::Vector{Float64}, mass_clas::Vector{Float64})
    combined_masses = zeros(length(mass_all))
    if sum(mass_all) == length(mass_all)
        # For the first classifier 
        combined_masses = deepcopy(mass_clas)
    else
        # For classifiers after the second
        for i in 1:(length(combined_masses)-1)
            # For each class
            combined_masses[i] = mass_all[i] * mass_clas[end] + mass_all[end] * mass_clas[i] + mass_all[i] * mass_clas[i]
        end
        # For frame of discernment
        combined_masses[end] = mass_all[end] * mass_clas[end]
        if round(sum(combined_masses), digits=3) > 1.0
            println("The sum of combined mass should be lower than 1.0.")
            println(mass_all, mass_clas, combined_masses)
            exit(1)
        end 

    end

    return combined_masses
end

function class_inference(seed::Int64, fucs::FUCS, state::Vector{Union{Float64, Int64, String}}, inference::String)
    match_set::Vector{FClassifier} = @views generate_match_set(fucs, state, -1, true)

    if inference == "vote"
        return voting_based_inference(seed, fucs, match_set)
    elseif inference == "swin"
        return single_winner_based_inference(seed, fucs, match_set)
    elseif inference == "ds"
        return Dempster_Shafer_theory_based_inference(seed, fucs, match_set)
    else
        error("A class inference scheme should be vote, swin, or ds.")
    end
end

function random_max_index(seed::Int64, arr)
    rng = MersenneTwister(seed)
    max_value = maximum(arr)
    max_indices = findall(x -> x == max_value, arr)
    return rand(rng, max_indices)
end

function voting_based_inference(seed::Int64, fucs::FUCS, match_set::Vector{FClassifier})::Int64
    experienced_match_set::Vector{FClassifier} = filter(clas -> clas.experience > fucs.parameters.theta_exploit, match_set)
    vote_array::Vector{Float64} = zeros(fucs.env.num_actions)
    @simd for clas in experienced_match_set
        vote_array[argmax(clas.weight_array)] += clas.fitness * clas.matching_degree * clas.numerosity
    end
    return random_max_index(seed, vote_array) - 1
end

function single_winner_based_inference(seed::Int64, fucs::FUCS, match_set::Vector{FClassifier})::Int64
    rng = MersenneTwister(seed)
    experienced_match_set::Vector{FClassifier} = filter(clas -> clas.experience > fucs.parameters.theta_exploit, match_set)
    if isempty(experienced_match_set)
        return rand(rng, 0:fucs.env.num_actions-1)
    else
        max_product = maximum(clas -> clas.fitness * clas.matching_degree, experienced_match_set)
        max_product_classes = filter(clas -> clas.fitness * clas.matching_degree == max_product, experienced_match_set)
        
        if length(max_product_classes) == 1
            return argmax(max_product_classes[1].weight_array) - 1
        else
            # If multiple max_product_clas are found
            actions = map(clas -> argmax(clas.weight_array) - 1, max_product_classes)
            return rand(rng, actions)
        end
    end
end

function Dempster_Shafer_theory_based_inference(seed::Int64, fucs::FUCS, match_set::Vector{FClassifier})::Int64
    bet_p_set = @views generate_BetP_set(fucs, match_set)
    return random_max_index(seed, bet_p_set) - 1
end
