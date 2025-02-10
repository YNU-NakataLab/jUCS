include("ucs.jl")

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

function make_classifier_list(self::Helper, ucs::UCS)::Array{Any, 2}
    classifier_list = Array{Any}(undef, length(ucs.population) + 1, 10)
    classifier_list[1, 1] = "Classifier"
    classifier_list[1, 2] = "Condition (lower:upper)"
    classifier_list[1, 3] = "Action"
    classifier_list[1, 4] = "Accuracy"
    classifier_list[1, 5] = "Fitness"
    classifier_list[1, 6] = "Correct Track"
    classifier_list[1, 7] = "Experience"
    classifier_list[1, 8] = "Time Stamp"
    classifier_list[1, 9] = "Correct Set Size"
    classifier_list[1, 10] = "Numerosity"

    i::Int64 = 2
    for clas in ucs.population
        condition::String = "["
        for j in 1:length(clas.condition)
            l, u = get_lower_upper_bounds(clas.condition[j])
            if l == 0.0 && u == 1.0
                condition *= "#, "
            else
                condition *= string(round(l, digits=3)) * ":" * string(round(u, digits=3)) * ", "
            end
        end
        condition = chop(condition, tail=2) * "]"
        classifier_list[i, 1] = clas.id
        classifier_list[i, 2] = condition
        classifier_list[i, 3] = clas.action
        classifier_list[i, 4] = clas.accuracy
        classifier_list[i, 5] = clas.fitness
        classifier_list[i, 6] = clas.correct_track
        classifier_list[i, 7] = clas.experience
        classifier_list[i, 8] = clas.time_stamp
        classifier_list[i, 9] = clas.correct_set_size
        classifier_list[i, 10] = clas.numerosity
        i += 1
    end
    return classifier_list
end

function make_parameter_list(args)::Array{Any, 2}
    parameter_list = Array{Any}(undef, 17, 2)

    parameter_list[1,:] = ["N", args["N"]]
    parameter_list[2,:] = ["beta", args["beta"]]
    parameter_list[3,:] = ["acc0", args["acc0"]]
    parameter_list[4,:] = ["nu", args["nu"]]
    parameter_list[5,:] = ["theta_GA", args["theta_GA"]]
    parameter_list[6,:] = ["chi", args["chi"]]
    parameter_list[7,:] = ["mu", args["mu"]]
    parameter_list[8,:] = ["theta_del", args["theta_del"]]
    parameter_list[9,:] = ["delta", args["delta"]]
    parameter_list[10,:] = ["theta_sub", args["theta_sub"]]
    parameter_list[11,:] = ["m_0", args["m0"]]
    parameter_list[12,:] = ["r_0", args["r0"]]
    parameter_list[13,:] = ["doGASubsumption", args["do_GA_subsumption"]]
    parameter_list[14,:] = ["doCSSubsumption", args["do_correct_set_subsumption"]]
    parameter_list[15,:] = ["useMAM", args["use_MAM"]]
    parameter_list[16,:] = ["tau", args["tau"]]
    parameter_list[17,:] = ["P_hash", args["P_hash"]]

    return parameter_list
end

function val_with_spaces(val::Any)
    s = ""
    str_val = string(val)
    for i in 1:12-length(str_val)
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
function get_scores_per_epoch(seed::Int64, ucs::UCS, data::Array{Union{Float64, Int64, String}, 2}, changed_data::Array{Union{Float64, String}, 2})::Tuple{Float64, Float64, Float64, Float64}
    true_labels::Vector{Int64} = [Int64(data[row_index, size(data, 2)]) for row_index in 1:size(data, 1)]
    predicted_labels::Vector{Int64} = [class_inference(seed, ucs, changed_data[row_index, 1:size(changed_data, 2)]) for row_index in 1:size(changed_data, 1)]

    # Determine the common levels
    all_levels = unique([true_labels; predicted_labels])

    cat_true_labels = categorical(true_labels, levels=all_levels, ordered=true)
    cat_predicted_labels = categorical(predicted_labels, levels=all_levels, ordered=true)
   
    # Step 1: Calculate confusion matrix
    cm = confusion_matrix(cat_true_labels, cat_predicted_labels)

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

    # Step 3: Calculate Macro Accuracy, Precision, Recall, and F1
    macro_accuracy = accuracy(true_labels, predicted_labels) * 100
    macro_precision = mean(values(precision_per_class)) * 100
    macro_recall = mean(values(recall_per_class)) * 100
    macro_f1 = mean(values(f1_per_class)) * 100

    return macro_accuracy, macro_precision, macro_recall, macro_f1
end

function change_array_type(original_array::Array{Union{Float64, Int64, String}, 2})::Array{Union{Float64, String}, 2}
    original_array = original_array[:, 1:end-1]
    converted_array = Array{Union{Float64, String}, 2}(undef, size(original_array))

    for i in 1:size(original_array, 1)
        for j in 1:size(original_array, 2)
            element = original_array[i, j]
            if element isa Int
                # Convert Int64 to Float64
                converted_array[i, j] = float(element)
            else
                # Keep Float64 and String as they are
                converted_array[i, j] = element
            end
        end
    end
    return converted_array
end

function class_inference(seed::Int64, ucs::UCS, state::Vector{Union{Float64, String}})::Int64
    match_set::Vector{Classifier} = @views generate_match_set(ucs, state, -1, true)
    fitness_sum_array::Vector{Float64} = @views generate_fitness_sum_array(ucs, match_set)
    action::Int64 = random_max_index(seed, fitness_sum_array) - 1
    return action
end

function random_max_index(seed::Int64, arr)
    rng = MersenneTwister(seed)
    max_value = maximum(arr)
    max_indices = findall(x -> x == max_value, arr)
    return rand(rng, max_indices)
end