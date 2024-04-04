#4.1 and 4.2
using Statistics

function confusionMatrix(outputs::AbstractArray{Bool,1}, targets::AbstractArray{Bool,1})
    VP = sum(outputs .& targets)
    VN = sum((.!outputs) .& (.!targets))
    FP = sum(outputs .& .!targets)
    FN = sum(.!outputs .& targets)
    accuracy = (VP + VN + FP + FN) == 0 ? 0.0 : (VN + VP) / (VP + VN + FP + FN)
    error_rate = (VP + VN + FP + FN) == 0 ? 0.0 : (FN + FP) / (VP + VN + FP + FN)
    sensitivity = (VP + FN) == 0 ? 1.0 : VP / (FN + VP)
    specificity = (VN + FP) == 0 ? 1.0 : VN / (FP + VN)
    ppv = (VP + FP) == 0 ? 1.0 : VP / (VP + FP)
    npv = (VN + FN) == 0 ? 1.0 : VN / (VN + FN)
    f1_score = (sensitivity + ppv) == 0 ? 0.0 : 2 * (ppv * sensitivity) / (ppv + sensitivity)
    confusion_matrix = [VN FP; FN VP]
    return (accuracy, error_rate, sensitivity, specificity, ppv, npv, f1_score, confusion_matrix)
end;

function confusionMatrix(outputs::AbstractArray{<:Real,1}, targets::AbstractArray{Bool,1}; threshold::Real=0.5)
    binary_outputs = outputs .≥ threshold
    return confusionMatrix(binary_outputs, targets)
end;

function printConfusionMatrix(outputs::AbstractArray{Bool,1}, targets::AbstractArray{Bool,1})
    results = confusionMatrix(outputs, targets)
    
    println("accuracy: ", results[1])
    println("Error Rate: ", results[2])
    println("Sensitivity: ", results[3])
    println("Specificity: ", results[4])
    println("PPV: ", results[5])
    println("NPV: ", results[6])
    println("F1-Score: ", results[7])
    println("Confusion Matrix:")
    println(results[8])
end;

function printConfusionMatrix(outputs::AbstractArray{<:Real,1}, targets::AbstractArray{Bool,1}; threshold::Real=0.5)
    binary_outputs = outputs .≥ threshold #? Si el número de outpost es mayor que el umbral, hacemos un print de la matriz de confusión
    printConfusionMatrix(binary_outputs, targets)
end;

function confusionMatrix(outputs::AbstractArray{Bool,2}, targets::AbstractArray{Bool,2}; weighted::Bool=true)
    n_patterns, n_classes = size(outputs)  # Obtener las dimensiones de las matrices
    if n_classes != size(targets, 2)  # Verificar si el número de columnas es diferente en las matrices outputs y targets
        error("Number of columns in outputs and targets matrices must be equal")
    end

    if n_classes == 1  # Si solo hay una clase, llamar a la función anterior
        return confusionMatrix(outputs[:, 1], targets[:, 1]; weighted=weighted)
    end

    if n_classes == 2  # Comprobar si el número de clases es igual a 2
        error("Invalid input dimensions")
    end

    # Inicializar vectores de métricas con ceros
    sensitivity = zeros(Float64, n_classes)
    specificity = zeros(Float64, n_classes)
    VPP = zeros(Float64, n_classes)
    VPN = zeros(Float64, n_classes)
    F1 = zeros(Float64, n_classes)

    # Calcular matriz de confusión(sin necesidad de inicializarla primero con ceros)
    confusion_matrix = [sum(outputs[:, j] .& targets[:, i]) for i in 1:n_classes, j in 1:n_classes] #Hacemos un doble bucle para rellenar la matriz con ceros de primeras, con el fin de reservar los huecos en memoria

    # Calcular métricas macro o weighted
    if weighted
        weights = sum(targets, dims=1)
        TP = sum(outputs .& targets, dims=1)
        TN = sum((.!outputs) .& (.!targets), dims=1)
        FP = sum(outputs .& .!targets, dims=1)
        FN = sum(.!outputs .& targets, dims=1)

        accuracy = sum(TP ./ (TP .+ FP) .* weights) / sum(weights)
        error_rate = 1 - accuracy
        sensitivity = sum(TP ./ (TP .+ FN) .* weights) / sum(weights)
        specificity = sum(TN ./ (TN .+ FP) .* weights) / sum(weights)
        VPP = sensitivity  # Same as sensitivity for multiclass
        VPN = specificity  # Same as specificity for multiclass
        F1 = 2 * sensitivity * accuracy / (sensitivity + accuracy)

    else
        TP = sum(outputs .& targets, dims=1)
        TN = sum((.!outputs) .& (.!targets), dims=1)
        FP = sum(outputs .& .!targets, dims=1)
        FN = sum(.!outputs .& targets, dims=1)

        accuracy = sum(TP ./ (TP .+ FP)) / n_classes
        error_rate = 1 - accuracy
        sensitivity = sum(TP ./ (TP .+ FN)) / n_classes
        specificity = sum(TN ./ (TN .+ FP)) / n_classes
        VPP = sensitivity  # Same as sensitivity for multiclass
        VPN = specificity  # Same as specificity for multiclass
        F1 = 2 * sensitivity * accuracy / (sensitivity + accuracy) / n_classes
    end

    # Unir los valores de métricas para cada clase en un único valor usando la estrategia macro o weighted
    sensitivity = weighted ? sensitivity : mean(sensitivity)
    specificity = weighted ? specificity : mean(specificity)
    VPP = weighted ? VPP : mean(VPP)
    VPN = weighted ? VPN : mean(VPN)
    F1 = weighted ? F1 : mean(F1)

    return (accuracy, error_rate, sensitivity, specificity, VPP, VPN, F1, confusion_matrix)
end;

# Definir función confusionMatrix para matrices de valores reales
function confusionMatrix(outputs::AbstractArray{<:Real,2}, targets::AbstractArray{Bool,2}; weighted::Bool=true)
    outputs_bool = classifyOutputs(outputs)
    return confusionMatrix(outputs_bool, targets; weighted=weighted)
end;

# Definir función confusionMatrix para vectores de cualquier tipo
function confusionMatrix(outputs::AbstractArray{<:Any,1}, targets::AbstractArray{<:Any,1}; weighted::Bool=true)
    # Asegurarse de que todas las clases del vector de outputs estén incluidas en el vector de targets
    @assert all([in(output, unique(targets)) for output in outputs]) "All elements of outputs must be present in targets"
    
    # Convertir los vectores targets y outputs a matrices one-hot
    targets_onehot = oneHotEncoding(targets)
    outputs_onehot = oneHotEncoding(outputs)
    
    # Llamar a la función confusionMatrix con las matrices one-hot
    return confusionMatrix(outputs_onehot, targets_onehot; weighted=weighted)
end;


function printConfusionMatrix(outputs::AbstractArray{Bool,2}, targets::AbstractArray{Bool,2}; weighted::Bool=true)
    metrics = confusionMatrix(outputs, targets; weighted=weighted)
    println("Metrics:")
    println("Precision: ", metrics[1])
    println("Error rate: ", metrics[2])
    println("Sensitivity: ", metrics[3])
    println("Specificity: ", metrics[4])
    println("VPP: ", metrics[5])
    println("VPN: ", metrics[6])
    println("F1: ", metrics[7])
    println("Confusion matrix:")
    println(metrics[8])
    return metrics
end

function printConfusionMatrix(outputs::AbstractArray{<:Real,2}, targets::AbstractArray{Bool,2}; weighted::Bool=true)
    outputs_bool = classifyOutputs(outputs)
    return printConfusionMatrix(outputs_bool, targets; weighted=weighted)
end

function printConfusionMatrix(outputs::AbstractArray{<:Any,1}, targets::AbstractArray{<:Any,1}; weighted::Bool=true)
    # Asegurar que todos los elementos de outputs estén presentes en targets
    @assert all([in(output, unique(targets)) for output in outputs]) "All elements of outputs must be present in targets"
    
    # Convertir los vectores targets y outputs a matrices one-hot
    targets_onehot = oneHotEncoding(targets)
    outputs_onehot = oneHotEncoding(outputs)
    
    # Llamar a la función confusionMatrix con las matrices one-hot
    metrics = confusionMatrix(outputs_onehot, targets_onehot; weighted=weighted)
    println("Metrics:")
    println("Precision: ", metrics[1])
    println("Error rate: ", metrics[2])
    println("Sensitivity: ", metrics[3])
    println("Specificity: ", metrics[4])
    println("VPP: ", metrics[5])
    println("VPN: ", metrics[6])
    println("F1: ", metrics[7])
    println("Confusion matrix:")
    println(metrics[8])
    return metrics
end


# Test 1: Matrices booleanas multiclase
outputs_bool = rand(Bool, 100, 5)
targets_bool = rand(Bool, 100, 5)

# Imprimir las matrices de confusión
printConfusionMatrix(outputs_bool, targets_bool)

# Test 2: Matrices reales multiclase
outputs_real = rand(100, 5)
targets_bool = rand(Bool, 100, 5)

# Imprimir las matrices de confusión
printConfusionMatrix(outputs_real, targets_bool)

# Test 3: Vectores de tipo Any multiclase
outputs_any = ["class1", "class2", "class3", "class1", "class2", "class3"]
targets_any = ["class1", "class2", "class3", "class1", "class2", "class3"]

# Imprimir las matrices de confusión
printConfusionMatrix(outputs_any, targets_any)




# Matriz de salidas balanceada
outputs_balanced = [
    1 0 0 0 0;
    0 1 0 0 0;
    0 0 1 0 0;
    0 0 0 1 0;
    0 0 0 0 1
]

# Matriz de salidas desbalanceada
outputs_imbalanced = [
    1 0 0 0 0;
    1 0 0 0 0;
    1 0 0 0 0;
    1 0 0 0 0;
    1 0 0 0 0
]

# Matriz de salidas aleatorias (seleccionando una clase más probable)
outputs_random = [
    1 0 0 0 0;
    0 1 0 0 0;
    0 0 1 0 0;
    0 1 0 0 0;
    0 0 0 0 1
]

# Matriz de salidas aleatorias desbalanceada (seleccionando una clase menos probable)
outputs_random_imbalanced = [
    1 0 0 0 0;
    1 0 0 0 0;
    1 0 0 0 0;
    2 0 0 0 0;
    1 0 0 0 0
]

# Matriz de salidas aleatorias (seleccionando una clase más probable)
outputs_random = [
    1 0 0 0 0;
    0 1 0 0 0;
    0 0 1 0 0;
    0 1 0 0 0;
    0 0 0 0 1
]

# Matriz de salidas aleatorias desbalanceada (seleccionando una clase menos probable)
outputs_random_imbalanced = [
    1 0 0 0 0;
    1 0 0 0 0;
    1 0 0 0 0;
    2 0 0 0 0;
    1 0 0 0 0
]

# Matriz de salidas aleatorias (seleccionando una clase más probable)
outputs_random = [
    1 0 0 0 0;
    0 1 0 0 0;
    0 0 1 0 0;
    0 1 0 0 0;
    0 0 0 0 1
]

# Matriz de salidas aleatorias desbalanceada (seleccionando una clase menos probable)
outputs_random_imbalanced = [
    1 0 0 0 0;
    1 0 0 0 0;
    1 0 0 0 0;
    2 0 0 0 0;
    1 0 0 0 0
]

# Test 1: Matrices booleanas multiclase
targets_bool = rand(Bool, 5, 5)

# Imprimir las matrices de confusión sin ponderación
printConfusionMatrix(outputs_balanced, targets_bool, weighted=false)

# Imprimir las matrices de confusión con ponderación
printConfusionMatrix(outputs_balanced, targets_bool, weighted=true)

