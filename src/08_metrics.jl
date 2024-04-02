#4.1 and 4.2
using Statistics

include.([
    "04_accuracy.jl",
    "01_oneHotEncoding.jl"
    ])

function confusionMatrix(outputs::AbstractArray{Bool,1}, targets::AbstractArray{Bool,1})
    VP = sum(outputs .& targets)
    VN = sum((.!outputs) .& (.!targets))
    FP = sum(outputs .& .!targets)
    FN = sum(.!outputs .& targets)

    # Precisión
    if (VP + VN + FP + FN) == 0
        accuracy = 0.0
    else
        accuracy = (VN + VP) / (VN + VP + FN + FP)
    end
    
    # Tasa de error
    if (VP + VN + FP + FN) == 0
        error_rate = 0.0
    else
        error_rate = (FN + FP) / (VN + VP + FN + FP)
    end
    
    # Sensibilidad (Recall)
    if VP == 0 && FN == 0
        sensitivity = 1.0
    else
        sensitivity = VP / (FN + VP)
    end
    
    # Especificidad
    if VN == 0 && FP == 0
        specificity = 1.0
    else
        specificity = VN / (FP + VN)
    end
    
    # Valor predictivo positivo (Precision)
    if VP == 0 && FP == 0
        ppv = 1.0
    else
        ppv = VP / (VP + FP)
    end

    # Valor predictivo negativo
    if VN == 0 && FN == 0
        npv = 1.0
    else
        npv = VN / (VN + FN)
    end
    
    # F1-score
    if sensitivity == 0 && ppv == 0
        f1_score = 0.0
    else
        f1_score = 2 * (ppv * sensitivity) / (ppv + sensitivity)
    end    

    # Matriz de confusión
    confusion_matrix = [VN FP; FN VP]

    # Devolvemos la tupla con todas las métricas y la matriz de confusión
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


# # Caso 1: Clasificación correcta
# outputs_correctos = [true, false, true, false, true]
# targets_correctos = [true, false, true, false, true]

# # Caso 2: Todos clasificados como positivos
# outputs_todos_positivos = [true, true, true, true, true]
# targets_todos_positivos = [true, false, true, false, true]

# # Caso 3: Todos clasificados como negativos
# outputs_todos_negativos = [false, false, false, false, false]
# targets_todos_negativos = [true, false, true, false, true]

# # Caso 4: Todos patrones positivos y clasificados como positivos
# outputs_patrones_positivos = [true, true, true, true, true]
# targets_patrones_positivos = [true, true, true, true, true]

# # Caso 5: Todos patrones negativos y clasificados como negativos
# outputs_patrones_negativos = [false, false, false, false, false]
# targets_patrones_negativos = [false, false, false, false, false]

# # Llamadas a la función confusionMatrix
# resultados_caso1 = confusionMatrix(outputs_correctos, targets_correctos)
# resultados_caso2 = confusionMatrix(outputs_todos_positivos, targets_todos_positivos)
# resultados_caso3 = confusionMatrix(outputs_todos_negativos, targets_todos_negativos)
# resultados_caso4 = confusionMatrix(outputs_patrones_positivos, targets_patrones_positivos)
# resultados_caso5 = confusionMatrix(outputs_patrones_negativos, targets_patrones_negativos)

# # Imprimir resultados
# println("Caso 1:")
# println("Precisión (accuracy): ", resultados_caso1[1])
# println("Tasa de error (error rate): ", resultados_caso1[2])
# println("Sensibilidad (recall): ", resultados_caso1[3])
# println("Especificidad (specificity): ", resultados_caso1[4])
# println("Valor predictivo positivo (precision): ", resultados_caso1[5])
# println("Valor predictivo negativo: ", resultados_caso1[6])
# println("F1-Score: ", resultados_caso1[7])
# println("Matriz de confusión:")
# println(resultados_caso1[8])

# println("\nCaso 2:")
# println("Precisión (accuracy): ", resultados_caso2[1])
# println("Tasa de error (error rate): ", resultados_caso2[2])
# println("Sensibilidad (recall): ", resultados_caso2[3])
# println("Especificidad (specificity): ", resultados_caso2[4])
# println("Valor predictivo positivo (precision): ", resultados_caso2[5])
# println("Valor predictivo negativo: ", resultados_caso2[6])
# println("F1-Score: ", resultados_caso2[7])
# println("Matriz de confusión:")
# println(resultados_caso2[8])

# println("\nCaso 3:")
# println("Precisión (accuracy): ", resultados_caso3[1])
# println("Tasa de error (error rate): ", resultados_caso3[2])
# println("Sensibilidad (recall): ", resultados_caso3[3])
# println("Especificidad (specificity): ", resultados_caso3[4])
# println("Valor predictivo positivo (precision): ", resultados_caso3[5])
# println("Valor predictivo negativo: ", resultados_caso3[6])
# println("F1-Score: ", resultados_caso3[7])
# println("Matriz de confusión:")
# println(resultados_caso3[8])

# println("\nCaso 4:")
# println("Precisión (accuracy): ", resultados_caso4[1])
# println("Tasa de error (error rate): ", resultados_caso4[2])
# println("Sensibilidad (recall): ", resultados_caso4[3])
# println("Especificidad (specificity): ", resultados_caso4[4])
# println("Valor predictivo positivo (precision): ", resultados_caso4[5])
# println("Valor predictivo negativo: ", resultados_caso4[6])
# println("F1-Score: ", resultados_caso4[7])
# println("Matriz de confusión:")
# println(resultados_caso4[8])

# println("\nCaso 5:")
# println("Precisión (accuracy): ", resultados_caso5[1])
# println("Tasa de error (error rate): ", resultados_caso5[2])
# println("Sensibilidad (recall): ", resultados_caso5[3])
# println("Especificidad (specificity): ", resultados_caso5[4])
# println("Valor predictivo positivo (precision): ", resultados_caso5[5])
# println("Valor predictivo negativo: ", resultados_caso5[6])
# println("F1-Score: ", resultados_caso5[7])
# println("Matriz de confusión:")
# println(resultados_caso5[8])

#-----------------------------------------------

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
    # Convertir outputs a valores booleanos si es necesario
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

