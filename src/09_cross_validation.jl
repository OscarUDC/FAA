#5 and 6
using Random
using Random:seed!

function crossvalidation(N::Int64, k::Int64)
    if k > N
        error("k cannot be greater than N")
    end
    subset = collect(1:k)
    subsets = repeat(subset, outer = ceil(Int, N/k))
    return shuffle!(subsets[1:N])
end;

function crossvalidation(targets::AbstractArray{Bool,1}, k::Int64)
    indexes = zeros(Int, length(targets))
    indexes[targets] .= crossvalidation(sum(targets), k)
    indexes[.!targets] .= crossvalidation(sum(.!targets), k)
    return indexes
end;

function crossvalidation(targets::AbstractArray{Bool,2}, k::Int64)
    #
    # Codigo a desarrollar
    #
end;

function crossvalidation(targets::AbstractArray{<:Any,1}, k::Int64)
    #
    # Codigo a desarrollar
    #
end;

# Test for crossvalidation(N, k)
println("Test 1: crossvalidation(N, k)")
try
    result = crossvalidation(10, 3)
    println("Test 1a (N=10, k=3) result: ", result)
catch e
    println("Test 1a (N=10, k=3) error: ", e)
end

try
    result = crossvalidation(5, 10)
    println("Test 1b (N=5, k=10) result: ", result)
catch e
    println("Test 1b (N=5, k=10) error: ", e)
end

# Test for crossvalidation with a boolean array
println("\nTest 2: crossvalidation with a boolean array")
bool_array = [true, true, false, true, false]
try
    result = crossvalidation(bool_array, 2)
    println("Test 2 (Boolean array, k=2) result: ", result)
catch e
    println("Test 2 (Boolean array, k=2) error: ", e)
end

targets = [true false; false true; true false; false true; true false]
k = 2
targets = [true false; false true; true false; false true; true false]
k = 2

# Llamada a la función con los datos de prueba
indices = crossvalidation(targets, k)

# Imprimir resultados
println("Indices para k = 2: ", indices)

# Prueba con un valor diferente de k, por ejemplo 3
k = 3
indices = crossvalidation(targets, k)
println("Indices para k = 3: ", indices)