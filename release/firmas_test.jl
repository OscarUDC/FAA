# La matriz booleana
matriz_bool = [
    1 0 0 0 0 0 0; 1 0 0 0 0 0 0; 1 0 0 0 0 0 0; 0 1 0 0 0 0 0; 1 0 0 0 0 0 0; 
    0 0 1 0 0 0 0; 0 0 0 1 0 0 0; 1 0 0 0 0 0 0; 1 0 0 0 0 0 0; 1 0 0 0 0 0 0; 
    1 0 0 0 0 0 0; 1 0 0 0 0 0 0; 1 0 0 0 0 0 0; 0 0 1 0 0 0 0; 1 0 0 0 0 0 0; 
    1 0 0 0 0 0 0; 1 0 0 0 0 0 0; 0 0 1 0 0 0 0; 0 0 1 0 0 0 0; 1 0 0 0 0 0 0; 
    0 1 0 0 0 0 0; 0 0 1 0 0 0 0; 0 0 1 0 0 0 0; 1 0 0 0 0 0 0; 1 0 0 0 0 0 0; 
    1 0 0 0 0 0 0; 0 1 0 0 0 0 0; 0 0 1 0 0 0 0; 1 0 0 0 0 0 0; 0 1 0 0 0 0 0; 
    0 0 0 1 0 0 0; 1 0 0 0 0 0 0; 0 1 0 0 0 0 0; 1 0 0 0 0 0 0; 1 0 0 0 0 0 0; 
    1 0 0 0 0 0 0; 0 1 0 0 0 0 0; 1 0 0 0 0 0 0; 1 0 0 0 0 0 0; 1 0 0 0 0 0 0; 
    1 0 0 0 0 0 0; 0 1 0 0 0 0 0; 1 0 0 0 0 0 0; 1 0 0 0 0 0 0; 1 0 0 0 0 0 0; 
    0 0 1 0 0 0 0; 1 0 0 0 0 0 0; 1 0 0 0 0 0 0; 1 0 0 0 0 0 0; 0 0 1 0 0 0 0; 
    0 0 1 0 0 0 0; 1 0 0 0 0 0 0; 0 1 0 0 0 0 0; 1 0 0 0 0 0 0; 1 0 0 0 0 0 0; 
    1 0 0 0 0 0 0; 1 0 0 0 0 0 0; 0 0 1 0 0 0 0; 1 0 0 0 0 0 0; 1 0 0 0 0 0 0; 
    1 0 0 0 0 0 0; 0 0 1 0 0 0 0;
]

# Convertir la matriz booleana a una matriz numérica
matriz_num = convert(Array{Float64}, matriz_bool)

# Función para convertir la matriz en un vector
function matriz_a_vector(matriz)
    vector = zeros(Float64, size(matriz, 1))
    for i in 1:size(matriz, 1)
        # Encontrar la posición del 1 en la fila
        idx = findfirst(matriz[i, :] .== 1)
        # Asignar el valor según la posición del 1
        vector[i] = (idx - 1) / 6
    end
    return vector
end

# Convertir la matriz en un vector
vector_resultante = matriz_a_vector(matriz_num)

# Mostrar el vector resultante
println(vector_resultante)

