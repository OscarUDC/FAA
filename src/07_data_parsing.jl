#2
using Random

function holdOut(N::Int, P::Real)
    if P < 0 || P > 1
        error("P must be in the interval [0, 1]")
    end
    # Generar una permutación aleatoria de los índices
    indices = randperm(N)

    # Calcular el número de patrones para el conjunto de test
    numTest = round(Int, N * P)

    # Separar los índices en conjuntos de entrenamiento y test
    testIndices = indices[1:numTest]
    trainIndices = indices[numTest+1:N]

    return Tuple{trainIndices, testIndices}
end;

function holdOut(N::Int, Pval::Real, Ptest::Real)
    if (Pval < 0 || Pval > 1) || (Ptest < 0 || Ptest > 1) || (Pval + Ptest > 1)
        error("Pval and Ptest must be in the interval [0, 1], and Pval + Ptest can't be greater than 1")
    end

    #separamos train y test de validación con holdOut
    trainTestIndexes, valIndexes = holdOut(N, Pval)

    #separamos train de test con holdOut otra vez
    trainIndexes, testIndexes = holdOut(length(trainTestIndexes), Ptest / (1 - Pval))

    #Returneamos todo xD
    return Tuple{trainIndexes, valIndexes, testIndexes}
end;