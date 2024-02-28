using Random

function holdOut(N::Int, P::Real)
    indexes = randperm(N)
    split = Int64(round(N * P))
    return tuple(indexes[1:split], indexes[split + 1:N])
end;

function holdOut(N::Int, Pval::Real, Ptest::Real)
    indexes = holdOut(N, Pval + Ptest)
    firstSplit, secondSplit = Int64(round(N * (Pval + Ptest))), Int64(round(N * Ptest))
    return tuple(indexes[1:firstSplit], [firstSplit + 1:secondSplit], [secondSplit + 1:N])
end;