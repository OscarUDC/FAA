function holdOut(N::Int, P::Real)
    if P < 0 || P > 1
        error("P must be in the interval [0, 1]")
    end
    indexes = randperm(N)
    inTest = Int(round(P * N))
    inTrain = N - inTest
    return (indexes[1:inTrain], indexes[inTrain + 1:end])
end;

function holdOut(N::Int, Pval::Real, Ptest::Real)
    if (Pval < 0 || Pval > 1) || (Ptest < 0 || Ptest > 1) || (Pval + Ptest > 1)
        error("Pval and Ptest must be in the interval [0, 1], and Pval + Ptest can't be greater than 1")
    end
    trainIndexes, otherIndexes = holdOut(N, Pval + Ptest)
    newN = length(otherIndexes)
    inTest = Int(floor(newN * Ptest / (Pval + Ptest)))
    inVal = newN - inTest
    return (trainIndexes, otherIndexes[1:inVal], otherIndexes[inVal + 1:end])
end;