using CSV
using DataFrames
using Plots
using MultivariateStats
import Pkg

# Cargar el archivo CSV
data = CSV.read("db\\raw\\ObesityDataSet_raw_and_data_sinthetic.csv", DataFrame)

# Definir las columnas continuas
continuos_columns = [:Age, :Height, :Weight, :FCVC, :NCP, :CH2O, :FAF, :TUE]

# Crear y guardar histogramas para cada variable continua
hist_plots = []

for col in continuos_columns
    hist_plot = histogram(data[!, col], xlabel=string(col), ylabel="Frequency", title=string(col)*" Distribution", legend=false)
    push!(hist_plots, hist_plot)
    savefig(hist_plot, "hist_$(col).png")
end

# Mostrar los histogramas
hist_plots

using Plots

# Definir las columnas continuas
continuos_columns = [:Age, :Height, :Weight, :FCVC, :NCP, :CH2O, :FAF, :TUE]

# Crear y guardar diagramas de dispersión para cada variable continua contra NObeyesdad
scatter_plots = []

for col in continuos_columns
    scatter_plot = scatter(data[!, col], data[!, :NObeyesdad], xlabel=string(col), ylabel="NObeyesdad", title=string(col)*" vs. NObeyesdad", legend=false)
    push!(scatter_plots, scatter_plot)
    savefig(scatter_plot, "scatter_$(col)_nobeysdad.png")
end

# Mostrar los plots
scatter_plots

gr()

# Obtener las columnas de interés
x_data = data.Age
y_data = data.Weight

# Función para generar el gif
anim = @animate for i in 1:size(data, 1)
    scatter(x_data[1:i], y_data[1:i], xlabel="Age", ylabel="Weight", legend=false)
end every 5

# Guardar el gif con mejor resolución
gif(anim, "scatter_animation.gif", fps = 10)