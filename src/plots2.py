import matplotlib.pyplot as plt

# Definir los datos
parametros = [
    {"C": 0.5, "kernel": "linear"},
    {"C": 1.0, "kernel": "poly", "gamma": "auto", "degree": 2, "coef0": 0.0},
    {"C": 1.5, "kernel": "rbf", "gamma": "auto"},
    {"C": 1.25, "kernel": "sigmoid", "gamma": "scale", "coef0": 3},
    {"C": 0.75, "kernel": "linear"},
    {"C": 1.75, "kernel": "poly", "gamma": "scale", "degree": 3, "coef0": 1.0},
    {"C": 2, "kernel": "rbf", "gamma": "auto"},
    {"C": 2, "kernel": "sigmoid", "gamma": "scale", "coef0": 2}
]

# Métricas para cada conjunto de parámetros
testAccuracy_means = [0.7603237398755496, 0.5580507655220023, 0.6409436931228369, 0.16627227950293638, 0.8124070830321999, 0.9185423339774668, 0.6698039214430898, 0.1672280217245326]
testAccuracy_std = [0.015356388828366145, 0.010093105983785034, 0.011067644404191493, 0.0007351427711771233, 0.01928230423386647, 0.015308426841891374, 0.01271319225743189, 0.0031929632359481122]

testErrorRate_means = [0.23967626012445048, 0.4419492344779977, 0.3590563068771631, 0.8337277204970637, 0.18759291696779998, 0.08145766602253326, 0.33019607855691013, 0.8327719782754673]
testErrorRate_std = [0.015356388828366145, 0.010093105983785034, 0.011067644404191494, 0.0007351427711770937, 0.01928230423386647, 0.015308426841891374, 0.01271319225743189, 0.0031929632359481313]

# Crear gráfico para testAccuracy
plt.errorbar(range(len(parametros)), testAccuracy_means, yerr=testAccuracy_std, label='testAccuracy', marker='o', linestyle='-')

# Crear gráfico para testErrorRate
plt.errorbar(range(len(parametros)), testErrorRate_means, yerr=testErrorRate_std, label='testErrorRate', marker='o', linestyle='-')

# Configurar el gráfico
plt.xticks(range(len(parametros)), [f"Params {i+1}" for i in range(len(parametros))], rotation=45)
plt.xlabel('Parameters')
plt.ylabel('Metric Value')
plt.title('Performance Metrics vs. Parameters for SVC')
plt.legend()
plt.grid(True)

# Mostrar el gráfico
plt.show()

