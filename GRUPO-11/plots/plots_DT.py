import matplotlib.pyplot as plt

# Definir los datos
depths = [6, 7, 5, 8, 9, 10, 12, 15]

# Métricas para cada profundidad
testAccuracy_means = [0.8882324268388843, 0.8943588888968396, 0.8280854090033809, 0.9298912264408294, 0.9308312613980293, 0.9317612013747372, 0.9336625487823419, 0.9346126675471874]
testAccuracy_std = [0.01899202335760786, 0.008541891522433091, 0.024772138917409155, 0.010437306727676746, 0.012437518375072421, 0.008827669171620672, 0.005293272232211878, 0.0045653520866117715]

testErrorRate_means = [0.11176757316111585, 0.1056411111031604, 0.17191459099661913, 0.07010877355917049, 0.06916873860197062, 0.06823879862526287, 0.06633745121765822, 0.06538733245281261]
testErrorRate_std = [0.01899202335760786, 0.008541891522433091, 0.024772138917409155, 0.010437306727676746, 0.012437518375072421, 0.008827669171620672, 0.005293272232211878, 0.0045653520866117715]

# Crear gráfico para testAccuracy
plt.errorbar(depths, testAccuracy_means, yerr=testAccuracy_std, label='testAccuracy', marker='o', linestyle='-')

# Crear gráfico para testErrorRate
plt.errorbar(depths, testErrorRate_means, yerr=testErrorRate_std, label='testErrorRate', marker='o', linestyle='-')

# Configurar el gráfico
plt.xlabel('Depth')
plt.ylabel('Metric Value')
plt.title('Performance Metrics vs. Depth')
plt.legend()
plt.grid(True)

# Mostrar el gráfico
plt.show()
