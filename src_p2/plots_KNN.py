import matplotlib.pyplot as plt

# Definir los datos
neighbors = [1, 2, 3, 4, 5, 6, 7, 15, 50]

# Métricas para cada número de vecinos
testAccuracy_means = [0.815209261636802, 0.790577678962949, 0.7963131677266434, 0.7835021648331273, 0.7868376326822577, 0.772625023710914, 0.7674149500969213, 0.7129831469539369, 0.5940647681113409]
testAccuracy_std = [0.015430424791617934, 0.012857016349008356, 0.007690640500907957, 0.014585115995791063, 0.01349737297975907, 0.01594562659816133, 0.016819403019208357, 0.029937806713740217, 0.013988362835479044]

testErrorRate_means = [0.1847907383631981, 0.20942232103705108, 0.20368683227335663, 0.21649783516687277, 0.21316236731774235, 0.22737497628908598, 0.23258504990307868, 0.28701685304606317, 0.4059352318886592]
testErrorRate_std = [0.015430424791617936, 0.012857016349008356, 0.007690640500907958, 0.014585115995791063, 0.01349737297975907, 0.01594562659816133, 0.016819403019208354, 0.029937806713740217, 0.013988362835479046]

# Crear gráfico para testAccuracy
plt.errorbar(neighbors, testAccuracy_means, yerr=testAccuracy_std, label='testAccuracy', marker='o', linestyle='-')

# Crear gráfico para testErrorRate
plt.errorbar(neighbors, testErrorRate_means, yerr=testErrorRate_std, label='testErrorRate', marker='o', linestyle='-')

# Configurar el gráfico
plt.xlabel('Number of Neighbors')
plt.ylabel('Metric Value')
plt.title('Performance Metrics vs. Number of Neighbors for KNN')
plt.legend()
plt.grid(True)

# Mostrar el gráfico
plt.show()
