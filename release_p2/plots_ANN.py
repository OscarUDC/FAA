import matplotlib.pyplot as plt

# Definir los datos
topology = [[10, 15], [4, 6], [15], [5, 13], [8, 9], [11], [5], [7, 14]]

# Métricas para cada topología
testAccuracy_means = [0.9532497781795459, 0.943104940824535, 0.8863602784007879, 0.9537885855727135, 0.9500725726224697, 0.8771733768037757, 0.8499583927148594, 0.9547180365984065]
testAccuracy_std = [0.005628308370474634, 0.008630585360372063, 0.012065502854621348, 0.006334702104108787, 0.0056834616358628345, 0.012599495765263726, 0.01922424965084851, 0.003159442794037841]

testErrorRate_means = [0.046750221820454095, 0.0568950591754651, 0.11363972159921198, 0.046211414427286636, 0.049927427377530156, 0.12282662319622424, 0.15004160728514057, 0.045281963401593336]
testErrorRate_std = [0.005628308370474642, 0.008630585360372062, 0.012065502854621386, 0.006334702104108804, 0.005683461635862866, 0.012599495765263695, 0.019224249650848492, 0.0031594427940378027]

# Crear gráfico para testAccuracy
plt.errorbar(range(len(topology)), testAccuracy_means, yerr=testAccuracy_std, label='testAccuracy', marker='o', linestyle='-')

# Crear gráfico para testErrorRate
plt.errorbar(range(len(topology)), testErrorRate_means, yerr=testErrorRate_std, label='testErrorRate', marker='o', linestyle='-')

# Configurar el gráfico
plt.xticks(range(len(topology)), topology, rotation=45)
plt.xlabel('Topology')
plt.ylabel('Metric Value')
plt.title('Performance Metrics vs. Topology for ANN')
plt.legend()
plt.grid(True)

# Mostrar el gráfico
plt.show()
