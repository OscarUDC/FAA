import numpy as np
import matplotlib.pyplot as plt
import itertools

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Matriz de confusión',
                          cmap=plt.cm.Blues):
    """
    Esta función imprime y grafica la matriz de confusión.
    La normalización se puede aplicar estableciendo el valor `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Matriz de confusión normalizada")
    else:
        print('Matriz de confusión sin normalización')

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('Etiqueta verdadera')
    plt.xlabel('Etiqueta predicha')

# Definir las clases
classes = ['Clase 1', 'Clase 2', 'Clase 3', 'Clase 4', 'Clase 5', 'Clase 6', 'Clase 7']

# Definir las matrices de confusión
matrices_confusion = [
    np.array([[45, 0, 0, 11, 0, 2, 0], [0, 41, 3, 14, 0, 0, 0], [0, 5, 62, 3, 1, 0, 0], [4, 0, 0, 54, 0, 0, 0], [0, 0, 1, 0, 58, 0, 1], [5, 0, 0, 0, 0, 50, 0], [0, 0, 0, 0, 0, 0, 65]]),
    np.array([[45, 0, 0, 11, 0, 2, 0], [0, 39, 4, 14, 1, 0, 0], [0, 1, 68, 0, 1, 0, 1], [2, 0, 0, 56, 0, 0, 0], [0, 0, 2, 0, 57, 0, 1], [5, 0, 0, 0, 0, 50, 0], [0, 0, 0, 0, 0, 0, 65]])
]

# Calcular la matriz de confusión promedio
matriz_promedio = np.mean(matrices_confusion, axis=0)

# Trazar la matriz de confusión promedio
plot_confusion_matrix(matriz_promedio, classes, normalize=True)
plt.show()
