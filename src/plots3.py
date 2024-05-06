import numpy as np
import matplotlib.pyplot as plt

def plot_confusion_matrix(matrix, classes):
    """
    Función para trazar una matriz de confusión.
    
    Args:
    - matrix (np.ndarray): Matriz de confusión.
    - classes (list): Lista de nombres de las clases.
    """
    plt.imshow(matrix, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Matriz de Confusión')
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    thresh = matrix.max() / 2.
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            plt.text(j, i, format(matrix[i, j], 'd'),
                     horizontalalignment="center",
                     color="white" if matrix[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('Etiqueta Verdadera')
    plt.xlabel('Etiqueta Predicha')
    plt.show()

# Definir las clases
classes = ["Clase 1", "Clase 2", "Clase 3", "Clase 4", "Clase 5", "Clase 6", "Clase 7"]

# Definir las matrices de confusión
matrices_confusion = [
    np.array([[45, 0, 0, 11, 0, 2, 0], [0, 41, 3, 14, 0, 0, 0], [0, 5, 62, 3, 1, 0, 0], [4, 0, 0, 54, 0, 0, 0], [0, 0, 1, 0, 58, 0, 1], [5, 0, 0, 0, 0, 50, 0], [0, 0, 0, 0, 0, 0, 65]]),
    np.array([[45, 0, 0, 11, 0, 2, 0], [0, 39, 4, 14, 1, 0, 0], [0, 1, 68, 0, 1, 0, 1], [2, 0, 0, 56, 0, 0, 0], [0, 0, 2, 0, 57, 0, 1], [5, 0, 0, 0, 0, 50, 0], [0, 0, 0, 0, 0, 0, 65]])
]

# Trazar las matrices de confusión
for i, matrix in enumerate(matrices_confusion):
    print("Matriz de confusión", i+1)
    plot_confusion_matrix(matrix, classes)
