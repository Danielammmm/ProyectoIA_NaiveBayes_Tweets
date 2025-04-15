from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

def evaluar_modelo(y_true, y_pred, etiquetas=['positive', 'negative', 'neutral']):
    # Imprimir métricas
    print("\n📊 Reporte de Clasificación:")
    print(classification_report(y_true, y_pred, target_names=etiquetas))

    # Matriz de Confusión
    cm = confusion_matrix(y_true, y_pred, labels=etiquetas)

    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=etiquetas, yticklabels=etiquetas)
    plt.xlabel("Predicho")
    plt.ylabel("Real")
    plt.title("Matriz de Confusión")
    plt.tight_layout()
    plt.savefig("docs/matriz_confusion.png")
    plt.show()
