# ProyectoIA_NaiveBayes_Tweets

Este proyecto consiste en una aplicación web que clasifica el sentimiento (positivo, negativo o neutro) de tweets usando el algoritmo Naïve Bayes implementado desde cero.

El sistema está desarrollado bajo una arquitectura en capas y contiene una interfaz web funcional para ingresar el texto a analizar y visualizar el resultado.

---

## 📁 Estructura del Proyecto

```
ProyectoIA_NaiveBayes_Tweets/
│
├── data/                            ← Capa de datos: carga y limpieza del dataset
│   ├── sentiment140_sample.csv     ← Subset del dataset original (opcional)
│   ├── cleaned_data.pkl            ← Dataset limpio, serializado
│   └── preprocessing.py           ← Funciones para limpiar y tokenizar texto
│
├── model/                           ← Capa del modelo Naïve Bayes
│   ├── naive_bayes.py             ← Implementación desde cero del clasificador
│   ├── metrics.py                 ← Cálculo de Precisión, Recall, F1 y matriz de confusión
│   └── model.pkl                  ← Modelo entrenado y serializado (binario)
│
├── service/                         ← Motor de inferencia / Backend con Flask
│   ├── inference.py               ← Funciones que usan el modelo para predecir
│   └── app.py                     ← Aplicación principal Flask, expone el API
│
├── web/                             ← Interfaz web (Frontend)
│   ├── static/
│   │   └── styles.css           ← Estilos CSS para la web
│   ├── templates/
│   │   └── index.html           ← Vista principal con formulario
│   └── script.js                 ← Lógica para enviar texto al backend y mostrar resultado
│
├── docs/                            ← Documentación y recursos de apoyo
│   ├── arquitectura.drawio       ← Diagrama de arquitectura en capas
│   ├── casos_uso.md              ← Casos de uso del sistema
│   └── resultados.md             ← Métricas del modelo y ejemplos
│
├── .gitignore                       ← Archivos y carpetas a ignorar por Git
├── README.md                        ← Descripción general, instrucciones y estructura
├── requirements.txt                 ← Lista de dependencias del proyecto
└── main.py                        ← Script principal para entrenar el modelo
```

---

## 📆 Tareas Generales (por Fases)

### Fase 1: Configuración Inicial y Dataset
- [ ] Crear repo y estructura de carpetas (**Daniela**)
- [ ] Descargar y limpiar subset de `sentiment140` (**Nilssen**)
- [ ] Crear entorno virtual e instalar dependencias (**Nilssen**)

### Fase 2: Preprocesamiento de Datos
- [ ] Implementar funciones de limpieza (`clean_text`) y tokenización (**Nilssen**)
- [ ] Generar representación en bolsa de palabras (**Nilssen**)

### Fase 3: Implementación del Modelo
- [ ] Crear clase `NaiveBayesClassifier` (**Daniela**)
- [ ] Entrenar y guardar modelo con `pickle` (**Daniela**)
- [ ] Evaluar modelo: Precisión, Recall, F1, Confusión (**Daniela**)

### Fase 4: Motor de Inferencia (Flask)
- [ ] Crear `app.py` con endpoint `/predict` (**Daniela**)
- [ ] Integrar modelo y limpieza de entrada (**Ambos**)
- [ ] Calcular tiempo de inferencia (**Daniela**)

### Fase 5: Interfaz Web
- [ ] Diseñar `index.html` con campo de texto (**Nilssen**)
- [ ] Implementar `script.js` para conectar con backend (**Nilssen**)
- [ ] Mostrar resultado y tiempo de respuesta (**Nilssen**)

### Fase 6: Documentación y Presentación
- [ ] Crear informe PDF con diagramas y explicaciones (**Ambos**)
- [ ] Completar `README.md` con instrucciones claras (**Daniela**)
- [ ] Preparar capturas y video de demo final (**Nilssen**)

---

## 🗓️ Cronograma sugerido (por semanas)

| Semana | Actividades principales |
|--------|--------------------------|
| Semana 1 | Configuración inicial, descarga y limpieza del dataset, entorno virtual |
| Semana 2 | Preprocesamiento, implementación de Naïve Bayes, pruebas básicas |
| Semana 3 | Backend Flask, conexión con frontend, validaciones |
| Semana 4 | Evaluación final del modelo, documentación, grabación del video, presentación |


