# Cómo Calcular las Métricas Reales del Modelo

Este documento explica cómo calcular las métricas reales de tu modelo de detección de spam y subirlas al sistema.

## Paso 1: Calcular Métricas con tu Modelo Entrenado

Usa el siguiente código Python después de entrenar tu modelo:

```python
import joblib
import json
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score

# Cargar tu modelo y vectorizador entrenados
modelo = joblib.load('modelo_spam.joblib')
vectorizador = joblib.load('vectorizador.joblib')

# Cargar tu conjunto de prueba (X_test, y_test)
# X_test debe contener los correos de prueba procesados
# y_test debe contener las etiquetas reales (spam/ham)

# Hacer predicciones
y_pred = modelo.predict(X_test)

# Calcular matriz de confusión
# Nota: ajusta pos_label según cómo estén codificadas tus etiquetas
# Si usas 'spam' y 'ham' como strings, usa pos_label='spam'
# Si usas 1 y 0, usa pos_label=1
tn, fp, fn, tp = confusion_matrix(y_test, y_pred, labels=['ham', 'spam']).ravel()

# Calcular métricas de rendimiento
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, pos_label='spam')
recall = recall_score(y_test, y_pred, pos_label='spam')
f1 = f1_score(y_test, y_pred, pos_label='spam')

# Calcular especificidad (tasa de verdaderos negativos)
specificity = tn / (tn + fp) if (tn + fp) > 0 else 0

# Crear el diccionario de métricas
metricas = {
    "confusion_matrix": {
        "true_negative": int(tn),
        "false_positive": int(fp),
        "false_negative": int(fn),
        "true_positive": int(tp)
    },
    "performance_metrics": {
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1_score": float(f1),
        "specificity": float(specificity)
    }
}

# Guardar en archivo JSON
with open('modelo_metricas.json', 'w', encoding='utf-8') as f:
    json.dump(metricas, f, indent=2, ensure_ascii=False)

print("✅ Métricas calculadas y guardadas en modelo_metricas.json")
print(f"\nAccuracy: {accuracy:.2%}")
print(f"Precision: {precision:.2%}")
print(f"Recall: {recall:.2%}")
print(f"F1-Score: {f1:.2%}")
print(f"Specificity: {specificity:.2%}")
```

## Paso 2: Subir el Archivo de Métricas

1. Ejecuta el código anterior para generar `modelo_metricas.json`
2. Copia el archivo `modelo_metricas.json` a la carpeta `backend/static/`
3. Sube los cambios a GitHub:
   ```bash
   git add backend/static/modelo_metricas.json
   git commit -m "Agregar métricas reales del modelo"
   git push origin main
   ```
4. Render automáticamente desplegará la nueva versión con las métricas reales

## Paso 3: Verificar

1. Visita tu aplicación en Vercel
2. Haz clic en "Ver Rendimiento del Modelo"
3. Verifica que las métricas mostradas corresponden a las calculadas

## Ejemplo de archivo modelo_metricas.json

```json
{
  "confusion_matrix": {
    "true_negative": 850,
    "false_positive": 45,
    "false_negative": 32,
    "true_positive": 873
  },
  "performance_metrics": {
    "accuracy": 0.9572,
    "precision": 0.9510,
    "recall": 0.9646,
    "f1_score": 0.9578,
    "specificity": 0.9497
  }
}
```

## Notas Importantes

- **Las etiquetas deben coincidir**: Asegúrate de que las etiquetas en tu modelo ('spam'/'ham' o 1/0) coincidan con las usadas en el cálculo de métricas
- **Usa el conjunto de prueba**: Nunca calcules métricas con el conjunto de entrenamiento, siempre usa datos que el modelo no haya visto durante el entrenamiento
- **Actualiza periódicamente**: Si reentrenas el modelo, recalcula y actualiza las métricas

## Interpretación de las Métricas

- **Accuracy**: Proporción de predicciones correctas (spam y ham)
- **Precision**: De los correos marcados como spam, cuántos realmente lo son
- **Recall**: De todos los spam reales, cuántos fueron detectados
- **F1-Score**: Balance entre precisión y recall
- **Specificity**: De todos los correos legítimos, cuántos fueron identificados correctamente como ham
