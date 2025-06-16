# Clasificación de condición de articulos en MercadoLibre

Este proyecto consiste en desarrollar un modelo de machine learning capaz de predecir si un producto de MercadoLibre es **nuevo** o **usado**.

# Descripción del problema

MercadoLibre recibe miles de publicaciones diariamente. La condición del producto (nuevo o usado) es una característica relevante tanto para compradores como para la experiencia general del sitio. La meta de este proyecto es construir un modelo que clasifique esta condición de forma precisa, a partir de datos como precio, envío, descripciones, y otras características.

# Datos

Los datos se encuentran en formato `.jsonlines` y fueron procesados mediante la función `build_dataset`. 

# Ingeniería de variables

Mediante un proceso minucioso se realizaron tranformaciones a las variables con la finalidad de que aportarán la mayor cantidad de información al modelo.

# Modelado

Se entrenaron modelos con **XGBoost**, evaluados principalmente con la métrica **F1-score**, ya que:

- El objetivo es clasificar correctamente ambas clases sin priorizar una sobre otra.
- F1 equilibra precisión y recall, ideal para tareas donde los errores en cualquier clase son igual de importantes.


## Hiperparámetros seleccionados

```python
{
    'n_estimators': 300,
    'max_depth': 5,
    'learning_rate': 0.12,
    'subsample': 0.7,
    'colsample_bytree': 0.8
}
```

# Evaluación final

| Métrica       | Modelo Final   |
|---------------|-----------|
| AUC ROC       | 0.8511    |
| Recall        | 0.8690    |
| Precisión     | 0.8598    |
| F1-score      | 0.8644    |



 