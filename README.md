# Kaggle Competition: Прогнозирование отзывов игр Steam

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![PyTorch](https://img.shields.io/badge/Deep%20Learning-PyTorch-red)
![Competition](https://img.shields.io/badge/Platform-Kaggle-orange)
![Status](https://img.shields.io/badge/Status-Completed-success)

## Описание проекта

Соревнование Kaggle по прогнозированию количества положительных и отрицательных отзывов игр на платформе Steam на основе метаданных и описаний игр. Задача регрессии с двумерным таргетом.

**Цель:** Построить модель, которая предсказывает количество положительных (`Positive`) и отрицательных (`Negative`) отзывов для игр Steam.

**Метрика:** Mean Absolute Error (MAE)

**Сложность:** Мультимодальные данные - числовые признаки, бинарные флаги и текстовые описания.

## Ключевые результаты

- **Реализована нейросетевая архитектура** на PyTorch для многомерной регрессии
- **Объединены разнородные данные:** числовые признаки, бинарные флаги, TF-IDF векторы описаний
- **Достигнута стабильная сходимость** с валидационным MAE ~180
- **Создан end-to-end пайплайн** от загрузки данных до генерации submission файла

## Технологический стек

- **Язык программирования:** Python 3.8+
- **Глубокое обучение:** PyTorch, torch.nn
- **Обработка данных:** pandas, numpy, scikit-learn
- **Обработка текста:** TF-IDF Vectorizer, stopwords removal
- **Предобработка:** StandardScaler, обработка пропусков
- **Инфраструктура:** Google Colab с GPU T4
