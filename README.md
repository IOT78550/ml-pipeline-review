# Предсказание стоимости поездки в сервисе такси

## Для запуска введите

Run main.py

"""
README.md практически пуст
Ниже представлено возможное наполнение с более верным запуском, созданием папки models для хранения моделей и метрик, дополнительными файлами и т.д.
"""

# Предсказание стоимости поездки в сервисе такси

Проект реализует пайплайн машинного обучения для прогнозирования стоимости поездки такси на основе исторических данных Uber. Модель использует признаки, включая дату и время начала поездки, расстояние и другие переменные, для точного предсказания fare_amount.

## Стек и функциональность

**Основные технологии:**
- Python 3.8–3.11
- pandas, numpy — обработка табличных данных
- scikit-learn:
  - GradientBoostingRegressor — основная модель регрессии
  - train_test_split — разбиение на train/test
  - GridSearchCV — подбор гиперпараметров

**Модульная архитектура:**
- `src/data.py` — загрузка и разбиение данных
- `src/features.py` — инженерия временных признаков
- `src/model.py` — обертка над моделью с метриками, сохранением и тюнингом
- `src/main.py` — основной training pipeline

## Структура проекта

```
.
├── src/
│   ├── __init__.py
│   ├── data.py          # load_data, split_data
│   ├── features.py      # add_time_features
│   ├── model.py         # TaxiFareModel: fit/predict/evaluate/tune/save/load
│   └── main.py          # полный цикл обучения и оценки
├── tests/
│   ├── __init__.py
│   └── test_data.py     # базовый тест загрузки данных
├── data/
│   └── uber.csv         # исходные данные
├── models/
│   ├── taxi_model.pkl   # обученная модель
│   ├── taxi_config.json # конфиг + метрики
│   └── metrics.json     # метрики отдельно
├── requirements-base.txt
├── requirements-dev.txt
├── .gitignore
└── README.md
```

**Примечание:** каталоги `data/` и `models/` создаются локально. Файл `uber.csv` должен быть размещён в `data/` перед запуском обучения.

## Установка окружения

### Создание и активация виртуального окружения

```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
```

### Установка зависимостей

Для разработки (код + тесты + линтеры):
```bash
pip install -r requirements-dev.txt
```

Для production-среды (только runtime-зависимости):
```bash
pip install -r requirements-base.txt
```

- `requirements-base.txt` содержит зависимости для запуска модели (pandas, numpy, scikit-learn и т.п.)
- `requirements-dev.txt` дополнительно включает pytest, black, flake8, mypy и другие инструменты разработки

## Функционал модулей по обработке данных

### Модуль src/data.py

- **load_data(path)** — загружает CSV, проверяет наличие обязательных столбцов (`fare_amount`, `pickup_datetime`) и что датасет не пуст (иначе возбуждает `ValueError`)

- **split_data(data, test_size=0.2, random_state=42)** — разделяет данные на train/test и отделяет таргет `fare_amount` от признаков

### Модуль src/features.py

- **add_time_features(df)** — конвертирует `pickup_datetime` в datetime, добавляет признаки `hour` (час) и `day_of_week` (день недели), удаляет исходный столбец `pickup_datetime`

## Запуск обучения

Основной скрипт — `src/main.py`.

### Базовый запуск обучения и оценки

```bash
python -m src.main
```

Пайплайн выполняет:

1. Загрузку данных из `data/uber.csv` через `load_data`
2. Добавление временных признаков через `add_time_features`
3. Разделение на train/test через `split_data` с фиксированным `random_state` для обеспечения воспроизводимости
4. Инициализацию `TaxiFareModel` с базовой конфигурацией `MODEL_CONFIG`
5. Обучение модели (если тюнинг отключен)
6. Оценку на тестовой выборке через `model.evaluate(X_test, y_test)`:
   - R² — коэффициент детерминации
   - RMSE — корень из среднеквадратичной ошибки
   - MAE — средняя абсолютная ошибка
7. Логирование этапов и метрик через стандартный модуль logging
8. Сохранение артефактов:
   - модель в `models/taxi_model.pkl`
   - конфиг и метрики в `models/taxi_config.json`
   - отдельно метрики в `models/metrics.json`

## Подбор гиперпараметров (GridSearchCV)

В `src/main.py` есть флаг:

```python
ENABLE_HYPERPARAM_TUNING = False
```

Если установить `ENABLE_HYPERPARAM_TUNING = True` и запустить `python -m src.main`:

1. Будет вызван метод `TaxiFareModel.tune_hyperparameters(X_train, y_train, ...)`, использующий GridSearchCV над GradientBoostingRegressor
2. Подберутся лучшие значения основных гиперпараметров (по умолчанию по метрике `neg_root_mean_squared_error`)
3. Лучший конфиг сохранится в `self.config`, модель будет переинициализирована этими параметрами и обучена на всем train
4. Далее выполняется оценка на тесте и сохранение артефактов, как в базовом сценарии

При необходимости можно скорректировать `param_grid` непосредственно в `tune_hyperparameters` или передавать его из `main.py`.

## Использование обученной модели

Пример применения модели для предсказаний в отдельном скрипте:

```python
import pandas as pd
from src.model import TaxiFareModel

model = TaxiFareModel.load("models/taxi_model.pkl", "models/taxi_config.json")

new_data = pd.DataFrame({
    "hour": [10, 15],
    "day_of_week": [1, 4],
})

predictions = model.predict(new_data)
print(predictions)
```

Метод `load` восстанавливает внутреннюю модель, конфиг и последние сохраненные метрики, если передан путь к JSON.

## Тестирование

В проекте реализованы тесты для проверки загрузки данных и корректности основных операций.

### Запуск тестов

```bash
pytest -v
```

## Команды для разработки

При установленном `requirements-dev.txt` доступны следующие команды:

### Автоформатирование кода

```bash
black src/ tests/
```

### Проверка стиля (PEP8)

```bash
flake8 src/ tests/
```

### Проверка типов

```bash
mypy src/
```

## Дополнительные рекомендации

### Инициализация пустых модулей

Рекомендуется создать пустые `__init__.py` файлы в папках `src/` и `tests/`:

```bash
touch src/__init__.py
touch tests/__init__.py
```

### Файл .gitignore

Также рекомендуется создать файл `.gitignore` в корне репозитория со следующим содержимым:

```
### Данные
data/
!data/.gitkeep

### Модели и артефакты
models/
*.pkl
*.joblib

### Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
env/
venv/
ENV/
.venv

### IDE
.vscode/
.idea/
*.swp
*.swo
*~

### pytest
.pytest_cache/
.coverage
htmlcov/

### mypy
.mypy_cache/
.dmypy.json
dmypy.json

### Jupyter
.ipynb_checkpoints/

### OS
.DS_Store
Thumbs.db
```

Это необходимо, чтобы не коммитить ненужные файлы и артефакты в репозиторий.
