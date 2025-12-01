# Предсказание стоимости поездки в сервисе такси

## Для запуска введите

Run main.py

"""
README.md практически пуст
Ниже представлено возможное наполнение с более верным запуском
"""

# Предсказание стоимости поездки в сервисе такси
Проект реализует пайплайн машинного обучения для прогнозирования 
стоимости поездки такси на основе исторических данных Uber
(дата и время начала поездки, расстояние и другие признаки)

## Стек и функциональность
- Python 3.8–3.11
- pandas, numpy для работы с табличными данными
- scikit-learn для:
  - GradientBoostingRegressor - модель регрессии
  - train_test_split - разбиение на train/test
  - GridSearchCV - подбор гиперпараметров
- Модульная архитектура:
  - src/data.py - загрузка и разбиение данных
  - src/features.py - инженерия временных признаков
  - src/model.py - обертка над моделью с метриками, сохранением и тюнингом​
  - src/main.py - основной training pipeline

## Структура проекта
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
Каталоги data/ и models/ создаются локально; 
файл uber.csv должен быть размещён в data/ перед запуском обучения

## Установка окружения
Создание виртуального окружения и установка зависимостей:

python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

- Для разработки (код + тесты + линтеры):
pip install -r requirements-dev.txt

- Для production-среды (только runtime-зависимости):
pip install -r requirements-base.txt

requirements-base.txt содержит зависимости для запуска модели 
(pandas, numpy, scikit-learn и т.п.), а requirements-dev.txt 
дополнительно включает pytest, black, flake8, mypy
и другие инструменты разработки

## Функционал модулей по обработке данных
Модуль src/data.py:
- load_data(path) загружает CSV, проверяет наличие обязательных столбцов
  (fare_amount, pickup_datetime) и что датасет не пуст (иначе ValueError)

- split_data(data, test_size=0.2, random_state=42) разделяет данные
  на train/test и отделяет таргет fare_amount от признаков

Модуль src/features.py:

- add_time_features(df) конвертирует pickup_datetime в datetime;
  добавляет признаки hour (час) и day_of_week (день недели);
  удаляет исходный столбец pickup_datetime

## Запуск обучения
Основной скрипт — src/main.py.

Базовый запуск обучения и оценки:
python -m src.main

Пайплайн выполняет:
- загрузку данных из data/uber.csv через load_data
- добавление временных признаков через add_time_features
- разделение на train/test через split_data с фиксированным
  random_state, что обеспечивает воспроизводимость разбиения
- инициализацию TaxiFareModel с базовой конфигурацией MODEL_CONFIG
- обучение модели (если тюнинг отключен)
- оценку на тестовой выборке с помощью model.evaluate(X_test, y_test):
  - R^2 - коэффициент детерминации
  - rmse - корень из среднеквадратичной ошибки
  - mae - средняя абсолютная ошибка.
- логирование этапов и метрик через стандартный модуль logging​
- сохранение артефактов:
  - модель в models/taxi_model.pkl
  - конфиг и метрики в models/taxi_config.json
  - отдельно метрики в models/metrics.json.

## Подбор гиперпараметров (GridSearchCV)
В src/main.py есть флаг:​
ENABLE_HYPERPARAM_TUNING = False

Если установить:
ENABLE_HYPERPARAM_TUNING = True

то при запуске python -m src.main:
- будет вызван метод
  TaxiFareModel.tune_hyperparameters(X_train, y_train, ...),
  использующий GridSearchCV над GradientBoostingRegressor
- подберутся лучшие значения основных гиперпараметров
  (по умолчанию по метрике neg_root_mean_squared_error)
- лучший конфиг сохранится в self.config, модель будет
  переинициализирована этими параметрами и обучена на всем train
- далее выполняется оценка на тесте и сохранение артефактов,
  как в базовом сценарии

При необходимости можно скорректировать param_grid непосредственно 
в tune_hyperparameters или передавать его из main.py

## Использование обученной модели
Пример применения модели для предсказаний в отдельном скрипте:

import pandas as pd
from src.model import TaxiFareModel

model = TaxiFareModel.load("models/taxi_model.pkl", "models/taxi_config.json")

new_data = pd.DataFrame({
    "hour": [10, 15],
    "day_of_week": [1, 4],
})

predictions = model.predict(new_data)
print(predictions)

Метод load восстанавливает внутреннюю модель, конфиг и последние 
сохраненные метрики, если передан путь к JSON.

## Тестирование
В проекте уже был базовый тест tests/test_data.py, который проверяет, 
что load_data("data/uber.csv") возвращает непустой датасет.
Реализованы также дополнительные тесты. 

Запуск тестов:

pytest -v

## Команды для разработки
При установленном requirements-dev.txt доступны следующие команды:

### Автоформатирование кода
black src/ tests/

### Проверка стиля (PEP8)
flake8 src/ tests/

### Проверка типов
mypy src/

## Запуск всего training pipeline:
python -m src.main


## P.S. Некоторые дополнения
Рекомендуется создать пустые __init__.py файлы
в папке src/ и tests/:

touch src/__init__.py
touch tests/__init__.py

Также рекомендуется создать файл .gitignore в корне репозитория
(рядом с README.md) со следующим содержимым:
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

Это нужно, чтобы не коммитить ненужные файлы
