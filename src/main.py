import os
import pandas as pd
from src.data import load_data, split_data
from src.features import add_time_features
from src.model import TaxiFareModel

# Глобальная переменная
DATA_PATH = "data/main.csv"
"""
В данном месте неверно указан путь к данным,
используется uber.csv, а не main.csv
"""

# Загрузка и обработка данных
raw_data = load_data(DATA_PATH)
processed_data = add_time_features(raw_data)
X_train, X_test, y_train, y_test = split_data(processed_data)

# Обучение модели
model = TaxiFareModel()
model.fit(X_train, y_train)

# Оценка
score = model.model.score(X_test, y_test)
print(f"R²: {score:.2f}")
"""
Прямой доступ к .model, лучше использовать метод
evaluate().
R^2 (коэффициент детерминации) не всегда информативен 
для моделей, работающих с другими функциями потерь 
(например, MAE вместо MSE), либо при наличии сильной 
несбалансированности классов отражает только долю дисперсии, 
объясненной моделью. При этом классический R^2
(не скорректированный, Adj. R^2), как правило, растет
при увеличении числа объясняемых переменных, хотя
в действительности качество прогноза может снижаться.
Для большей объективности лучше использовать другие
метрики качества прогноза модели, а еще лучше проводить
сравнение сразу по нескольким метрикам:
- абсолютным - MAE и MSE (RMSE),
- относительным, например SMAPE (при этом не подходит для
случаев, если наблюдаемое или предсказанное значение равно 0,
ошибка резко возрастет до верхнего предела (200% или 100%);
также метрика нестабильна, когда истинное значение и прогноз
очень близки к нулю — в этом случае происходит деление
на число, очень близкое к нулю)
"""
"""
Ниже представлен исправленный вариант кода
с добавлением логирования
"""
"""
Основной скрипт для обучения и оценки модели 
прогнозирования стоимости такси:
- загрузка данных
- инженерия признаков
- разделение на train/test
- обучение модели
- оценка и сохранение результатов
"""
"""
import json
import logging
from pathlib import Path
import pandas as pd

from src.data import load_data, split_data
from src.features import add_time_features
from src.model import TaxiFareModel

# Конфигурация логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(name)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Пути и константы
DATA_PATH = "data/uber.csv"  # Исправлено на uber.csv
OUTPUT_DIR = Path("models")
MODEL_PATH = OUTPUT_DIR / "taxi_model.pkl"
CONFIG_PATH = OUTPUT_DIR / "taxi_config.json"
METRICS_PATH = OUTPUT_DIR / "metrics.json"

# Включать ли подбор гиперпараметров (GridSearchCV)
ENABLE_HYPERPARAM_TUNING = False

def main() -> tuple[TaxiFareModel, dict]:
    """
    """
    Главная функция для обучения и оценки модели.

    Returns:
        (model, metrics):
            model  — обученный экземпляр TaxiFareModel
            metrics — словарь с метриками на тестовой выборке
    """
    """
    # Создание директории для моделей
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    logger.info("Директория для моделей: %s", OUTPUT_DIR)

    # Загрузка данных
    logger.info("Загрузка данных из %s", DATA_PATH)
    try:
        raw_data = load_data(DATA_PATH)
    except Exception as exc:
        logger.exception("Ошибка при загрузке данных: %s", exc)
        raise
        
    logger.info(
        "Данные загружены: %d строк, %d столбцов",
        len(raw_data),
        len(raw_data.columns),
    )

    # Инженерия признаков
    logger.info("Добавление временных признаков")
    try:
        processed_data = add_time_features(raw_data)
    except Exception as exc:
        logger.exception("Ошибка при добавлении признаков: %s", exc)
        raise

    logger.info(
        "После инженерии признаков: %d строк, %d столбцов",
        len(processed_data),
        len(processed_data.columns),
    )

    # Разделение на train/test
    logger.info("Разделение данных на train/test")
    try:
        X_train, X_test, y_train, y_test = split_data(processed_data)
    except Exception as exc:
        logger.exception("Ошибка при разделении данных: %s", exc)
        raise

    logger.info(
        "Train: %d образцов, Test: %d образцов",
        len(X_train),
        len(X_test),
    )

    # Инициализация модели
    model = TaxiFareModel()
    logger.info("Модель TaxiFareModel инициализирована с базовым конфигом")

    # Подбор гиперпараметров (опционально)
    if ENABLE_HYPERPARAM_TUNING:
        logger.info("Старт подбора гиперпараметров (GridSearchCV)")
        try:
            best_params = model.tune_hyperparameters(
                X_train,
                y_train,
                # Можно передать свой param_grid, если нужно
                # param_grid={
                #     "n_estimators": [100, 200, 300],
                #     "learning_rate": [0.05, 0.1],
                #     "max_depth": [3, 5, 7],
                #     "subsample": [0.8, 1.0],
                # },
                cv=3,
                n_jobs=-1,
                scoring="neg_root_mean_squared_error",
                verbose=1,
            )
            logger.info("Лучшие параметры GridSearchCV: %s", best_params)
        except Exception as exc:
            logger.exception("Ошибка при подборе гиперпараметров: %s", exc)
            raise
    else:
        # Обычное обучение без подбора
        logger.info("Обучение модели без подбора гиперпараметров")
        try:
            model.fit(X_train, y_train)
        except Exception as exc:
            logger.exception("Ошибка при обучении модели: %s", exc)
            raise

    # Оценка модели на тестовой выборке
    logger.info("Оценка качества модели на тестовом наборе")
    try:
        metrics = model.evaluate(X_test, y_test)
    except Exception as exc:
        logger.exception("Ошибка при оценке модели: %s", exc)
        raise

    logger.info("Метрики качества на тесте:")
    logger.info("  R²   : %.4f", metrics["r2"])
    logger.info("  RMSE : %.4f", metrics["rmse"])
    logger.info("  MAE  : %.4f", metrics["mae"])

    # Сохранение артефактов (модель + конфиг + метрики)
    logger.info("Сохранение модели в %s", MODEL_PATH)
    try:
        model.save(str(MODEL_PATH), str(CONFIG_PATH))
    except Exception as exc:
        logger.exception("Ошибка при сохранении модели/конфига: %s", exc)
        raise
        
    logger.info("Сохранение метрик в %s", METRICS_PATH)
    METRICS_PATH.write_text(json.dumps(metrics, 
                                       indent=2, 
                                       ensure_ascii=False), 
                            encoding="utf-8")

    logger.info("Обучение и оценка завершены успешно")
    return model, metrics


if __name__ == "__main__":
    main()
    """
