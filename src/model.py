from sklearn.ensemble import GradientBoostingRegressor
"""
Необходимо добавить дополнительные импорты библиотек
(будет ниже в исправленном варианте)
"""

class TaxiFareModel:
    def __init__(self):
    """
    Необходимо обратить внимание на следующие моменты:

    В модели отсутствует random_state, из-за чего
    результаты будут невоспроизводимы

    Отсутствует подбор/настройка гиперпараметров
    модели, что может привести к переобучению

    Отсутствует документация класса и методов
    """
        self.model = GradientBoostingRegressor()

    def fit(self, X, y):
    """
    Необходимо обратить внимание на следующие моменты:

    Отсутствует указание типов данных и документация,
    а также валидация входных данных
    """
        self.model.fit(X, y)

    def predict(self, X):
    """
    Необходимо обратить внимание на следующие моменты:

    Отсутствует указание типов данных и документация

    Отсутствует проверка на fit() - можно вызвать
    predict() до fit(), что приведет к ошибке
    """
        return self.model.predict(X)

"""
Исправленный вариант кода с описанием:
Модуль определения и обучения модели прогнозирования стоимости 
поездки такси

Содержит:
    - конфигурацию модели (MODEL_CONFIG)
    - класс TaxiFareModel:
        * обучение (fit)
        * предсказание (predict)
        * оценка качества (evaluate)
        * подбор гиперпараметров (tune_hyperparameters)
        * сохранение/загрузка модели и конфига (save / load)
"""
"""
from typing import Optional, Dict, Any

import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.model_selection import GridSearchCV

# Фиксируем random_state для воспроизводимости как на уровне сплита,
# так и на уровне самой модели
RANDOM_STATE: int = 42

# Базовая конфигурация гиперпараметров модели (baseline)
MODEL_CONFIG: Dict[str, Any] = {
    "n_estimators": 100,
    "learning_rate": 0.1,
    "max_depth": 5,
    "min_samples_split": 2,
    "min_samples_leaf": 1,
    "random_state": RANDOM_STATE,
    "subsample": 0.8,
}


class TaxiFareModel:
"""    
    """
    Модель прогнозирования стоимости поездки такси 
    на базе GradientBoostingRegressor

    Обертка над моделью scikit-learn обеспечивает:
    - единый конфиг гиперпараметров;
    - контроль обученности модели;
    - вычисление и сохранение метрик;
    - сохранение/загрузку артефактов;
    - подбор гиперпараметров через GridSearchCV

    Attributes:
    model: внутренняя модель scikit-learn
    is_fitted: флаг, указывающий, была ли модель обучена
    metrics: словарь с последними рассчитанными метриками
    config: конфигурация гиперпараметров, с которой
    инициализирована модель
    """
    """

    def __init__(self, 
                 config: Optional[Dict[str, Any]] = None) -> None:
    """
        """
        Инициализирует модель с заданной конфигурацией.

        Args:
        config: словарь с гиперпараметрами 
        GradientBoostingRegressor (если None, используется 
        базовый MODEL_CONFIG)

        Пример:
        >>> model = TaxiFareModel()
        >>> custom = {"n_estimators": 200, "max_depth": 3}
        >>> model2 = TaxiFareModel(config=custom)
        """
        """
        self.config: Dict[str, Any] = (config or MODEL_CONFIG).copy()
        # Гарантируем наличие random_state в конфиге
        self.config.setdefault("random_state", RANDOM_STATE)

        self.model: GradientBoostingRegressor = GradientBoostingRegressor(
            **self.config
        )
        self.is_fitted: bool = False
        self.metrics: Dict[str, float] = {}

    # Базовые методы: fit / predict / evaluate
    def fit(self, X: pd.DataFrame, y: pd.Series) -> None:
        """
        """
        Обучает модель на переданных данных

        Args:
        X: матрица признаков формы (n_samples, n_features)
        y: целевая переменная формы (n_samples,)

        Raises:
        ValueError: если данные пустые или размеры 
        X и y не совпадают
        """
        """
        if X.empty or y.empty:
            raise ValueError("Входные данные не могут быть пустыми")

        if len(X) != len(y):
            raise ValueError(
                f"X и y должны иметь одинаковое число строк "
                f"X: {len(X)}, y: {len(y)}"
            )

        self.model.fit(X, y)
        self.is_fitted = True

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        """
        Делает предсказания для новых данных

        Args:
        X: матрица признаков формы (n_samples, n_features)

        Returns:
        np.ndarray: вектор предсказаний формы (n_samples,)

        Raises:
        RuntimeError: если модель не была обучена
        """
        """
        if not self.is_fitted:
            raise RuntimeError(
                "Модель не обучена. Сначала вызовите fit(X_train, y_train)"
            )

        return self.model.predict(X)

    def evaluate(self, 
                 X: pd.DataFrame, 
                 y: pd.Series) -> Dict[str, float]:
        """
        """
        Оценивает качество модели на заданных данных

        Вычисляет метрики:
        - r2: коэффициент детерминации;
        - rmse: корень из среднеквадратичной ошибки;
        - mae: средняя абсолютная ошибка

        Args:
        X: матрица признаков
        y: вектор истинных значений целевой переменной

        Returns:
        dict: словарь с метриками {'r2': ..., 'rmse': ..., 'mae': ...}.

        Raises:
        RuntimeError: если модель не была обучена
        """
        """
        if not self.is_fitted:
            raise RuntimeError("Модель не обучена")

        y_pred = self.predict(X)

        self.metrics = {
            "r2": float(r2_score(y, y_pred)),
            "rmse": float(np.sqrt(mean_squared_error(y, y_pred))),
            "mae": float(mean_absolute_error(y, y_pred)),
        }

        return self.metrics

    # Подбор гиперпараметров через GridSearchCV
    def tune_hyperparameters(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        param_grid: Optional[Dict[str, list]] = None,
        cv: int = 3,
        n_jobs: int = -1,
        scoring: str = "neg_root_mean_squared_error",
        verbose: int = 0,
    ) -> Dict[str, Any]:
        """
        """
        Подбирает гиперпараметры модели с помощью 
        GridSearchCV и фиксирует лучший конфиг

        Args:
        X: матрица признаков (n_samples, n_features)
        y: целевая переменная (n_samples,)
        param_grid: словарь с сеткой гиперпараметров для перебора
        (если None, используется дефолтная небольшая сетка)
        cv: число фолдов кросс-валидации
        n_jobs: число параллельных потоков (-1 = все ядра)
        scoring: метрика для оптимизации 
        (по умолчанию минимизируем RMSE)
        verbose: уровень логирования GridSearchCV

        Returns:
        dict: лучшие найденные гиперпараметры (best_params_)

        Side effects:
        - обновляет self.config лучшими параметрами;
        - переинициализирует self.model с лучшим конфигом;
        - обучает модель на всех данных X, y с этим конфигом;
        - помечает модель как обученную (is_fitted = True)
        """
        """
        if X.empty or y.empty:
            raise ValueError("Входные данные не могут быть пустыми")

        if len(X) != len(y):
            raise ValueError(
                f"X и y должны иметь одинаковое число строк. "
                f"X: {len(X)}, y: {len(y)}"
            )

        if param_grid is None:
            # Небольшая, но адекватная сетка по ключевым гиперпараметрам
            param_grid = {
                "n_estimators": [100, 200],
                "learning_rate": [0.05, 0.1],
                "max_depth": [3, 5],
                "subsample": [0.8, 1.0],
            }

        base_estimator = GradientBoostingRegressor(
            random_state=RANDOM_STATE
        )

        grid_search = GridSearchCV(
            estimator=base_estimator,
            param_grid=param_grid,
            cv=cv,
            n_jobs=n_jobs,
            scoring=scoring,
            verbose=verbose,
        )

        grid_search.fit(X, y)

        # Используем best_params_ напрямую (не get_params())
        best_params: Dict[str, Any] = grid_search.best_estimator_.copy()
        # На всякий случай явно фиксируем random_state
        best_params["random_state"] = RANDOM_STATE

        # Обновляем конфиг и модель лучшими параметрами
        self.config = best_params
        self.model = GradientBoostingRegressor(**self.config)
        self.model.fit(X, y)
        self.is_fitted = True

        return best_params

    # Сохранение и загрузка модели/конфига/метрик
    def save(self, 
             model_path: str, 
             config_path: Optional[str] = None) -> None:
        """
        """
        Сохраняет обученную модель и (опционально) 
        конфиг с метриками

        Args:
        model_path: Путь для сохранения модели (.pkl)
        config_path: Путь для сохранения конфига и метрик (.json)
        """
        """
        model_file = Path(model_path)
        model_file.parent.mkdir(parents=True, exist_ok=True)

        with open(model_file, "wb") as f:
            pickle.dump(self.model, f)

        if config_path:
            config_file = Path(config_path)
            config_file.parent.mkdir(parents=True, 
                                     exist_ok=True)

            config_data = {
                "config": self.config,
                "metrics": self.metrics,
                "random_state": RANDOM_STATE,
                "model_path": str(model_file),
            }
            with open(config_file, "w", encoding="utf-8") as f:
                json.dump(config_data, f, indent=2, ensure_ascii=False)

    @classmethod
    def load(cls, model_path: str,
             config_path: Optional[str] = None) -> "TaxiFareModel":
        """
        """
        Загружает обученную модель с конфигом и метриками

        Args:
        model_path: путь к сохраненной модели (.pkl)
        config_path: путь к сохраненному конфигу (.json)
        (последнее опционально)

        Returns:
        TaxiFareModel: экземпляр с загруженной внутренней моделью, 
        восстановленным конфигом и метриками

        Raises:
        FileNotFoundError: если файл модели не найден
        """
        """
        model_file = Path(model_path)
        if not model_file.exists():
            raise FileNotFoundError(f"Файл модели не найден: {model_path}")
        
        config = None
        metrics = {}

        # Восстанавливаем конфиг и метрики из JSON, если файл передан
        if config_path:
            config_file = Path(config_path)
            if config_file.exists():
                with open(config_file, "r", encoding="utf-8") as f:
                    config_data = json.load(f)
                    config = config_data.get("config")
                    metrics = config_data.get("metrics", {})

        # Инициализируем экземпляр с загруженным конфигом
        instance = cls(config=config)

        # Загружаем сохраненную модель
        with open(model_file, "rb") as f:
            instance.model = pickle.load(f)

        # Помечаем модель как обученную и восстанавливаем метрики
        instance.is_fitted = True
        instance.metrics = metrics

        return instance
        """
