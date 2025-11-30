import pandas as pd

"""
Необходимо импортрировать дополнительные используемые библиотеки,
перенести импорт train_test_split сюда, зафискировать RANDOM_STATE
и показатели (колонки) датасета в виде отдельных переменных:

from typing import Tuple
import pandas as pd
from sklearn.model_selection import train_test_split

RANDOM_STATE = 42
REQUIRED_COLUMNS = ['fare_amount', 'pickup_datetime']
"""

def load_data(path):
    """
    Следует обратить внимание на следующие места для улучшения:
    Отсутствует обработка ошибок:
    Если файл не найден или некорректен, функция вызовет 
    необработанное исключение
    
    Отсутсвие валидации данных:
    Функция не проверяет наличие обязательных столбцов 
    (fare_amount, pickup_datetime)
    
    Отсутствие документации:
    Нет docstring с описанием параметров, возвращаемого 
    значения и примеров
    """
    return pd.read_csv(path)
    """
    Так может выглядеть исправленный вариант с описанием (ниже):
    """
    """
    Загружает данные из CSV файла с валидацией.
    
    Args:
        path: Путь к CSV файлу
        
    Returns:
        pd.DataFrame: Загруженные данные
        
    Raises:
        FileNotFoundError: Если файл не найден
        ValueError: Если отсутствуют обязательные столбцы
        
    Example:
        >>> df = load_data('data/uber.csv')
        >>> assert 'fare_amount' in df.columns
    """
    """
    try:
        data = pd.read_csv(path)
    except FileNotFoundError:
        raise FileNotFoundError(f"Файл {path} не найден")
    
    # Проверка обязательных столбцов
    missing_cols = set(REQUIRED_COLUMNS) - set(data.columns)
    if missing_cols:
        raise ValueError(f"Отсутствуют обязательные столбцы: {missing_cols}")
    
    # Проверка на отсутствие данных
    if data.empty:
        raise ValueError("Датасет пуст")
    
    return data
    """

def split_data(data):
    """
    Используется импорт внутри функции:
    from sklearn.model_selection import train_test_split 
    находится внутри функции. Это затрудняет анализ 
    зависимостей и не соответствует стандарту PEP8
    
    Отсутствует random_state:
    Без фиксации random_state результаты сплитования (деления 
    данных на обучающую и тестовую выборки) не воспроизводимы
    
    Наличие зависимости от имени столбца:
    Код предполагает наличие столбца 'fare_amount', но 
    не проверяет это
    
    Отсутствие статической типизации:
    Функции не имеют type hints, что усложняет понимание и отладку
    """
    from sklearn.model_selection import train_test_split
    features = data.drop('fare_amount', axis=1)
    target = data['fare_amount']
    return train_test_split(features, target, test_size=0.2)
    """
    Так может выглядеть исправленный вариант с описанием (ниже):
    """
    """
def split_data(
    data: pd.DataFrame, 
    test_size: float = 0.2,
    random_state: int = RANDOM_STATE
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    """
    Разделяет данные на обучающую и тестовую выборки
    
    Args:
        data: Входные данные
        test_size: Доля тестового набора (default=0.2)
        random_state: Seed для воспроизводимости (default=42)
        
    Returns:
        Tuple[X_train, X_test, y_train, y_test]
        
    Note:
        random_state зафиксирован для гарантии воспроизводимости 
        результатов
    """
    """
    if 'fare_amount' not in data.columns:
        raise ValueError("Столбец 'fare_amount' не найден в данных")
    
    features = data.drop('fare_amount', axis=1)
    target = data['fare_amount']
    
    return train_test_split(
        features, 
        target, 
        test_size=test_size,
        random_state=random_state
    )
    """
