import pandas as pd
from datetime import datetime

"""
Импорт datetime не используется, рекомендуется его убрать
"""

def add_time_features(df):
    df['pickup_datetime'] = pd.to_datetime(df['pickup_datetime'])
    df['hour'] = df['pickup_datetime'].dt.hour
    df['day_of_week'] = df['pickup_datetime'].dt.dayofweek
    return df.drop('pickup_datetime', axis=1)
"""
Функция add_time_features модифицирует датасет напрямую,
без создания копии, что не является хорошей практикой

Рекомендуется добавить обработку ошибок (в частности, проверку
формата) для 'pickup_datetime'

Также следует добавить в функцию аннотацию типов данных
и написать про работу функции

Ниже представлен исправленный вариант:
"""
"""
import pandas as pd
from typing import List

def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
"""
"""
Добавляет временные признаки из столбца 'pickup_datetime'.
    
    Создает новые столбцы:
    - hour: час суток (0-23)
    - day_of_week: день недели (0-6)
    
    Args:
        df: входной DataFrame (не модифицируется)
        
    Returns:
        pd.DataFrame: новый DataFrame с добавленными признаками
        
    Raises:
        KeyError: если столбец 'pickup_datetime' отсутствует
        ValueError: если невозможно распарсить дату
        
    Example:
        >>> df = pd.DataFrame({'pickup_datetime': ['2020-01-01 10:30:00']})
        >>> result = add_time_features(df)
        >>> assert 'hour' in result.columns
"""
"""
# Создание копии для избежания модификации исходных данных
    df_copy = df.copy()
    
    if 'pickup_datetime' not in df_copy.columns:
        raise KeyError("Столбец 'pickup_datetime' не найден")
    
    try:
        df_copy['pickup_datetime'] = pd.to_datetime(df_copy['pickup_datetime'])
    except Exception as e:
        raise ValueError(f"Ошибка при парсинге даты: {e}")
    
    # Добавление временных признаков
    df_copy['hour'] = df_copy['pickup_datetime'].dt.hour
    df_copy['day_of_week'] = df_copy['pickup_datetime'].dt.dayofweek
    
    # Удаление исходного столбца (больше не нужен)
    return df_copy.drop('pickup_datetime', axis=1)

def get_feature_names() -> List[str]:
    """
    Возвращает список имен генерируемых признаков, 
    что полезно для документации и отладки
    """
    return ['hour', 'day_of_week']
"""
