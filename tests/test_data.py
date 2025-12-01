import pandas as pd
from src.data import load_data

def test_load_data():
    df = load_data("data/uber.csv")
    assert not df.empty
"""
Тест достаточно минималистичен, проверяет,
что DataFrame не пустой. Можно расширить

Присутствует жесткая привязка к пути
"data/uber.csv"

Отсутствует проверка структуры данных,
наличия обязательных столбцов

Не тестируются другие функции, например,
split_data(), add_time_features() и др.
"""


"""
Ниже представлен исправленный код
(запуск: pytest tests/test_data.py -v
        pytest tests/test_data.py -v --cov=src.data)
"""
import pytest
import pandas as pd
import numpy as np
from pathlib import Path
from src.data import load_data, split_data, REQUIRED_COLUMNS, RANDOM_STATE

@pytest.fixture
def sample_data():
    """
    Фикстура: создание тестовых данных
    """
    return pd.DataFrame({
        'fare_amount': [10.5, 15.0, 12.3],
        'pickup_datetime': ['2020-01-01 10:00:00', '2020-01-01 11:00:00', '2020-01-01 12:00:00'],
        'distance': [1.2, 2.5, 1.8]
    })

@pytest.fixture
def temp_csv(tmp_path, sample_data):
    """
    Фикстура: сохранение тестовых данных в CSV
    """
    csv_file = tmp_path / "test_data.csv"
    sample_data.to_csv(csv_file, index=False)
    return str(csv_file)

class TestLoadData:
    """
    Тесты для функции load_data()
    """
    
    def test_load_data_success(self, temp_csv):
        """
        Проверка успешной загрузки данных
        """
        df = load_data(temp_csv)
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 3
    
    def test_load_data_file_not_found(self):
        """
        Проверка обработки несуществующего файла
        """
        with pytest.raises(FileNotFoundError):
            load_data("nonexistent_file.csv")
    
    def test_load_data_required_columns(self, temp_csv):
        """
        Проверка наличия обязательных столбцов
        """
        df = load_data(temp_csv)
        for col in REQUIRED_COLUMNS:
            assert col in df.columns, f"Столбец {col} не найден"
    
    def test_load_data_empty_dataframe(self, tmp_path):
        """
        Проверка обработки пустого файла
        """
        empty_csv = tmp_path / "empty.csv"
        pd.DataFrame().to_csv(empty_csv, index=False)
        
        with pytest.raises(ValueError, match="Датасет пуст"):
            load_data(str(empty_csv))

class TestSplitData:
    """
    Тесты для функции split_data()
    """
    
    def test_split_data_reproducibility(self, sample_data):
        """
        Проверка воспроизводимости (одинаковый random_state)
        """
        X_train1, X_test1, y_train1, y_test1 = split_data(sample_data)
        X_train2, X_test2, y_train2, y_test2 = split_data(sample_data)
        
        pd.testing.assert_frame_equal(X_train1, X_train2)
        pd.testing.assert_series_equal(y_train1, y_train2)
    
    def test_split_data_proportions(self, sample_data):
        """
        Проверка правильного разделения данных
        """
        X_train, X_test, y_train, y_test = split_data(
            sample_data, 
            test_size=0.2,
            random_state=RANDOM_STATE
        )
        
        assert len(X_train) + len(X_test) == len(sample_data)
        assert len(X_test) / len(sample_data) == pytest.approx(0.2, abs=0.1)
    
    def test_split_data_no_fare_column(self):
        """
        Проверка обработки отсутствующего столбца fare_amount
        """
        bad_data = pd.DataFrame({'other_col': [1, 2, 3]})
        
        with pytest.raises(ValueError, match="fare_amount"):
            split_data(bad_data)
    
    def test_split_data_contains_target(self, sample_data):
        """
        Проверка, что целевая переменная исключена из features
        """
        X_train, X_test, y_train, y_test = split_data(sample_data)
        
        assert 'fare_amount' not in X_train.columns
        assert 'fare_amount' not in X_test.columns
