# test_model.py
import pandas as pd
from src.airline import load_data
from src.airline import create_sequences
import numpy as np
from sklearn.preprocessing import MinMaxScaler

def test_data_not_empty():
    df = load_data()
     # or mock small data
    assert not df.empty

# def test_datetime_conversion():
#     df = load_data()
#     # df['Month'] = pd.to_datetime(df['Month'])
#     assert pd.api.types.is_datetime64_any_dtype(df['Month'])


def test_sequence_shape():
    df = load_data()
    scaler = MinMaxScaler()
    data = scaler.fit_transform(df.values)
    data = np.arange(20).reshape(-1,1)
    X, y = create_sequences(data, seq_length=5)
    
    assert X.shape == (15, 5, 1)
    assert y.shape == (15, 1)