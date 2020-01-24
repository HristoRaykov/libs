import numpy as np
import pandas as pd

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

from ts_train_test_split.train_test_split import split_time_series_by_time_steps
from ts_transformer.pipeline import create_pipeline

LAG_COLS = ['adj_close', 'volume', 'adj_open', 'adj_low',
            'adj_high', 'ema', 'macd', 'macd_diff', 'macd_signal', 'psar',
            'psar_up_ind', 'psar_down_ind', 'cci', 'stoch_osc', 'stoch_osc_signal',
            'rsi', 'roc', 'bb_ma', 'bb_hb', 'bb_hb_ind', 'bb_lb', 'bb_lb_ind',
            'atr', 'std', 'cmf', 'obv', 'vwap']

COLS_TO_DROP = ['open', 'close', 'low', 'high', "industry", "name", "exchange"]

TIME_INDEPENDENT_COLS = []

GROUP_COL = "ticker"
DATE_COL = "date"
TARGET_COL = "adj_close"

LAG_SUFFIX_KW = "_lag_"
LAG_DIFF_SUFFIX_KW = "_lag_diff_"

stocks = pd.read_csv("/home/hristocr/Desktop/dataset.csv")
stocks["date"] = stocks["date"].astype(np.datetime64)

pipeline = create_pipeline(group_col=GROUP_COL, date_col=DATE_COL, target_col=TARGET_COL,
                           create_lags=True, lag_cols=LAG_COLS, lag_num=5, drop_lag_col=True,
                           trans_index=True,
                           create_ind_vars=True,
                           time_step_trans=True)

df = pipeline.fit_transform(stocks)

train, test = split_time_series_by_time_steps(df, n_time_steps=5)
