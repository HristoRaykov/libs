import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

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

# all_stocks = pd.read_csv("/home/hristocr/Desktop/tech_ind_time_series.csv")
#
# stocks = all_stocks[all_stocks["ticker"].isin(["AAPL", "GS"])]
# stocks["date"] = stocks["date"].astype(np.datetime64)
# stocks = stocks.drop(COLS_TO_DROP, axis=1)
# stocks.to_csv("/home/hristocr/Desktop/dataset.csv", index=False)

stocks = pd.read_csv("/home/hristocr/Desktop/dataset.csv")
stocks["date"] = stocks["date"].astype(np.datetime64)

pipeline = create_pipeline(group_col=GROUP_COL, date_col=DATE_COL, target_col=TARGET_COL,
                           create_lags=True, lag_cols=LAG_COLS, lag_num=5, drop_lag_col=True,
                           trans_index=True,
                           create_ind_vars=True,
                           time_step_trans=True)

df = pipeline.fit_transform(stocks)

scaler = pipeline.named_steps["scale"]
reversed = scaler.inverse_transform(df)

sc = MinMaxScaler()

y = stocks["adj_close"]
y_scaled = sc.fit_transform(y.values.reshape(-1, 1))
y_rev = sc.inverse_transform(y_scaled)

targets = df[[("lag_0", TARGET_COL)]]
df1 = df.drop([("lag_0", TARGET_COL)], axis=1)


r = scaler.inverse_transform(df)

level1 = df.columns.get_level_values(1)

x = df.loc[:, df.columns.get_level_values(1) == "adj_close"]

repeating = pd.Series(level1)
repeating.value_counts()

desc = df.describe()
