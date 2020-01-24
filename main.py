import numpy as np
import pandas as pd

from time_series_transformer.pipeline import create_pipeline



NOT_IND_VARS_COLS = ['ticker', 'exchange', 'name', 'industry', "date"]

NOT_SCALE_COLS = ['ticker', 'exchange', 'name', 'industry', "date", "adj_close"]


# LAG_COLS = ["adj_close", 'adj_open', 'adj_low', 'adj_high', "volume", 'ema', 'macd']
#
# NOT_IND_VARS_COLS = ['ticker', "date"]
#
# NOT_SCALE_COLS = ['ticker', "date", "adj_close"]

all_stocks = pd.read_csv("/home/hristocr/Desktop/tech_ind_time_series.csv")

stocks = all_stocks[all_stocks["ticker"].isin(["AAPL", "GS"])]
stocks["date"] = stocks["date"].astype(np.datetime64)

# test_data = stocks[
# 	["ticker", "date", "sector", "adj_close", 'adj_open', 'adj_low', 'adj_high', "volume", 'ema', 'macd']]

pipeline = create_pipeline(group_col="ticker", target_col="adj_close",
                           create_lags=True, lag_cols=LAG_COLS, lag_num=5, drop_lag_col=True,
                           create_ind_vars=True, not_ind_vars_cols=NOT_IND_VARS_COLS,
                           scale=True, not_scale_cols=NOT_SCALE_COLS)

dataset = pipeline.fit_transform(stocks)

dataset.to_csv("/home/hristocr/Desktop/dataset.csv", index=False)

dataset = dataset.set_index(["ticker", "date"])



df = pd.DataFrame(np.random.random((2, 6)))
df.columns = pd.MultiIndex.from_product([["lag_1", "lag_2", "lag_3"], ['feature_1', 'feature_2']], names=["lags", "features"])
df.index.name = "samples"
df


df1 = pd.DataFrame(np.random.random((2,1)))
df1

df2 = pd.concat([df,df1], axis=1)
df2.columns
