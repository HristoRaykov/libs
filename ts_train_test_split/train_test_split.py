def split_time_series_by_time_steps(df, n_time_steps=None):
	"""
	df index is Multiindex with level_0='stock tickers' and level_1='dates'
	"""
	
	time_steps = df.index.get_level_values(1).unique()
	split_time_step = time_steps[-5]
	train = df[df.index.get_level_values(1) < split_time_step]
	test = df[df.index.get_level_values(1) >= split_time_step]
	
	return train, test
