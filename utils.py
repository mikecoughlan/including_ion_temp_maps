import gc
import glob
import math
import os
import pickle
from datetime import datetime
from functools import partial
from multiprocessing import Manager, Pool

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
# import shapely
from dateutil import parser
# from geopack import geopack, t89
from matplotlib.cm import ScalarMappable
from matplotlib.collections import PatchCollection
from matplotlib.colors import Normalize
from matplotlib.patches import Circle, Wedge
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler, OneHotEncoder
from spacepy import pycdf
from tqdm import tqdm

pd.options.mode.chained_assignment = None

os.environ["CDF_LIB"] = "~/CDF/lib"

data_dir = '../../../../data/'
twins_dir = '../data/twins/'
supermag_dir = data_dir+'supermag/feather_files/'
regions_dict = data_dir+'mike_working_dir/identifying_regions_data/adjusted_regions.pkl'
regions_stat_dict = data_dir+'mike_working_dir/identifying_regions_data/twins_era_stats_dict_radius_regions_min_2.pkl'


def loading_dicts():
	'''
	Loads the regional dictionaries and stats dictionaries

	Returns:
		regions (dict): dictionary containing the regional dictionaries
		stats (dict): dictionary containing the regional stats dictionaries including rsd and mlt data
	'''

	print('Loading regional dictionaries....')

	with open(regions_dict, 'rb') as f:
		regions = pickle.load(f)

	regions = {f'region_{reg}': regions[f'region_{reg}'] for reg in region_numbers}

	with open(regions_stat_dict, 'rb') as g:
		stats = pickle.load(g)

	stats = {f'region_{reg}': stats[f'region_{reg}'] for reg in region_numbers}

	return regions, stats


def loading_supermag(station):
	'''
	Loads the supermag data

	Args:
		station (string): station of interest

	Returns:
		df (pd.dataframe): dataframe containing the supermag data with a datetime index
	'''

	print(f'Loading station {station}....')
	df = pd.read_feather(supermag_dir+station+'.feather')

	# limiting the analysis to the nightside
	df.set_index('Date_UTC', inplace=True, drop=True)
	df.index = pd.to_datetime(df.index, format='%Y-%m-%d %H:%M:$S')
	df['theta'] = (np.arctan2(df['N'], df['E']) * 180 / np.pi)	# calculates the angle of B_H
	df['cos_theta'] = np.cos(df['theta'] * np.pi / 180)			# calculates the cosine of the angle of B_H
	df['sin_theta'] = np.sin(df['theta'] * np.pi / 180)			# calculates the sine of the angle of B_H

	return df


def loading_twins_maps(full_map=False):
	'''
	Loads the twins maps

	Returns:
		maps (dict): dictionary containing the twins maps
	'''

	times = pd.read_feather('outputs/regular_twins_map_dates.feather')
	twins_files = sorted(glob.glob(twins_dir+'*.cdf', recursive=True))

	maps = {}

	for file in twins_files:
		twins_map = pycdf.CDF(file)
		for i, date in enumerate(twins_map['Epoch']):
			if full_map:
				if len(np.unique(twins_map['Ion_Temperature'][i])) == 1:
					continue
			else:
				if len(np.unique(twins_map['Ion_Temperature'][i][35:125,50:110])) == 1:
					continue
			check = pd.to_datetime(date.strftime(format='%Y-%m-%d %H:%M:%S'), format='%Y-%m-%d %H:%M:%S')
			if check in times.values:
				maps[check.round('T').strftime(format='%Y-%m-%d %H:%M:%S')] = {}
				if full_map:
					maps[check.round('T').strftime(format='%Y-%m-%d %H:%M:%S')]['map'] = twins_map['Ion_Temperature'][i]
				else:
					maps[check.round('T').strftime(format='%Y-%m-%d %H:%M:%S')]['map'] = twins_map['Ion_Temperature'][i][35:125,50:110]

	return maps


def loading_solarwind(omni=False, limit_to_twins=False):
	'''
	Loads the solar wind data

	Returns:
		df (pd.dataframe): dataframe containing the solar wind data
	'''

	print('Loading solar wind data....')
	if omni:
		df = pd.read_feather('../data/SW/omniData.feather')
		df.set_index('Epoch', inplace=True, drop=True)
		df.index = pd.to_datetime(df.index, format='%Y-%m-%d %H:%M:$S')
	else:
		df = pd.read_feather('../data/SW/ace_data.feather')
		df.set_index('ACEepoch', inplace=True, drop=True)
		df.index = pd.to_datetime(df.index, format='%Y-%m-%d %H:%M:$S')

	if limit_to_twins:
		df = df[pd.to_datetime('2009-07-20'):pd.to_datetime('2017-12-31')]

	return df

def getting_mean_lat(stations):

	# getting the mean latitude of the stations
	latitudes = []
	for station in stations:
		stat = loading_supermag(station)
		latitudes.append(stat['MLAT'].mean())

	mean_lat = np.mean(latitudes)

	return mean_lat


def combining_stations_into_regions(stations, rsd, features=None, mean=False, std=False, maximum=False, median=False, map_keys=None):


	start_time = pd.to_datetime('2009-07-20')
	end_time = pd.to_datetime('2017-12-31')
	twins_time_period = pd.date_range(start=start_time, end=end_time, freq='min')

	regional_df = pd.DataFrame(index=twins_time_period)

	# creating a dataframe for each feature with the twins time period as the index and storing them in a dict
	feature_dfs = {}
	if features is not None:
		for feature in features:
			feature_dfs[feature] = pd.DataFrame(index=twins_time_period)

	for stat in stations:
		df = loading_supermag(stat)
		df = df[start_time:end_time]
		if features is not None:
			for feature in features:
				feature_dfs[feature][f'{stat}_{feature}'] = df[feature]
	if features is not None:
		for feature in features:
			if mean:
				if feature == 'N' or feature == 'E':
					regional_df[f'{feature}_mean'] = feature_dfs[feature].abs().mean(axis=1)
				else:
					regional_df[f'{feature}_mean'] = feature_dfs[feature].mean(axis=1)
			if std:
				regional_df[f'{feature}_std'] = feature_dfs[feature].std(axis=1)
			if maximum:
				if feature == 'N' or feature == 'E':
					regional_df[f'{feature}_max'] = feature_dfs[feature].abs().max(axis=1)
				else:
					regional_df[f'{feature}_max'] = feature_dfs[feature].max(axis=1)
			if median:
				if feature == 'N' or feature == 'E':
					regional_df[f'{feature}_median'] = feature_dfs[feature].abs().median(axis=1)
				else:
					regional_df[f'{feature}_median'] = feature_dfs[feature].median(axis=1)

	indexer = pd.api.indexers.FixedForwardWindowIndexer(window_size=15)

	regional_df['rsd'] = rsd['max_rsd']['max_rsd']
	regional_df['rolling_rsd'] = rsd['max_rsd']['max_rsd'].rolling(indexer, min_periods=1).max()
	regional_df['MLT'] = rsd['max_rsd']['MLT']
	regional_df['cosMLT'] = np.cos(regional_df['MLT'] * 2 * np.pi * 15 / 360)
	regional_df['sinMLT'] = np.sin(regional_df['MLT'] * 2 * np.pi * 15 / 360)

	if map_keys is not None:
		segmented_df = regional_df[regional_df.index.isin(map_keys)]
		return segmented_df

	else:
		return regional_df


class RegionPreprocessing():

	def __init__(self, cluster=None, region=None, features=None, mean=False, std=False, maximum=False, median=False, **kwargs):

		if cluster is None:
			raise ValueError('Must specify a cluster to analyze.')

		if region is None:
			raise ValueError('Must specify a region to analyze.')

		self.cluster = cluster
		self.region = region
		self.features = features
		self.mean = mean
		self.std = std
		self.maximum = maximum
		self.median = median

		self.__dict__.update(kwargs)
		print(self.__dict__)
		self.forecast = self.__dict__.get('forecast', 15)
		self.window = self.__dict__.get('window', 15)
		self.classification = self.__dict__.get('classification', False)

		print(self.__dict__)

		print(f'Forecast: {self.forecast}, Window: {self.window}, Classification: {self.classification}')



	def loading_supermag(self, station):
		'''
		Loads the supermag data

		Args:
			station (string): station of interest

		Returns:
			df (pd.dataframe): dataframe containing the supermag data with a datetime index
		'''

		print(f'Loading station {station}....')
		df = pd.read_feather(supermag_dir+station+'.feather')

		# limiting the analysis to the nightside
		df.set_index('Date_UTC', inplace=True, drop=True)
		df.index = pd.to_datetime(df.index, format='%Y-%m-%d %H:%M:$S')
		df['theta'] = (np.arctan2(df['N'], df['E']) * 180 / np.pi)	# calculates the angle of B_H
		df['cos_theta'] = np.cos(df['theta'] * np.pi / 180)			# calculates the cosine of the angle of B_H
		df['sin_theta'] = np.sin(df['theta'] * np.pi / 180)			# calculates the sine of the angle of B_H

		return df


	def classification_column(self, df, param, percentile=0.99, forecast=15, window=1):
		'''
		Creating a new column which labels whether there will be a crossing of threshold
			by the param selected in the forecast window.

		Args:
			df (pd.dataframe): dataframe containing the param values.
			param (str): the paramaeter that is being examined for threshold crossings (dBHt for this study).
			thresh (float or list of floats): threshold or list of thresholds to define parameter crossing.
			forecast (int): how far out ahead we begin looking in minutes for threshold crossings.
								If forecast=30, will begin looking 30 minutes ahead.
			window (int): time frame in which we look for a threshold crossing starting at t=forecast.
								If forecast=30, window=30, we look for threshold crossings from t+30 to t+60

		Returns:
			pd.dataframe: df containing a bool column called crossing and a persistance colmun
		'''

		# creating the shifted parameter column
		thresh = df[param].quantile(percentile)

		# print(f'Threshold: {thresh}')

		df[f'shifted_{param}'] = df[param].shift(-self.forecast)					# creates a new column that is the shifted parameter. Because time moves foreward with increasing

		if window > 0:																				# index, the shift time is the negative of the forecast instead of positive.
			indexer = pd.api.indexers.FixedForwardWindowIndexer(window_size=self.window)			# Yeah this is annoying, have to create a forward rolling indexer because it won't do it automatically.
			df['window_max'] = df[f'shifted_{param}'].rolling(indexer, min_periods=1).max()		# creates new column in the df labeling the maximum parameter value in the forecast:forecast+window time frame
		# df['pers_max'] = df[param].rolling(0, min_periods=1).max()						# looks backwards to find the max param value in the time history limit
		else:
			df['window_max'] = df[f'shifted_{param}']
		# df.reset_index(drop=False, inplace=True)											# resets the index

		'''This section creates a binary column for each of the thresholds. Binary will be one if the parameter
			goes above the given threshold, and zero if it does not.'''

		conditions = [(df['window_max'] < thresh), (df['window_max'] >= thresh)]			# defining the conditions
		# pers_conditions = [(df['pers_max'] < thresh), (df['pers_max'] >= thresh)]			# defining the conditions for a persistance model

		binary = [0, 1] 																	# 0 if not cross 1 if cross

		df['classification'] = np.select(conditions, binary)						# new column created using the conditions and the binary
		# df['persistance'] = np.select(pers_conditions, binary)				# creating the persistance column

		# df.drop(['pers_max', 'window_max', f'shifted_{param}'], axis=1, inplace=True)			# removes the working columns for memory purposes
		df.drop(['window_max', f'shifted_{param}'], axis=1, inplace=True)			# removes the working columns for memory purposes

		return df


	def getting_dbdt_dataframe(self):

		dbdt_df = pd.DataFrame(index=pd.date_range(start='2009-07-20', end='2017-12-31 23:59:00', freq='min'))
		for station in self.region['stations']:
			# loading the station data
			station_df = pd.read_feather(supermag_dir + station + '.feather')
			station_df.set_index('Date_UTC', inplace=True)
			station_df.index = pd.to_datetime(station_df.index)
			# creating the dbdt time series
			dbdt_df[station] = station_df['dbht']

		return dbdt_df


	def finding_mlt(self):
		'''Finding the station in the middle of the region by geolongitude and using that as the MLT for the region.'''

		# if the difference in geolon is greater than 180 degrees then convert all the values from 0-360 to -180-180
		lons_list = list(self.lons_dict.values())
		if max(lons_list) - min(lons_list) > 180:
			for key, value in self.lons_dict.items():
				if value > 180:
					value = value - 360
		lons_list = list(self.lons_dict.values())
		median = np.median(lons_list)
		# finding the station closest to the median longitude
		closest = min(lons_list, key=lambda x:abs(x-median))
		# finding the station that is closest to the median longitude
		station = [key for key, value in self.lons_dict.items() if value == closest][0]

		return self.mlt_df[station]


	def calculating_rsd(self):

		dbdt_df = self.getting_dbdt_dataframe()
		rsd = pd.DataFrame(index=dbdt_df.index)

		# calculating the RSD
		for col in dbdt_df.columns:
			ss = dbdt_df[col]
			temp_df = dbdt_df.drop(col,axis=1)
			ra = temp_df.mean(axis=1)
			rsd[col] = ss-ra

		max_rsd = rsd.max(axis=1)
		max_station = rsd.idxmax(axis=1)
		rsd['max_rsd'] = max_rsd
		rsd['max_station'] = max_station

		return rsd


	def combining_stations_into_regions(self, map_keys=None):

		start_time = pd.to_datetime('2009-07-20')
		end_time = pd.to_datetime('2017-12-31')
		twins_time_period = pd.date_range(start=start_time, end=end_time, freq='min')

		regional_df = pd.DataFrame(index=twins_time_period)
		self.mlt_df = pd.DataFrame(index=twins_time_period)
		self.lons_dict = {}

		# creating a dataframe for each feature with the twins time period as the index and storing them in a dict
		feature_dfs = {}
		if self.features is not None:
			for feature in self.features:
				feature_dfs[feature] = pd.DataFrame(index=twins_time_period)

		for stat in self.region['stations']:
			df = self.loading_supermag(stat)
			self.lons_dict[stat] = df['GEOLON'].loc[df['GEOLON'].first_valid_index()]
			df = df[start_time:end_time]
			self.mlt_df[stat] = df['MLT']
			if self.features is not None:
				for feature in self.features:
					feature_dfs[feature][f'{stat}_{feature}'] = df[feature]
		if self.features is not None:
			for feature in self.features:
				if self.mean:
					if feature == 'N' or feature == 'E':
						regional_df[f'{feature}_mean'] = feature_dfs[feature].abs().mean(axis=1)
					else:
						regional_df[f'{feature}_mean'] = feature_dfs[feature].mean(axis=1)
				if self.std:
					regional_df[f'{feature}_std'] = feature_dfs[feature].std(axis=1)
				if self.maximum:
					if feature == 'N' or feature == 'E':
						regional_df[f'{feature}_max'] = feature_dfs[feature].abs().max(axis=1)
					else:
						regional_df[f'{feature}_max'] = feature_dfs[feature].max(axis=1)
				if self.median:
					if feature == 'N' or feature == 'E':
						regional_df[f'{feature}_median'] = feature_dfs[feature].abs().median(axis=1)
					else:
						regional_df[f'{feature}_median'] = feature_dfs[feature].median(axis=1)

		indexer = pd.api.indexers.FixedForwardWindowIndexer(window_size=15)
		mlt = self.finding_mlt()
		rsd = self.calculating_rsd()

		regional_df['rsd'] = rsd['max_rsd']
		regional_df['rolling_rsd'] = rsd['max_rsd'].rolling(indexer, min_periods=1).max()
		regional_df['MLT'] = mlt
		regional_df['cosMLT'] = np.cos(regional_df['MLT'] * 2 * np.pi * 15 / 360)
		regional_df['sinMLT'] = np.sin(regional_df['MLT'] * 2 * np.pi * 15 / 360)

		if self.classification:

			regional_df = self.classification_column(df=regional_df, param='rsd', percentile=0.99)

		if map_keys is not None:
			regional_df = regional_df[regional_df.index.isin(map_keys)]

		return regional_df


	def __call__(self, cluster_dict='cluster_dict.pkl', **kwargs):

		with open(cluster_dict, 'rb') as f:
			self.clusters = pickle.load(f)

		self.region = self.clusters[self.cluster]['regions'][self.region]

		regional_df = self.combining_stations_into_regions()

		return regional_df


def calculate_percentiles(df, param, mlt_span, percentile):

	# splitting up the regions based on MLT value into 1 degree bins
	mlt_bins = np.arange(0, 24, mlt_span)
	mlt_perc = {}
	for mlt in mlt_bins:
		mlt_df = df[df['MLT'].between(mlt, mlt+mlt_span)]
		mlt_df.dropna(inplace=True, subset=[param])
		mlt_perc[f'{mlt}'] = mlt_df[param].quantile(percentile)

	return mlt_perc


def splitting_and_scaling(input_array, target_array, dates=None, scaling_method='standard', test_size=0.2, val_size=0.25, random_seed=42):
		'''
		Splits the data into training, validation, and testing sets and scales the data.

		Args:
			scaling_method (string): scaling method to use for the solar wind and supermag data.
									Options are 'standard' and 'minmax'. Defaults to 'standard'.
			test_size (float): size of the testing set. Defaults to 0.2.
			val_size (float): size of the validation set. Defaults to 0.25. This equates to a 60-20-20 split for train-val-test
			random_seed (int): random seed for reproducibility. Defaults to 42.

		Returns:
			np.array: training input array
			np.array: testing input array
			np.array: validation input array
			np.array: training target array
			np.array: testing target array
			np.array: validation target array
		'''

		if dates is not None:
			x_train, x_test, y_train, y_test, dates_train, dates_test = train_test_split(input_array, target_array, dates, test_size=test_size, random_state=random_seed)
			x_train, x_val, y_train, y_val, dates_train, dates_val = train_test_split(x_train, y_train, dates_train, test_size=val_size, random_state=random_seed)

			dates_dict = {'train':dates_train, 'test':dates_test, 'val':dates_val}
		else:
			x_train, x_test, y_train, y_test = train_test_split(input_array, target_array, test_size=test_size, random_state=random_seed)
			x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=val_size, random_state=random_seed)


		# defining the TWINS scaler
		if scaling_method == 'standard':
			scaler = StandardScaler()
		elif scaling_method == 'minmax':
			scaler = MinMaxScaler()
		else:
			raise ValueError('Must specify a valid scaling method for TWINS. Options are "standard" and "minmax".')

		# scaling the TWINS data
		x_train = scaler.fit_transform(x_train.reshape(-1, x_train.shape[-1])).reshape(x_train.shape)
		x_test = scaler.transform(x_test.reshape(-1, x_test.shape[-1])).reshape(x_test.shape)
		x_val = scaler.transform(x_val.reshape(-1, x_val.shape[-1])).reshape(x_val.shape)

		return x_train, x_test, x_val, y_train, y_test, y_val, dates_dict



def classification_column(df, param, thresh, forecast, window):
		'''
		Creating a new column which labels whether there will be a crossing of threshold
			by the param selected in the forecast window.

		Args:
			df (pd.dataframe): dataframe containing the param values.
			param (str): the paramaeter that is being examined for threshold crossings (dBHt for this study).
			thresh (float or list of floats): threshold or list of thresholds to define parameter crossing.
			forecast (int): how far out ahead we begin looking in minutes for threshold crossings.
								If forecast=30, will begin looking 30 minutes ahead.
			window (int): time frame in which we look for a threshold crossing starting at t=forecast.
								If forecast=30, window=30, we look for threshold crossings from t+30 to t+60

		Returns:
			pd.dataframe: df containing a bool column called crossing and a persistance colmun
		'''


		df[f'shifted_{param}'] = df[param].shift(-forecast)					# creates a new column that is the shifted parameter. Because time moves foreward with increasing

		if window > 0:																				# index, the shift time is the negative of the forecast instead of positive.
			indexer = pd.api.indexers.FixedForwardWindowIndexer(window_size=window)			# Yeah this is annoying, have to create a forward rolling indexer because it won't do it automatically.
			df['window_max'] = df[f'shifted_{param}'].rolling(indexer, min_periods=1).max()		# creates new column in the df labeling the maximum parameter value in the forecast:forecast+window time frame
		# df['pers_max'] = df[param].rolling(0, min_periods=1).max()						# looks backwards to find the max param value in the time history limit
		else:
			df['window_max'] = df[f'shifted_{param}']
		# df.reset_index(drop=False, inplace=True)											# resets the index

		'''This section creates a binary column for each of the thresholds. Binary will be one if the parameter
			goes above the given threshold, and zero if it does not.'''

		conditions = [(df['window_max'] < thresh), (df['window_max'] >= thresh)]			# defining the conditions
		# pers_conditions = [(df['pers_max'] < thresh), (df['pers_max'] >= thresh)]			# defining the conditions for a persistance model

		binary = [0, 1] 																	# 0 if not cross 1 if cross

		df['classification'] = np.select(conditions, binary)						# new column created using the conditions and the binary
		# df['persistance'] = np.select(pers_conditions, binary)				# creating the persistance column

		# df.drop(['pers_max', 'window_max', f'shifted_{param}'], axis=1, inplace=True)			# removes the working columns for memory purposes
		df.drop(['window_max', f'shifted_{param}'], axis=1, inplace=True)			# removes the working columns for memory purposes

		return df



def storm_extract(df, lead=24, recovery=48, sw_only=False, twins=False, target=False, target_var=None, concat=False, map_keys=None, classification=False):

	'''
	Pulling out storms using a defined list of datetime strings, adding a lead and recovery time to it and
	appending each storm to a list which will be later processed.

	Args:
		data (list of pd.dataframes): ACE and supermag data with the test set's already removed.
		lead (int): how much time in hours to add to the beginning of the storm.
		recovery (int): how much recovery time in hours to add to the end of the storm.
		sw_only (bool): True if this is the solar wind only data, will drop dbht from the feature list.

	Returns:
		list: ace and supermag dataframes for storm times
		list: np.arrays of shape (n,2) containing a one hot encoded boolean target array
	'''
	storms, y = list(), list()				# initalizing the lists
	all_storms, all_targets = pd.DataFrame(), pd.DataFrame()

	# setting the datetime index
	if 'Date_UTC' in df.columns:
		pd.to_datetime(df['Date_UTC'], format='%Y-%m-%d %H:%M:%S')
		df.reset_index(drop=True, inplace=True)
		df.set_index('Date_UTC', inplace=True, drop=True)
	else:
		print('Date_UTC not in columns. Check to make sure index is datetime not integer.')

	df.index = pd.to_datetime(df.index)

	# loading the storm list
	if twins and map_keys is None:
		storm_list = pd.read_feather('outputs/regular_twins_map_dates.feather', columns=['dates'])
		storm_list = storm_list['dates']
	elif twins and map_keys is not None:
		storm_list = pd.DataFrame({'dates':[pd.to_datetime(key, format='%Y-%m-%d %H:%M:%S') for key in map_keys]})
		storm_list = storm_list['dates']
	else:
		storm_list = pd.read_csv('stormList.csv', header=None, names=['dates'])
		storm_list = storm_list['Date_UTC']

	stime, etime = [], []					# will store the resulting time stamps here then append them to the storm time df

	# will loop through the storm dates, create a datetime object for the lead and recovery time stamps and append those to different lists
	for date in storm_list:
		if isinstance(date, str):
			date = pd.to_datetime(date, format='%Y-%m-%d %H:%M:%S')
		if twins:
			stime.append(date.round('T')-pd.Timedelta(minutes=lead))
			etime.append(date.round('T')+pd.Timedelta(minutes=recovery))
		else:
			stime.append(date-pd.Timedelta(hours=lead))
			etime.append(date+pd.Timedelta(hours=recovery))

	# adds the time stamp lists to the storm_list dataframes

	storm_list = pd.DataFrame(storm_list, columns=['dates'])
	storm_list['stime'] = stime
	storm_list['etime'] = etime

	storm_list = pd.DataFrame({'stime':stime, 'etime':etime})
	if classification:
		ohe = OneHotEncoder().fit(np.array([0,1]).reshape(-1,1))


	for start, end in zip(storm_list['stime'], storm_list['etime']):		# looping through the storms to remove the data from the larger df
		if start < df.index[0] or end > df.index[-1]:						# if the storm is outside the range of the data, skip it
			continue
		storm = df[(df.index >= start) & (df.index <= end)]

		if len(storm) != 0:
			if target:
				if classification:
					y.append(ohe.transform(storm[target_var].values.reshape(-1,1)).toarray())	

				else:
					y.append(storm[target_var].values)

				storm.drop(target_var, axis=1, inplace=True)
				storms.append(storm)
			else:
				storms.append(storm)			# creates a list of smaller storm time dataframes

	if concat:
		for storm, tar in zip(storms, y):
			all_storms = pd.concat([all_storms, storm], axis=0, ignore_index=False)
			storm.reset_index(drop=True, inplace=True)		# resetting the storm index and simultaniously dropping the date so it doesn't get trained on
			all_targets = pd.concat([all_targets, pd.DataFrame(tar)], axis=0, ignore_index=True)
			all_targets.reset_index(drop=True, inplace=True)

		return all_storms

	else:
		return storms, y


def split_sequences(sequences, targets=None, n_steps=30, include_target=True, dates=None, model_type='classification', maps=None):
	'''
		Takes input from the input array and creates the input and target arrays that can go into the models.

		Args:
			sequences (np.array): input features. Shape = (length of data, number of input features)
			results_y: series data of the targets for each threshold. Shape = (length of data, 1)
			n_steps (int): the time history that will define the 2nd demension of the resulting array.
			include_target (bool): true if there will be a target output. False for the testing data.

		Returns:
			np.array (n, time history, n_features): array for model input
			np.array (n, 1): target array
		'''
	if maps is None:
		maps = [None] * len(sequences)
	X, y, twins_maps, to_drop = list(), list(), list(), list()							# creating lists for storing results
	index_to_drop = 0
	for sequence, target, twins in zip(sequences, targets, maps):	# looping through the sequences and targets
		for i in range(len(sequence)-n_steps):			# going to the end of the dataframes
			end_ix = i + n_steps						# find the end of this pattern
			if end_ix > len(sequence):					# check if we are beyond the dataset
				break
			seq_x = sequence[i:end_ix, :]				# grabs the appropriate chunk of the data
			if include_target:
				if np.isnan(seq_x).any():
					if dates is not None:				# doesn't add arrays with nan values to the training set
						to_drop.append(index_to_drop)
						index_to_drop += 1
					continue
				if model_type == 'classification':
					if np.isnan(target[end_ix, :]).any():
						to_drop.append(index_to_drop)
						index_to_drop += 1
						continue
					seq_y1 = target[end_ix, :]				# gets the appropriate target
				elif model_type == 'regression':
					if np.isnan(target[end_ix]):
						to_drop.append(index_to_drop)
						index_to_drop += 1
						continue
					seq_y1 = target[end_ix]					# gets the appropriate target
				else:
					raise ValueError('Must specify a valid model type. Options are "classification" and "regression".')
				y.append(seq_y1)
			X.append(seq_x)
			if maps is not None:
				twins_maps.append(twins)
			index_to_drop += 1

	return np.array(X), np.array(y), to_drop, np.array(twins_maps)
