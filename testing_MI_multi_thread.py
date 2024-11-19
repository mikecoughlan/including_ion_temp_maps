import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
import os
import gc
import glob
import torch
import utils
import sklearn
from scipy.special import erf
from sklearn.calibration import calibration_curve
from sklearn.metrics import (auc, brier_score_loss, confusion_matrix,
                             mean_absolute_error, mean_squared_error,
                             precision_recall_curve, r2_score, roc_curve,
							 precision_score, recall_score, f1_score,
                             mutual_info_score)
from sklearn.feature_selection import mutual_info_regression, r_regression
from scipy.stats import entropy
import utils
import math
import skimage.measure
from tqdm import tqdm

model_type = 'twins'
shap_dir = 'outputs/shap_values'
scaler_dir = 'models/'
rsd_results_dir = 'outputs/rsd'
dbht_results_dir = 'outputs/dbht_max'
VERSION = 'extended_v0-1'
shap_files = glob.glob(f'{shap_dir}/*{VERSION}*.pkl')
scaler_files = glob.glob(f'{scaler_dir}/*{VERSION}.pkl')
results_files = glob.glob(f'{scaler_dir}/*{VERSION}.feather')
with open('cluster_dict.pkl', 'rb') as f:
	cluster_dict = pickle.load(f)
print(f'Version of scikit learn: {sklearn.__version__}')
list_of_oversampled_regions = ['CEU-2', 'GRL-0', 'GRL-1', 'GRL-2', 'SVLB', 'JPN-0', 'JPN-1']

scalers = {}
for cluster in cluster_dict.values():
	for key, region in cluster['regions'].items():
		if not os.path.exists(f'outputs/shap_values/swmag_region_{key}_extended_v0-1_dbht.pkl'):
			continue
		scalers[key] = {}
		if os.path.exists(f'models/dbht_max/region_{key}_version_extended_v0-1_dbht_scaler.pkl'):
			with open(f'models/dbht_max//region_{key}_version_extended_v0-1_dbht_scaler.pkl', 'rb') as f:
				scaler_values = pickle.load(f)
				scalers[key]['swmag'] = scaler_values
				print(f'{key} scalers loaded')

# individual_cluster_to_examine = 'canadian_cluster'
features = ['dbht_median', 'MAGNITUDE_median', 'MAGNITUDE_std', 'sin_theta_std', 'cos_theta_std', 'cosMLT', 'sinMLT',
				'B_Total', 'BX_GSE', 'BY_GSM', 'BZ_GSM', 'Vx', 'Vy', 'Vz', 'proton_density', 'logT']
mean_shap_rsd, mean_shap_dbht, X, mu, sigma, error = [], [], [], [], [], []
dataframes_to_make = ['high', 'mid', 'low']
total_dict = {'inputs':{'dbht':{}, 'rsd':{}}, 'shap_values':{'dbht':{}, 'rsd':{}}}
seg_dict = {f'{feature}':{'shap_values':{}, 'input_values':{}} for feature in features}
with open('cluster_dict.pkl', 'rb') as f:
	cluster_dict = pickle.load(f)
for feat in seg_dict.keys():
	for df in dataframes_to_make:
		seg_dict[feat]['input_values'][df] = pd.DataFrame()
		seg_dict[feat]['shap_values'][df] = pd.DataFrame()
		total_dict['inputs']['dbht'][df] = pd.DataFrame(columns=features)
		total_dict['inputs']['rsd'][df] = pd.DataFrame(columns=features)
		total_dict['shap_values']['dbht'][df] = pd.DataFrame(columns=features)
		total_dict['shap_values']['rsd'][df] = pd.DataFrame(columns=features)
for clust, cluster in cluster_dict.items():
	for reg, region in cluster_dict[clust]['regions'].items():
		if (os.path.exists(f'outputs/shap_values/swmag_region_{reg}_extended_v0-1.pkl')) and (os.path.exists(f'outputs/shap_values/swmag_region_{reg}_extended_v0-1_dbht.pkl')):
			# with open(f'outputs/shap_values/swmag_region_{reg}_extended_v0-1.pkl', 'rb') as f:
			# 	rsd_shap_values = pickle.load(f)
			with open(f'outputs/shap_values/swmag_region_{reg}_extended_v0-1_dbht.pkl', 'rb') as f:
				dbht_shap_values = pickle.load(f)

				cluster['regions'][reg]['MLAT'] = utils.getting_mean_lat(region['stations'])
				cluster['regions'][reg]['shap'] = {}
				# cluster['regions'][reg]['shap']['testing_data'] = shap_values['testing_data']
				# cluster['regions'][reg]['shap']['rsd_shap'] = np.concatenate([rsd_shap_values['shap_values'][i][1][:,:,:,:] for i in range(len(rsd_shap_values['shap_values']))], axis=0)
				cluster['regions'][reg]['shap']['dbht_shap'] = np.concatenate([dbht_shap_values['shap_values'][i][1][:,:,:,:] for i in range(len(dbht_shap_values['shap_values']))], axis=0)
				# cluster['regions'][key]['twins_shap']['mean_shap'] = np.concatenate([shap_values['shap_values'][i][0][0][:,:,:,:] for i in range(len(shap_values['shap_values']))], axis=0)
			# region[f'{model_type}_shap']['std_shap'] = np.concatenate([region[f'{model_type}_shap']['shap_values'][i][1][:,:,:,:] for i in range(len(region[f'{model_type}_shap']['shap_values']))], axis=0)

			# finding the places where the rsd is positive adn dbht negative and extracting those indicies
			# rsd_pos_dbht_neg = np.where((rsd_shap_values['ytest'][:,1] == 1) & (dbht_shap_values['ytest'][:,1] == 0))
			# rsd_neg_dbht_pos = np.where((rsd_shap_values['ytest'][:,1] == 0) & (dbht_shap_values['ytest'][:,1] == 1))

			# cluster['regions'][reg]['shap']['rsd_shap'] = cluster['regions'][reg]['shap']['rsd_shap'][rsd_neg_dbht_pos]
			# cluster['regions'][reg]['shap']['dbht_shap'] = cluster['regions'][reg]['shap']['dbht_shap'][rsd_neg_dbht_pos]
			# cluster['regions'][reg]['shap']['testing_data'] = rsd_shap_values['testing_data'][rsd_neg_dbht_pos]
			cluster['regions'][reg]['shap']['testing_data'] = dbht_shap_values['testing_data']

			for key in region['shap'].keys():
				if isinstance(region['shap'][key], torch.Tensor):
					region['shap'][key] = region['shap'][key].cpu().detach().numpy()

			try:
				# region['shap']['rsd_shap'] = region['shap']['rsd_shap'].reshape(region['shap']['rsd_shap'].shape[0], region['shap']['rsd_shap'].shape[2], region['shap']['rsd_shap'].shape[3])
				region['shap']['dbht_shap'] = region['shap']['dbht_shap'].reshape(region['shap']['dbht_shap'].shape[0], region['shap']['dbht_shap'].shape[2], region['shap']['dbht_shap'].shape[3])
				# region['shap']['testing_data'] = region['shap']['testing_data'].reshape(region['shap']['testing_data'].shape[0], region['shap']['testing_data'].shape[2], region['shap']['testing_data'].shape[3])
				# region['shap']['std_shap'] = region['shap']['std_shap'].reshape(region['shap']['std_shap'].shape[0], region['shap']['std_shap'].shape[2], region['shap']['std_shap'].shape[3])
				region['shap']['testing_data'] = region['shap']['testing_data'].reshape(region['shap']['testing_data'].shape[0], region['shap']['testing_data'].shape[2], region['shap']['testing_data'].shape[3])

				# rsd_mean_added = np.sum(np.sum(np.abs(region['shap']['rsd_shap']), axis=1),axis=1)
				dbht_mean_added = np.sum(np.sum(np.abs(region['shap']['dbht_shap']), axis=1),axis=1)
				# region['shap']['rsd_shap'] = region['shap']['rsd_shap']/rsd_mean_added[:,None,None]
				region['shap']['dbht_shap'] = region['shap']['dbht_shap']/dbht_mean_added[:,None,None]

				# std_added = np.sum(np.sum(np.abs(region['shap']['std_shap']), axis=1),axis=1)
				# region['shap']['std_shap'] = region['shap']['std_shap']/std_added[:,None,None]

			except:
				# print('We already did this, skipping....')
				pass

			region_scaler = scalers[reg]['swmag']

			lat = 'high' if cluster['regions'][reg]['MLAT'] > 68 else 'mid' if cluster['regions'][reg]['MLAT'] > 55 else 'low'
			# for v, var in enumerate(seg_dict.keys()):
			# 	seg_dict[var]['shap_values'][lat] = pd.concat([seg_dict[var]['shap_values'][lat], pd.DataFrame(region['shap']['rsd_shap'][:,:,v])], axis=0)
			# 	seg_dict[var]['input_values'][lat] = pd.concat([seg_dict[var]['input_values'][lat], pd.DataFrame(region['shap']['testing_data'][:,:,v])], axis=0)
				# time_dict[var]['std_shap_df'] = pd.concat([time_dict[var]['std_shap_df'], pd.DataFrame(region[f'{model_type}_shap']['std_shap'][:,:,v])], axis=0)
				# time_dict[var]['transformed_X_df'] = pd.concat([time_dict[var]['transformed_X_df'], \
				# 												pd.DataFrame(np.array([region_scaler.inverse_transform(region['shap']['testing_data'][i,:,:]) \
				# 												for i in range(region['shap']['testing_data'].shape[0])])[:,:,v])], axis=0)
			total_dict['inputs']['dbht'][lat] = pd.concat([total_dict['inputs']['dbht'][lat], pd.DataFrame(region_scaler.inverse_transform(np.concatenate(region['shap']['testing_data'][:,:,:])), columns=features)], axis=0)
			# total_dict['inputs']['rsd'][lat] = pd.concat([total_dict['inputs']['rsd'][lat], pd.DataFrame(region_scaler.inverse_transform(np.concatenate(region['shap']['testing_data'][:,:,:])), columns=features)], axis=0)
			total_dict['shap_values']['dbht'][lat] = pd.concat([total_dict['shap_values']['dbht'][lat], pd.DataFrame(np.concatenate(region['shap']['dbht_shap'][:,:,:]), columns=features)], axis=0)
			# total_dict['shap_values']['rsd'][lat] = pd.concat([total_dict['shap_values']['rsd'][lat], pd.DataFrame(np.concatenate(region['shap']['rsd_shap'][:,:,:]), columns=features)], axis=0)
		region = {}
		gc.collect()

# time_columns = [f't-{60-i}' for i in range(0,60)]
# for var in seg_dict.keys():
# 	for df in seg_dict[var]['shap_values'].keys():
# 		print(f'Creating {var} {df} dataframe')
# 		seg_dict[var]['shap_values'][df].columns = time_columns
# 		seg_dict[var]['input_values'][df].columns = time_columns

for lat in ['high', 'mid', 'low']:
	total_dict['inputs']['dbht'][lat].reset_index(drop=True, inplace=True)
	# total_dict['inputs']['rsd'][lat].reset_index(drop=True, inplace=True)
	total_dict['shap_values']['dbht'][lat].reset_index(drop=True, inplace=True)
	# total_dict['shap_values']['rsd'][lat].reset_index(drop=True, inplace=True)

# dfs = ['high', 'mid', 'low']
# rsd_MI_dict = {feature:[] for feature in features}
# for col in total_dict['inputs']['rsd']['low'].columns:
# 	for df in dfs:
# 		rsd_MI_dict[col].append(mutual_info_regression(total_dict['inputs']['rsd'][df][col].to_numpy().reshape(-1,1), total_dict['shap_values']['rsd'][df][col].to_numpy())[0])

print('Calculating MI for dbht')
mis = {'high':[], 'mid':[], 'low':[]}
for lat in ['high', 'mid', 'low']:
	for col in total_dict['inputs']['dbht'][lat].columns:
		mis[lat].append(mutual_info_regression(total_dict['inputs']['dbht'][lat][col].to_numpy().reshape(-1,1), total_dict['shap_values']['dbht'][lat][col].to_numpy(), random_state=42, n_jobs=-1)[0])

# var = 'BY_GSM'
dfs = ['high', 'mid', 'low']
to_plot = 'shap_values'

for df in dfs:
	fig, axes = plt.subplots(4, 4, figsize=(20,20))
	fig.suptitle(f'dbht {df} latitude input vs. shap values')
	for i, col in enumerate(total_dict['inputs']['dbht'][lat].columns):
		j = i//4
		k = i%4
		axes[j,k].hist2d(total_dict['inputs']['dbht'][df][col], total_dict['shap_values']['dbht'][df][col], bins=50, norm=mpl.colors.LogNorm())
		plot_min = min(total_dict['inputs']['dbht'][df][col].min(), total_dict['shap_values']['dbht'][df][col].min())
		plot_max = max(total_dict['inputs']['dbht'][df][col].max(), total_dict['shap_values']['dbht'][df][col].max())
		axes[j,k].set_xlabel('input value', fontsize=10)
		axes[j,k].set_ylabel('shap value', fontsize=10)
		# axes[j,k].bar([i-60 for i in range(0,60)], total_dict[var]['rsd'][df].abs().mean(axis=0), fill=False, label='rsd', edgecolor='orange')
		axes[j,k].set_title(col+f'MI: {mis[df][i]}', fontsize=15)
	plt.subplots_adjust(hspace=0.5)
	plt.savefig(f'plots/background_dbht_inputs_v_shap_{df}.png')