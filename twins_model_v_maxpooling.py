####################################################################################
#
# exmining_twins_and_supermag/non_twins_modeling_v0.py
#
# Performing the modeling using the Solar Wind and Ground Magnetomoeter data.
# TWINS data passes through a pre-trained autoencoder that reduces the TWINS maps
# to a reuced dimensionality. This data is then concatenated onto the model after
# both branches of the CNN hae been flattened, and before the dense layers.
# Similar model to Coughlan (2023) but with a different target variable.
#
####################################################################################


import argparse
# Importing the libraries
import datetime
import gc
import glob
import json
import math
import os
import pickle
import subprocess
import time

import matplotlib
import matplotlib.animation as animation
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import tqdm
from scipy.stats import boxcox
from scipy import special
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from spacepy import pycdf
from torch.utils.data import DataLoader, Dataset, TensorDataset
from torchsummary import summary
from torchvision.models.feature_extraction import (create_feature_extractor,
                                                   get_graph_node_names)

import utils

pd.options.mode.chained_assignment = None

os.environ["CDF_LIB"] = "~/CDF/lib"

data_directory = '../../../../data/'
supermag_dir = '../data/supermag/feather_files/'
regions_dict = 'mike_working_dir/identifying_regions_data/identifying_regions_data/twins_era_identified_regions_min_2.pkl'
regions_stat_dict = 'mike_working_dir/identifying_regions_data/identifying_regions_data/twins_era_stats_dict_radius_regions_min_2.pkl'
working_dir = data_directory+'mike_working_dir/twins_data_modeling/'


DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Device: {DEVICE}')


CONFIG = {'time_history':30,
			'random_seed':42,
			'filters':128,
			'learning_rate':1e-7,
			'epochs':500,
			'loss':'mse',
			'early_stop_patience':25,
			'batch_size':128}


# TARGET = 'rsd'
# VERSION = 'twins_alt_v4_dbht'





class CRSP(nn.Module):
	'''
	Defining the CRPS loss function for model training.
	'''

	def __init__(self):
		super(CRSP, self).__init__()

	def forward(self, y_pred, y_true):

		# splitting the y_pred tensor into mean and std

		mean, std = torch.unbind(y_pred, dim=-1)
		# y_true = torch.unbind(y_true, dim=-1)

		# making the arrays the right dimensions
		mean = mean.unsqueeze(-1)
		std = std.unsqueeze(-1)
		y_true = y_true.unsqueeze(-1)

		# calculating the error
		crps = torch.mean(self.calculate_crps(self.epsilon_error(y_true, mean), std))

		return crps

	def epsilon_error(self, y, u):

		epsilon = torch.abs(y - u)

		return epsilon

	def calculate_crps(self, epsilon, sig):

		crps = torch.mul(sig, (torch.add(torch.mul(torch.div(epsilon, sig), torch.erf(torch.div(epsilon, torch.mul(np.sqrt(2), sig)))), \
								torch.sub(torch.mul(torch.sqrt(torch.div(2, np.pi)), torch.exp(torch.div(torch.mul(-1, torch.pow(epsilon, 2)), \
								(torch.mul(2, torch.pow(sig, 2)))))), torch.div(1, torch.sqrt(torch.tensor(np.pi)))))))

		# crps = sig * ((epsilon / sig) * torch.erf((epsilon / (np.sqrt(2) * sig))) + torch.sqrt(torch.tensor(2 / np.pi)) * torch.exp(-epsilon ** 2 / (2 * sig ** 2)) - 1 / torch.sqrt(torch.tensor(np.pi)))

		return crps


class ACCRUE(nn.Module):
	''' Defining the ACCRUE cost function from Camporeale & Care (2020)'''
	def __init__(self):
		super(ACCRUE, self).__init__()
	
	def forward(self, y_pred, y_true):
		# splitting the y_pred tensor into mean and std
		mean, std = torch.unbind(y_pred, dim=-1)

		# making the arrays the right dimensions
		mean = mean.unsqueeze(-1)
		std = std.unsqueeze(-1)
		y_true = y_true.unsqueeze(-1)
		N = y_true.shape[0]

		# calculating the error
		crps = torch.mean(self.calculate_crps(self.epsilon_error(y_true, mean), std))
		rs = torch.sub(torch.mean(self.calculate_rs(self.eta(y_true, mean, std), N)), torch.div(1, torch.sqrt(torch.mul(2, torch.tensor(np.pi)))))
		beta = self.calculate_beta(self.epsilon_error(y_true, mean), N)
		# beta = 0.5

		accrue = torch.add(torch.mul(crps, beta), torch.mul(rs, torch.sub(1, beta)))

		return accrue

	# def est_beta(y_pred, y_true):

	# 	y_true = y_true.cpu().detach().numpy()
	# 	y_pred = y_pred.cpu().detach().numpy()

	# 	d = y_true - y_pred
	# 	d = (d - d.mean())/d.std()
	# 	N = len(d)
	# 	i = np.arange(1, N+1)
	# 	# RS_min_1 = 1/(np.sqrt(np.pi)*N)
	# 	RS_min_2 = -1*erfinv((2*i-1)/N-1)**2
	# 	# RS_min_3 = np.sqrt(2/np.pi)/2

	# 	RS_min = np.nansum(np.exp(RS_min_2)/np.sqrt(np.pi)/N)
	# 	# CRPS from matlab code
	# 	sigma = np.abs(d)/np.sqrt(np.log(2))+1e-6
	# 	dum = d/sigma/np.sqrt(2)
	# 	CRPS_min = np.nanmean(sigma*(np.sqrt(2)*dum*erf(dum)
	# 							+ np.sqrt(2/np.pi)*np.exp(-1*dum**2) 
	# 							- 1/np.sqrt(np.pi)))

	# 	return RS_min/(RS_min+CRPS_min), CRPS_min, RS_min


	# def get_loss(self, y_pred, y_true, X, training=False):

	# 	# import ipdb;ipdb.set_trace()
	# 	d = y_true[:, 1] - y_true[:, 0]
	# 	# d = (d - self.mean)/self.std
	# 	# y_pred = (y_pred - self.mean)/self.std

	# 	# d = d.cuda()
	# 	N = d.shape[0]
	# 	sigma = torch.exp(y_pred).squeeze()

	# 	x = torch.zeros(sigma.shape[0])
	# 	CRPS = torch.zeros(sigma.shape[0])
	# 	RS = torch.zeros(sigma.shape[0])

	# 	x = d/sigma
	# 	x = x/np.sqrt(2)

	# 	# import ipdb;ipdb.set_trace()

	# 	ind = torch.argsort(x)
	# 	ind_orig = torch.argsort(ind)+1
	# 	ind_orig = ind_orig
		# x = x.to(self.device)
		# CRPS_1 = np.sqrt(2)*x*erf(x)
		# CRPS_2 = np.sqrt(2/np.pi)*torch.exp(-x**2) 
		# CRPS_3 = 1/torch.sqrt(torch.tensor(np.pi))
		# # CRPS_1 = CRPS_1.to(self.device)
		# # CRPS_2 = CRPS_2.to(self.device)
		# CRPS_3 = CRPS_3

		# CRPS = sigma*(CRPS_1 + CRPS_2 - CRPS_3)

		# # import ipdb;ipdb.set_trace()
		# RS = N*(x/N*(erf(x)+1) - 
		# 	x*(2*ind_orig-1)/N**2 + 
		# 	torch.exp(-x**2)/np.sqrt(np.pi)/N)

		# RS = RS
		# loss = CRPS/self.CRPS_min+RS/self.RS_min
		# loss = torch.mean(loss)

		# return loss

	def epsilon_error(self, y, u):

		epsilon = torch.abs(y - u)

		return epsilon


	def eta(self, y, u, sig):

		eta = torch.div(self.epsilon_error(y,u), torch.mul(torch.sqrt(torch.tensor(2)), sig))

		return eta


	def calculate_crps(self, epsilon, sig):

		crps = torch.mul(sig,(torch.add(torch.mul(torch.div(epsilon, sig), torch.erf(torch.div(epsilon, torch.mul(np.sqrt(2), sig)))), \
								torch.sub(torch.mul(torch.sqrt(torch.div(2, np.pi)), torch.exp(torch.div(torch.mul(-1, torch.pow(epsilon, 2)), \
								(torch.mul(2, torch.pow(sig, 2)))))), torch.div(1, torch.sqrt(torch.tensor(np.pi)))))))
		
		return crps

	
	def calculate_rs(self, eta, N):
		''' Function to calculate the reliability score of the model'''

		ind = torch.argsort(eta, dim=0)
		ind_orig = torch.argsort(ind)+1

		# getting a tensor that contains numbers 1 to N
		i = torch.sub(torch.mul(2,torch.arange(1, N+1)),1)

		i = i.to(DEVICE)
		N = torch.tensor(N).to(DEVICE)
		eta = eta.to(DEVICE)
		ind_orig = ind_orig.to(DEVICE)

		# RS with triling terms
		# rs = torch.sub(torch.mean(torch.add(torch.mul(eta, torch.add(torch.erf(eta),1)), \
		# 				torch.add(torch.mul(torch.mul(-1, torch.div(eta, N)), i),
		# 				torch.div(torch.exp(torch.mul(-1,torch.pow(eta,2))), torch.sqrt(torch.tensor(np.pi)))))),\
		# 				torch.div(1, torch.sqrt(torch.mul(2, torch.tensor(np.pi)))))

		# RS without trailing term
		rs = torch.mul(N,(torch.add(torch.mul(torch.div(eta,N), torch.add(torch.erf(eta),1)), \
						torch.add(torch.mul(torch.mul(-1, torch.div(torch.sub(torch.mul(2,ind_orig),1), torch.pow(N,2))), eta),
						torch.div(torch.exp(torch.mul(-1,torch.pow(eta,2))), torch.mul(N,torch.sqrt(torch.tensor(np.pi))))))))
		
		return rs

	def crps_min(self, epsilon, N):
		''' Function to calculate the theoretical minimum CRPS of the model'''
		crps_min = torch.mul(torch.div(torch.sqrt(torch.log(torch.tensor(4))), torch.mul(2,N)),epsilon)

		return torch.sum(crps_min)
	

	def rs_min(self, N):

		i = torch.sub(torch.div(torch.sub(torch.mul(2,torch.arange(1, N+1)),1),N),1)

		# RS with trailing terms
		# rs_min = torch.sub((torch.mul(torch.div(1,torch.sqrt(torch.tensor(np.pi))), \
		# 					torch.mean(torch.exp(torch.mul(-1, torch.pow(torch.erfinv(i),2)))))), \
		# 					torch.div(1,torch.sqrt(torch.mul(2, torch.tensor(np.pi)))))

		# RS without trailing terms
		rs_min = (torch.mul(torch.div(1,torch.mul(N,torch.sqrt(torch.tensor(np.pi)))), \
							torch.exp(torch.mul(-1, torch.pow(torch.erfinv(i),2)))))
		
		return torch.sum(rs_min)

	
	def calculate_beta(self, epsilon, N):

		crps_min = self.crps_min(epsilon, N)
		rs_min = self.rs_min(N)

		beta = torch.div(crps_min, torch.add(crps_min, rs_min))

		return beta


# class TWINSModel(nn.Module):
# 	def __init__(self):
# 		# this on is the RSD model
# 		super(TWINSModel, self).__init__()

# 		self.maxpooling = nn.Sequential(

# 			nn.MaxPool2d(kernel_size=(3,5), stride=(3,5)),
# 			# nn.Flatten()

# 		)

# 		self.cnn_block = nn.Sequential(

# 			nn.Conv2d(in_channels=1, out_channels=128, kernel_size=2, stride=1, padding='same'),
# 			nn.ReLU(),
# 			nn.MaxPool2d(kernel_size=2, stride=2),
# 			nn.Conv2d(in_channels=128, out_channels=256, kernel_size=2, stride=1, padding='same'),
	# 		nn.ReLU(),
	# 		# nn.Flatten(),
	# 	)

	# 	self.fc_block = nn.Sequential(
	# 		nn.Linear((256*15*7)+(360), 256),
	# 		nn.ReLU(),
	# 		nn.Dropout(0.2),
	# 		nn.Linear(256, 128),
	# 		nn.ReLU(),
	# 		nn.Dropout(0.2),
	# 		nn.Linear(128, 2),
	# 		nn.Sigmoid()
	# 	)

	# def forward(self, swmag, twins):

	# 	pooled = self.maxpooling(twins)
	# 	pooled = torch.reshape(pooled, (-1, 30*12))

	# 	# x_input = torch.cat((swmag, reduced), dim=3)

	# 	swmag_output = self.cnn_block(swmag)
	# 	swmag_output = torch.reshape(swmag_output, (-1, 256*15*7))

	# 	x_input = torch.cat((swmag_output, pooled), dim=1)

	# 	output = self.fc_block(x_input)

	# 	# clipping to avoid values too small for backprop
	# 	output = torch.clamp(output, min=1e-9)

	# 	return output

class TWINSModel(nn.Module):
	def __init__(self):
		super(TWINSModel, self).__init__()

		self.maxpooling = nn.Sequential(

			nn.MaxPool2d(kernel_size=(3,5), stride=(3,5)),
			# nn.Flatten()

		)

		self.cnn_block = nn.Sequential(

			nn.Conv2d(in_channels=1, out_channels=128, kernel_size=2, stride=1, padding='same'),
			nn.ReLU(),
			nn.MaxPool2d(kernel_size=2, stride=2),
			nn.Conv2d(in_channels=128, out_channels=256, kernel_size=2, stride=1, padding='same'),
			nn.ReLU(),
			# nn.Flatten(),
		)

		self.fc_block = nn.Sequential(
			nn.Linear((256*15*7)+(360), 256),
			nn.ReLU(),
			nn.Dropout(0.2),
			nn.Linear(256, 128),
			nn.ReLU(),
			nn.Dropout(0.2),
			nn.Linear(128, 2),
			nn.Sigmoid()
		)

	def forward(self, swmag, twins):

		pooled = self.maxpooling(twins)
		pooled = torch.reshape(pooled, (-1, 30*12))

		# x_input = torch.cat((swmag, reduced), dim=3)

		swmag_output = self.cnn_block(swmag)
		swmag_output = torch.reshape(swmag_output, (-1, 256*15*7))

		x_input = torch.cat((swmag_output, pooled), dim=1)

		output = self.fc_block(x_input)

		# clipping to avoid values too small for backprop
		output = torch.clamp(output, min=1e-9)

		return output

	
	def predict(self, swmag, twins, return_numpy=False):

		if not isinstance(swmag, torch.Tensor):
			swmag = torch.tensor(swmag).to(DEVICE, dtype=torch.float)
		if not isinstance(twins, torch.Tensor):
			twins = torch.tensor(twins).to(DEVICE, dtype=torch.float)
		
		self.eval()
		with torch.no_grad():
			
			output = self.forward(swmag, twins)

		if return_numpy:
			output = output.cpu().numpy()
		
		else:
			output = output.cpu()
		
		return output



class Early_Stopping():
	'''
	Class to create an early stopping condition for the model.

	'''

	def __init__(self, decreasing_loss_patience=25):
		'''
		Initializing the class.

		Args:
			decreasing_loss_patience (int): the number of epochs to wait before stopping the model if the validation loss does not decrease
			pretraining (bool): whether the model is being pre-trained. Just used for saving model names.

		'''

		# initializing the variables
		self.decreasing_loss_patience = decreasing_loss_patience
		self.loss_counter = 0
		self.training_counter = 0
		self.best_score = None
		self.early_stop = False
		self.best_epoch = None

	def __call__(self, train_loss, val_loss, model, optimizer, epoch):
		'''
		Function to call the early stopping condition.

		Args:
			train_loss (float): the training loss for the model
			val_loss (float): the validation loss for the model
			model (object): the model to be saved
			epoch (int): the current epoch

		Returns:
			bool: whether the model should stop training or not
		'''

		# using the absolute value of the loss for negatively orientied loss functions
		# val_loss = abs(val_loss)

		# initializing the best score if it is not already
		self.model = model
		self.optimizer = optimizer
		if self.best_score is None:
			self.best_train_loss = train_loss
			self.best_score = val_loss
			self.best_loss = val_loss
			self.save_checkpoint(val_loss)
			self.best_epoch = epoch

		# if the validation loss greater than the best score add one to the loss counter
		elif val_loss >= self.best_score:
			self.loss_counter += 1

			# if the loss counter is greater than the patience, stop the model training
			if self.loss_counter >= self.decreasing_loss_patience:
				gc.collect()
				print(f'Engaging Early Stopping due to lack of improvement in validation loss. Best model saved at epoch {self.best_epoch} with a training loss of {self.best_train_loss} and a validation loss of {self.best_score}')
				return True

		# if the validation loss is less than the best score, reset the loss counter and use the new validation loss as the best score
		else:
			self.best_train_loss = train_loss
			self.best_score = val_loss
			self.best_epoch = epoch

			# saving the best model as a checkpoint
			self.save_checkpoint(val_loss)
			self.loss_counter = 0
			self.training_counter = 0

			return False

	def save_checkpoint(self, val_loss):
		'''
		Function to continually save the best model.

		Args:
			val_loss (float): the validation loss for the model
		'''

		# saving the model if the validation loss is less than the best loss
		self.best_loss = val_loss
		print('Saving checkpoint!')

		torch.save({'model': self.model.state_dict(),
					'optimizer':self.optimizer.state_dict(),
					'best_epoch':self.best_epoch,
					'finished_training':False},
					f'models/{TARGET}/region_{REGION}_{VERSION}.pt')


def resume_training(model, optimizer):
	'''
	Function to resume training of a model if it was interupted without completeing.

	Args:
		model (object): the model to be trained
		optimizer (object): the optimizer to be used
		pretraining (bool): whether the model is being pre-trained

	Returns:
		object: the model to be trained
		object: the optimizer to be used
		int: the epoch to resume training from
	'''

	try:
		checkpoint = torch.load(f'models/{TARGET}/region_{REGION}_{VERSION}.pt')
		model.load_state_dict(checkpoint['model'])
		optimizer.load_state_dict(checkpoint['optimizer'])
		epoch = checkpoint['best_epoch']
		finished_training = checkpoint['finished_training']
	except KeyError:
		model.load_state_dict(torch.load(f'models/{TARGET}/region_{REGION}_{VERSION}.pt'))
		optimizer = None
		epoch = 0
		finished_training = True

	return model, optimizer, epoch, finished_training


def fit_model(model, train, val, val_loss_patience=25, overfit_patience=5, num_epochs=500):

	'''
	_summary_: Function to train the swmag model.

	Args:
		model (object): the model to be trained
		train (torch.utils.data.DataLoader): the training data
		val (torch.utils.data.DataLoader): the validation data
		val_loss_patience (int): the number of epochs to wait before stopping the model
									if the validation loss does not decrease
		overfit_patience (int): the number of epochs to wait before stopping the model
									if the training loss is significantly lower than the
									validation loss
		num_epochs (int): the number of epochs to train the model
		pretraining (bool): whether the model is being pre-trained

	Returns:
		object: the trained model
	'''
	N_train = len(train)*CONFIG['batch_size']
	N_val = len(val)*CONFIG['batch_size']
	optimizer = optim.Adam(model.parameters(), lr=1e-7)
	# checking if the model has already been trained, loading it if it exists
	if os.path.exists(f'models/{TARGET}/region_{REGION}_{VERSION}.pt'):
		model, optimizer, current_epoch, finished_training = resume_training(model=model, optimizer=optimizer)
	else:
		finished_training = False
		current_epoch = 0

	if current_epoch is None:
		current_epoch = 0

	# checking to see if the model was already trained or was interupted during training
	if not finished_training:

		# initializing the lists to hold the training and validation loss which will be used to plot the losses as a function of epoch
		train_loss_list, val_loss_list = [], []

		# moving the model to the available device
		model.to(DEVICE)

		# defining the loss function and the optimizer
		criterion = CRSP()
		# criterion = ACCRUE()
		# criterion = nn.BCELoss()
		optimizer = optim.Adam(model.parameters(), lr=1e-7)

		# initalizing the early stopping class
		early_stopping = Early_Stopping(decreasing_loss_patience=val_loss_patience)

		# looping through the epochs
		while current_epoch < num_epochs:

			# starting the clock for the epoch
			stime = time.time()

			# setting the model to training mode
			model.train()

			# initializing the running loss
			running_training_loss, running_val_loss = 0.0, 0.0

			# using the training set to train the model
			for swmag, twins, y in train:

				# moving the data to the available device
				swmag = swmag.to(DEVICE, dtype=torch.float)
				twins = twins.to(DEVICE, dtype=torch.float)
				y = y.to(DEVICE, dtype=torch.float)

				# forward pass
				output = model(swmag, twins)

				output = output.squeeze()

				# calculating the loss
				loss = criterion(output, y)

				# backward pass
				optimizer.zero_grad()
				loss.backward()
				optimizer.step()

				# emptying the cuda cache
				swmag = swmag.to('cpu')
				twins = twins.to('cpu')
				y = y.to('cpu')

				# adding the loss to the running training loss
				running_training_loss += loss.to('cpu').item()


			# setting the model to eval mode so the dropout layers are not used during validation and weights are not updated
			model.eval()

			# using validation set to check for overfitting
			# looping through the batches
			for swmag, twins, y in val:

				# moving the data to the available device
				swmag = swmag.to(DEVICE, dtype=torch.float)
				twins = twins.to(DEVICE, dtype=torch.float)
				y = y.to(DEVICE, dtype=torch.float)

				# forward pass with no gradient calculation
				with torch.no_grad():

					output = model(swmag, twins)
					# output = output.view(len(output),2)
					output = output.squeeze()

					val_loss = criterion(output, y)

					# emptying the cuda cache
					swmag = swmag.to('cpu')
					twins = twins.to('cpu')
					y = y.to('cpu')

					# adding the loss to the running val loss
					running_val_loss += val_loss.to('cpu').item()

			# getting the average loss for the epoch
			loss = running_training_loss/len(train)
			val_loss = running_val_loss/len(val)

			# adding the loss to the list
			train_loss_list.append(loss)
			val_loss_list.append(val_loss)

			# checking for early stopping or the end of the training epochs
			if (early_stopping(train_loss=loss, val_loss=val_loss, model=model, optimizer=optimizer, epoch=current_epoch)) or (current_epoch == num_epochs-1):

				# saving the final model
				gc.collect()

				# clearing the cuda cache
				torch.cuda.empty_cache()
				gc.collect()
				# breaking the loop
				break

			# getting the time for the epoch
			epoch_time = time.time() - stime

			# printing the loss for the epoch
			print(f'Epoch [{current_epoch}/{num_epochs}], Loss: {loss:.4f} Validation Loss: {val_loss:.4f}' + f' Epoch Time: {epoch_time:.2f} seconds')

			# emptying the cuda cache
			torch.cuda.empty_cache()

			# updating the epoch
			current_epoch += 1

		# transforming the lists to a dataframe to be saved
		loss_tracker = pd.DataFrame({'train_loss':train_loss_list, 'val_loss':val_loss_list})
		loss_tracker.to_feather(f'outputs/{VERSION}_loss_tracker.feather')

		gc.collect()

	else:
		# loading the model if it has already been trained.
		try:
			final = torch.load(f'models/{TARGET}/region_{REGION}_{VERSION}.pt')
			model.load_state_dict(final['model'])
		except KeyError:
			model.load_state_dict(torch.load(f'models/{TARGET}/region_{REGION}_{VERSION}.pt'))

	return model


def evaluation(model, test, test_dates):
	'''
	Function using the trained models to make predictions with the testing data.

	Args:
		model (object): pre-trained model
		test_dict (dict): dictonary with the testing model inputs and the real data for comparison
		split (int): which split is being tested

	Returns:
		dict: test dict now containing columns in the dataframe with the model predictions for this split
	'''
	print(f'length of test dates: {len(test_dates)}')
	# creting an array to store the predictions
	predicted_mean, predicted_std, swmag_list, twins_list, ytest_list = [], [], [], [], []
	# setting the encoder and decoder into evaluation model
	model.eval()

	# creating a loss value
	running_loss = 0.0

	# making sure the model is on the correct device
	model.to(DEVICE, dtype=torch.float)

	with torch.no_grad():

		for swmag, twins, y in test:

			swmag = swmag.to(DEVICE, dtype=torch.float)
			twins = twins.to(DEVICE, dtype=torch.float)
			y = y.to(DEVICE, dtype=torch.float)

			predicted = model(swmag, twins)

			predicted = predicted.squeeze()

			# getting shape of tensor
			loss = F.mse_loss(predicted[:,0], y)
			running_loss += loss.item()

			# making sure the predicted value is on the cpu
			if predicted.get_device() != -1:
				predicted = predicted.to('cpu')
			if swmag.get_device() != -1:
				swmag = swmag.to('cpu')
			if twins.get_device() != -1:
				twins = twins.to('cpu')
			if y.get_device() != -1:
				y = y.to('cpu')

			# adding the decoded result to the predicted list after removing the channel dimension
			predicted = torch.squeeze(predicted, dim=1).numpy()

			predicted_mean.append(predicted[:,0])
			predicted_std.append(predicted[:,1])

			swmag = torch.squeeze(swmag, dim=1).numpy()
			twins = torch.squeeze(twins, dim=1).numpy()

			swmag_list.append(swmag)
			twins_list.append(twins)
			ytest_list.append(y)

	print(f'Evaluation Loss: {running_loss/len(test)}')

	# transforming the lists to arrays
	predicted_mean = np.concatenate(predicted_mean, axis=0)
	predicted_std = np.concatenate(predicted_std, axis=0)
	swmag_list = np.concatenate(swmag_list, axis=0)
	twins_list = np.concatenate(twins_list, axis=0)
	ytest_list = np.concatenate(ytest_list, axis=0)

	results_df = pd.DataFrame({'predicted_mean':predicted_mean, 'predicted_std':predicted_std, 'actual':ytest_list, 'dates':test_dates['Date_UTC']})

	print(f'results df shape: {results_df.shape}')
	print(f'results df: {results_df.head()}')

	return results_df


def main():
	'''
	Pulls all the above functions together. Outputs a saved file with the results.

	'''
	if not os.path.exists(f'outputs/{TARGET}'):
		os.makedirs(f'outputs/{TARGET}')
	if not os.path.exists(f'models/{TARGET}'):
		os.makedirs(f'models/{TARGET}')

	# loading all data and indicies
	print('Loading data...')
	train_swmag, train_twins, ytrain, val_swmag, val_twins, yval, test_swmag, test_twins, ytest, dates_dict = utils.getting_prepared_data(target_var=TARGET, cluster=CLUSTER, region=REGION, version=VERSION, config=CONFIG)
	print(f'shape of train swmag: {train_swmag.shape}; shape of train twins: {train_twins.shape}; shape of train y: {ytrain.shape}')
	print(f'shape of val swmag: {val_swmag.shape}; shape of val twins: {val_twins.shape}; shape of val y: {yval.shape}')
	print(f'shape of test swmag: {test_swmag.shape}; shape of test twins: {test_twins.shape}; shape of test y: {ytest.shape}')
	
	with open(f'outputs/dates_dict_region_{REGION}_version_{VERSION}.pkl', 'wb') as f:
		pickle.dump(dates_dict, f)

	# creating the dataloaders
	train = DataLoader(list(zip(train_swmag, train_twins, ytrain)), batch_size=CONFIG['batch_size'], shuffle=True)
	val = DataLoader(list(zip(val_swmag, val_twins, yval)), batch_size=CONFIG['batch_size'], shuffle=True)
	test = DataLoader(list(zip(test_swmag, test_twins, ytest)), batch_size=CONFIG['batch_size'], shuffle=False)

	# batch = next(iter(train))
	# print(f'batch shape: {batch[0]}')
	# creating the model
	print('Creating model....')

	# setting random seed
	torch.manual_seed(CONFIG['random_seed'])
	torch.cuda.manual_seed(CONFIG['random_seed'])

	model = TWINSModel()

	# printing model summary
	model.to(DEVICE)

	# fitting the model
	print('Fitting model...')
	model = fit_model(model, train, val, val_loss_patience=25, num_epochs=CONFIG['epochs'])

	# clearing the model so the best one can be loaded without overwhelming the gpu memory
	model = None
	model = TWINSModel()

	# loading the best model version
	final = torch.load(f'models/{TARGET}/region_{REGION}_{VERSION}.pt')

	# setting the finished training flag to True
	final['finished_training'] = True

	# getting the best model state dict
	model.load_state_dict(final['model'])

	# saving the final model
	torch.save(final, f'models/{TARGET}/region_{REGION}_{VERSION}.pt')


	# making predictions
	print('Making predictions...')
	results_df = evaluation(model, test, dates_dict['test'])
	print(f'shape of results df: {results_df.shape}')
	print(results_df.tail())
	results_df.to_feather(f'outputs/{TARGET}/twins_modeling_region_{REGION}_version_{VERSION}.feather')

	# clearing the session to prevent memory leaks
	gc.collect()


if __name__ == '__main__':

	
	args = argparse.ArgumentParser(description='Modeling the SWMAG data')
	args.add_argument('--target', type=str, help='The target variable to be modeled')
	args.add_argument('--region', type=str, help='The region to be modeled')
	args.add_argument('--cluster', type=str, help='The cluster containing the region to be modeled')
	args.add_argument('--version', type=str, help='The version of the model to be run')
	args.add_argument('--oversampling', type=str, help='Whether to oversample the data or not', default='False')

	args = args.parse_args()

	def str2bool(v):
		if isinstance(v, bool):
			return v
		if v.lower() in ('yes', 'true', 't', 'y', '1'):
			return True
		elif v.lower() in ('no', 'false', 'f', 'n', '0'):
			return False
		else:
			raise argparse.ArgumentTypeError('Boolean value expected.')


	# global TARGET
	# global REGION
	# global CLUSTER
	if str2bool(args.oversampling):
		VERSION = args.version+'_oversampling'
	else:
		VERSION = args.version

	TARGET = args.target
	REGION = args.region
	CLUSTER = args.cluster
	OVERSAMPLING = str2bool(args.oversampling)

	main()

	print('It ran. God job!')
