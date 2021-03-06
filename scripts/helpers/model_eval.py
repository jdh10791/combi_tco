# Tools for model testing and evaluation
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from stats_misc import r_squared

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold, LeaveOneGroupOut

from mlxtend.feature_selection import SequentialFeatureSelector as SFS

def multi_min(arrays):
	"""
	Return the minimum scalar value of multiple arrays
	
	Args:
		arrays: list of numpy arrays
	"""
	mins = []
	for arr in arrays:
		mins.append(np.min(arr))
	return min(mins)

def multi_max(arrays):
	"""
	Return the maximum scalar value of multiple arrays
	
	Args:
		arrays: list of numpy arrays
	"""
	maxs = []
	for arr in arrays:
		maxs.append(np.max(arr))
	return max(maxs)
	
class repeating_KFold():
	"""
	KFold splitter that performs multiple independent splits of the dataset
	Intended for use with shuffle=True to reduce bias for one particular train-test split
	
	Args:
		repeat: int, number of times to repeat
		kw: kwargs to pass to KFold
	"""
	def __init__(self,repeat,**kw):
		self.repeat = repeat
		self.kf = KFold(**kw)
		
	def split(self,X,y=None,groups=None):
		for n in range(self.repeat):
			for train,test in self.kf.split(X,y,groups):
				yield train,test
	
def KFold_cv(estimator,X,y,sample_weight=None,k=5,pipeline_learner_step=1):
	"""
	Perform k-fold cross-validation
	Returns: actual and predicted (for test set), train scores, test scores
	
	Args:
		estimator: sklearn estimator instance
		X: data matrix (nxm)
		y: response (n-vector)
		sample_weight: weights for fitting data. If None, defaults to equal weights
		k: number of folds. Default 5
		pipeline_learner_step: if estimator is a Pipeline instance, index of the learner step
	"""
	kf = KFold(k,shuffle=True)
	train_scores = np.empty(k)
	test_scores = np.empty(k)
	actual = np.zeros_like(y)
	pred = np.zeros_like(y)
	for i, (train_index,test_index) in enumerate(kf.split(X)):
		X_train, X_test = X[train_index], X[test_index]
		y_train, y_test = y[train_index], y[test_index]
		if sample_weight is not None:
			w_train = sample_weight[train_index]
			w_test = sample_weight[test_index]
		
		if sample_weight is not None:
			if type(estimator)==Pipeline:
				#if estimator is a Pipeline, need to specify name of learning step in fit_params for sample_weight
				est_name = estimator.steps[pipeline_learner_step][0]
				estimator.fit(X_train,y_train,**{f'{est_name}__sample_weight':w_train})
				train_scores[i] = estimator.score(X_train,y_train,**{f'{est_name}__sample_weight':w_train})
				test_scores[i] = estimator.score(X_test,y_test,**{f'{est_name}__sample_weight':w_test})
			else:
				estimator.fit(X_train,y_train,sample_weight=w_train)
				train_scores[i] = estimator.score(X_train,y_train,sample_weight=w_train)
				test_scores[i] = estimator.score(X_test,y_test,sample_weight=w_test)
		else:
			# not all estimators' fit() methods accept sample_weight arg - can't just pass None
			estimator.fit(X_train,y_train)
			train_scores[i] = estimator.score(X_train,y_train)
			test_scores[i] = estimator.score(X_test,y_test)
		actual[test_index] = y_test
		pred[test_index] = estimator.predict(X_test)
	
	return actual, pred, train_scores, test_scores
	
def repeated_KFold_cv(estimator,X,y,repeat,sample_weight=None,k=5,pipeline_learner_step=1):
	actuals = np.empty((repeat,len(y)))
	preds = np.empty_like(actuals)
	tot_test_scores = np.empty(repeat)
	for n in range(repeat):
		act,pred,train,test = KFold_cv(estimator,X,y,sample_weight,k)
		tot_test_score = r_squared(act,pred,w=sample_weight)
		actuals[n] = act
		preds[n] = pred
		tot_test_scores[n] = tot_test_score

	# actual = np.mean(actuals,axis=0)
	# pred = np.mean(preds,axis=0)
	# mean_tot_score = np.mean(tot_test_scores)
	return actuals, preds, tot_test_scores
	
def KFold_pva(estimator,X,y,k=5,ax=None,logscale=False,s=10):
	"""
	Perform k-fold cross-validation and plot predicted vs. actual for test set
	
	Args:
		estimator: sklearn estimator instance
		X: data matrix (nxm)
		y: response (n-vector)
		k: number of folds. Default 5
		ax: axis on which to plot
		logscale: if True, plot as log-log 
		s: marker size
		
	Returns: 
		train_scores: k-array of train scores
		test_scores: k-array of test scores
		tot_test_score: overall test score (r2) considering all test folds together 
	"""
	y, y_pred, train_scores, test_scores = KFold_cv(estimator,X,y,k=k)
	tot_test_score = r_squared(y,y_pred)
	if ax is None:
		fig, ax = plt.subplots()
	axmin = multi_min([y,y_pred])
	axmax = multi_max([y,y_pred])
	if logscale==False:
		ax.scatter(y,y_pred,s=s)
		ax.plot([axmin,axmax],[axmin,axmax],'g')
	elif logscale==True:
		ax.loglog(y,y_pred,'o',markersize=s)
		ax.loglog([axmin,axmax],[axmin,axmax],'g')
	ax.set_xlabel('Actual')
	ax.set_ylabel('Predicted')
	
	return train_scores, test_scores, tot_test_score
	
def repeated_KFold_pva(estimator,X,y,repeat,plot_type='series',sample_weight=None,k=5,pipeline_learner_step=1,ax=None,logscale=False,s=10):
	"""
	Perform k-fold cross-validation and plot predicted vs. actual for test set
	
	Args:
		estimator: sklearn estimator instance
		X: data matrix (nxm)
		y: response (n-vector)
		sample_weight: weights for fitting data. If None, defaults to equal weights
		k: number of folds. Default 5
		pipeline_learner_step: if estimator is a Pipeline instance, index of the learner step
		ax: axis on which to plot
		logscale: if True, plot as log-log 
		s: marker size
		
	Returns: 
		train_scores: k-array of train scores
		test_scores: k-array of test scores
		tot_test_score: overall test score (r2) considering all test folds together 
	"""
	actuals, preds, tot_test_scores = repeated_KFold_cv(estimator,X,y,repeat,sample_weight,k,pipeline_learner_step)
	if ax is None:
		fig, ax = plt.subplots()
	axmin = multi_min([np.min(actuals),np.min(preds)])
	axmax = multi_max([np.max(actuals),np.max(preds)])
	
	if plot_type=='series':
		if logscale==False:
			for y, y_pred in zip(actuals, preds):
				ax.scatter(y,y_pred,s=s)
				ax.plot([axmin,axmax],[axmin,axmax],'g')
		elif logscale==True:
			for y, y_pred in zip(actuals, preds):
				ax.loglog(y,y_pred,'o',markersize=s)
				ax.loglog([axmin,axmax],[axmin,axmax],'g')
	elif plot_type=='mean':
		y = np.mean(actuals,axis=0)
		y_pred = np.mean(preds,axis=0)
		if logscale==False:
			ax.scatter(y,y_pred,s=s)
			ax.plot([axmin,axmax],[axmin,axmax],'g')
		elif logscale==True:
			ax.loglog(y,y_pred,'o',markersize=s)
			ax.loglog([axmin,axmax],[axmin,axmax],'g')
	
	ax.set_xlabel('Actual')
	ax.set_ylabel('Predicted')
	
	return tot_test_scores

def plot_pva(estimator,X,y,ax=None,logscale=False,s=10):
	"""
	Plot predicted vs. actual for estimator

	Args:
		estimator: fitted sklearn estimator instance
		X: data matrix (nxm)
		y: response (n-vector)
		ax: axis on which to plot
		logscale: if True, plot as log-log 
		s: marker size
	"""
	y_pred = estimator.predict(X)
	if ax is None:
		fig, ax = plt.subplots()
	
	axmin = multi_min([y,y_pred])
	axmax = multi_max([y,y_pred])
	if logscale==False:
		ax.scatter(y,y_pred,s=s)
		ax.plot([axmin,axmax],[axmin,axmax],'g')
	elif logscale==True:
		ax.loglog(y,y_pred,'o',markersize=s)
		ax.loglog([axmin,axmax],[axmin,axmax],'g')
	ax.set_xlabel('Actual')
	ax.set_ylabel('Predicted')	
	
def poly_transform(X,powers,inf_val=1e6,add_exp=False):
	"""
	Perform polynomial transform of data matrix
	Each column of X is raised to all of the powers specified
	
	Args:
		X: data matrix (n x m)
		powers: list or array of powers to apply (p-vector)
		inf_val: value with which to replace infs (infs may arise if zeros are raised to negative powers)
		add_exp: if True, add an exponential of each variable to the output matrix
	Returns: transformed matrix (n x (m*p))
	"""
	#each power needs to be applied to each column
	cpowers = np.array(powers*X.shape[1])
	Xr = X.repeat(len(powers),axis=1)

	#matrix to track and retain signs
	Xr_sign = np.sign(Xr)
	#zeros get 0 sign - change to 1
	Xr_sign[Xr_sign==0] = 1
	#elements that will be raised to even powers must be positive after operation - set signs to 1
	power_flag = (cpowers%2!=0).astype(int) #flag for non-even powers in cpowers
	Xr_sign = np.apply_along_axis(lambda x: x**power_flag, 1, Xr_sign) 
		#if even, flag = 0 -> sign becomes 1
		#if not even, flag=1 -> sign unchanged
	

	#feed absolute values in to allow square root of negatives, then reapply signs
	Xt = Xr_sign*(np.abs(Xr)**cpowers)
	
	#add exponential terms if specified
	if add_exp==True:
		#create new array with extra columns for exponentials
		Xt2 = np.zeros((Xt.shape[0], Xt.shape[1] + 2*X.shape[1]))
		for i in range(X.shape[1]):
			#fill leftmost columns with polynomial terms
			Xt2[:,i*(len(powers) + 2):(i+1)*(len(powers) + 2) - 2] = Xt[:,i*len(powers):(i+1)*len(powers)]
			#fill last 2 columns for each variable with exponential terms
			Xt2[:,(i+1)*(len(powers) + 2) - 2] = np.exp(-X[:,i]) #e^-x
			Xt2[:,(i+1)*(len(powers) + 2) - 1] = np.exp(X[:,i]) #e^x
		Xt = Xt2
		
	#set infinite values to inf_val
	Xt[Xt==np.inf] = inf_val
	
	#check for nans
	if multi_max((np.isnan(Xt),np.isinf(Xt)))==True:
		print('Warning: transformed array contains NaNs or infs')
	else:
		print('Transformed matrix is clean')
	
	return Xt

def plot_clustered_cv(estimator_class,X,y,clusters,standardize=True,ncol=3,sharex=False,sharey=False,**params):
	"""
	Perform traditional k-fold cross validation within each individual cluster
	Plot train and test predicted vs. actual plots
	Print train and test scores
	
	Args:
		estimator_class: sklearn estimator class
		X: data matrix (nxm)
		y: response (n-vector)
		clusters: list of cluster labels for observations (n-vector)
		standardize: if True, standardize inputs (center and whiten)
		ncol: number of columns for subplot grids. Default 3
		params: keyword args for estimator instantiation
	"""

	unique_clusters = np.unique(clusters)
	num_clusters = len(unique_clusters)

	train_scores = np.empty(num_clusters)
	test_scores = np.empty(num_clusters)

	nrow = int(np.ceil(num_clusters/ncol))
	#axes for train pva plots
	fig1, axes1 = plt.subplots(nrow,ncol,figsize=(ncol*3,nrow*3),sharex=sharex,sharey=sharey)
	#axes for test pva plots
	fig2, axes2 = plt.subplots(nrow,ncol,figsize=(ncol*3,nrow*3),sharex=sharex,sharey=sharey)

	if standardize==True:
		ss = StandardScaler()
		X = ss.fit_transform(X) #normalizing results in larger coefficients - less feature variance

	for i, (cluster,ax1,ax2) in enumerate(zip(unique_clusters,axes1.ravel(),axes2.ravel())):
		#train a model for each cluster
		cidx = np.where(clusters==cluster)
		#print(cluster,cidx)

		Xc = X[cidx]
		yc = y[cidx]

		lr = estimator_class(**params)
		lr.fit(Xc,yc)
		
		#plot training data
		ax1.set_title(cluster)
		plot_pva(lr,Xc,yc,ax=ax1,logscale=False)

		#plot test data
		ax2.set_title(cluster)
		train_score,test_score = KFold_pva(lr,Xc,yc,k=4,ax=ax2,logscale=False)
		train_scores[i] = np.mean(train_score)
		test_scores[i] = np.mean(test_score)

		#print(lr.intercept_)

	fig1.suptitle('Train',size=14)
	fig2.suptitle('Test',size=14)
	fig1.tight_layout()	 
	fig2.tight_layout()
	fig1.subplots_adjust(top=0.83)
	fig2.subplots_adjust(top=0.83)
	# axes1[1,2].axis('off')
	# axes2[1,2].axis('off')
	print("Train scores: ",train_scores,'\n\tMean: ', np.mean(train_scores))
	print("Test scores: ",test_scores,'\n\tMean: ', np.mean(test_scores))	
	
def plot_cluster_extrap(estimator_class,X,y,clusters,standardize=True,ncol=3,**params):
	"""
	Train single-cluster models and test extrapolation to remaining clusters
	For each training cluster, plot predicted vs. actual for each extrapolated cluster
	
	Args:
		estimator_class: sklearn estimator class
		X: data matrix (nxm)
		y: response (n-vector)
		clusters: list of cluster labels for observations (n-vector)
		standardize: if True, standardize inputs (center and whiten)
		ncol: number of columns for subplot grids. Default 3
		params: keyword args for estimator instantiation
	"""
	unique_clusters = np.unique(clusters)
	num_clusters = len(unique_clusters)

	nrow = int(np.ceil(num_clusters/ncol))

	train_scores = np.empty(len(unique_clusters))
	test_scores = np.empty((len(unique_clusters), len(unique_clusters)-1))

	if standardize==True:
		ss = StandardScaler()
		X = ss.fit_transform(X) #normalizing results in larger coefficients - less feature variance

	for i, cluster in enumerate(unique_clusters):
		#train a model to one cluster
		cidx = np.where(clusters==cluster)

		Xc = X[cidx]
		yc = y[cidx]

		lr = estimator_class(**params)
		lr.fit(Xc,yc)
		train_scores[i] = lr.score(Xc,yc)

		#axes for test pva plots
		fig, axes = plt.subplots(nrow,ncol,figsize=(ncol*3,nrow*3),sharex=True,sharey=True)

		#predict other clusters
		for j, oc in enumerate(unique_clusters[unique_clusters != cluster]):
			#oc = other cluster
			ax = axes[int(j/2),j%2]
			ax.set_title(oc)

			#get data for other cluster
			ocidx = np.where(clusters==oc)
			Xoc = X[ocidx]
			yoc = y[ocidx]

			#plot and score
			plot_pva(lr,Xoc,yoc,ax=ax,logscale=False)
			test_scores[i,j] = lr.score(Xoc,yoc)

		fig.suptitle(f'Model trained on {cluster}', size=14)
		fig.tight_layout()
		fig.subplots_adjust(top=0.9)
		print(f"{cluster} train score: ",train_scores[i])
		print(f"{cluster} test scores: ",test_scores[i],'\n\tMean: ', np.mean(test_scores[i]))
		
	print("Mean train score: ",np.mean(train_scores))
	print("Mean test score: ",np.mean(test_scores))
	
def loco_cv(estimator,X,y,clusters,standardize=True):
	"""
	Perform LOCO-CV
	
	Args:
		estimator: sklearn estimator instance
		X: data matrix (nxm)
		y: response (n-vector)
		clusters: list of cluster labels for observations (n-vector)
		standardize: if True, standardize inputs (center and whiten)
		
	Returns: 
		train_scores: training set scores 
		test_scores: test set scores (one score per cluster)
		test_pred: test set predicted values
		test_act: test set actual values
		train_pred: test set predicted values 
		train_act: train set predicted values
	"""
	if type(X) == pd.core.frame.DataFrame:
		X = X.values
	if type(y) == pd.core.series.Series:
		y = y.values
		
	
	if standardize==True:
		ss = StandardScaler()
		X = ss.fit_transform(X)
	
	unique_clusters = np.unique(clusters)
	num_clusters = len(unique_clusters)
	
	train_scores = np.empty(num_clusters)
	test_scores = np.empty(num_clusters)
	train_act = []
	train_pred = []
	test_act = []
	test_pred = []
	for i, c in enumerate(unique_clusters):
		print(f'Test cluster {c}')
		train_idx = np.where(clusters!=c)
		test_idx = np.where(clusters==c)
		X_train = X[train_idx]
		y_train= y[train_idx]
		X_test = X[test_idx]
		y_test = y[test_idx]
		estimator.fit(X_train,y_train)
		train_pred.append(estimator.predict(X_train))
		train_act.append(y_train)
		test_pred.append(estimator.predict(X_test))
		test_act.append(y_test)
		train_scores[i] = estimator.score(X_train,y_train)
		test_scores[i] = estimator.score(X_test,y_test)
	
	return train_scores, test_scores, test_pred, test_act, train_pred, train_act
	
def plot_loco_cv(estimator_class,X,y,clusters,standardize=True,ncol=4,sharex=False,sharey=False,**params):
	"""
	Perform LOCO-CV. Plot train and test predicted vs. actual for each cluster, as well as predicted vs. actual cluster means
	
	Args:
		estimator: sklearn estimator class
		X: data matrix (nxm)
		y: response (n-vector)
		clusters: list of cluster labels for observations (n-vector)
		standardize: if True, standardize inputs (center and whiten)
		ncol: number of columns for subplot grids. Default 3
		params: keyword args for estimator instantiation
	"""
	try:
		est = estimator_class(**params)
	except TypeError:
		est = estimator_class
	
	train_scores, test_scores, test_pred, test_act, train_pred, train_act = loco_cv(est,X,y,clusters,standardize=standardize)
	print("Train scores: ", train_scores, "\n\tMean train score: ", np.mean(train_scores))
	print("Test scores: ", test_scores, "\n\tMean test score: ", np.mean(test_scores))
#	  fig, ax = plt.subplots()
#	  ax.scatter(train_scores, test_scores)
#	  ax.scatter(np.mean(train_scores), np.mean(test_scores),marker='d')
#	  ax.axhline(0,ls='--')
#	  ax.axvline(0,ls='--')
#	  ax.set_xlabel('train score')
#	  ax.set_ylabel('test score')
	
	#plot train data
	# fig1, ax1 = plt.subplots()
	# for i, (pred, act) in enumerate(zip(train_pred, train_act)):
		# ax1.scatter(act, pred,label=i,s=8)
	# axmin = multi_min([ax1.get_xlim(),ax1.get_ylim()])
	# axmax = multi_max([ax1.get_xlim(),ax1.get_ylim()])
	# ax1.plot([axmin,axmax],[axmin,axmax],label='Ideal')
	# ax1.legend()
	
	
	#plot predicted vs. actual
	unique_clusters = np.unique(clusters)
	num_clusters = len(unique_clusters)
	nrow = int(np.ceil(num_clusters/ncol))
	#axes for train data
	fig1, axes1 = plt.subplots(nrow,ncol,figsize=(ncol*3,nrow*3),sharex=sharex,sharey=sharey)
	#axes for test data
	fig2, axes2 = plt.subplots(nrow,ncol,figsize=(ncol*3,nrow*3),sharex=sharex,sharey=sharey)
	act_means = []
	pred_means = []
	for i, (c,ax1,ax2) in enumerate(zip(unique_clusters,axes1.ravel(),axes2.ravel())):
		for ax in ax1, ax2:
		#plot test data
			if ax==ax1:
				act, pred = train_act[i], train_pred[i]
			elif ax==ax2:
				act, pred = test_act[i], test_pred[i]
			ax.scatter(act,pred,s=6)
			ax.set_title(c)
			axmin = multi_min((act,pred))
			axmax = multi_max((act,pred))
			ax.plot([axmin,axmax],[axmin,axmax],'g-',label='Ideal')
			ax.axhline(np.mean(act),ls='--',c='k',label='Actual Cluster Mean')
			ax.axhline(np.mean(pred),ls='--',c='r',label='Predicted Cluster Mean')
			ax.set_xlabel('Actual')
			ax.set_ylabel('Predicted')
		#ax.legend()
		
		fig1.suptitle('Train',size=14)
		fig2.suptitle('Test',size=14)
		for fig in fig1,fig2:
			fig.tight_layout()
			fig.subplots_adjust(top=0.8)
		
		act_means.append(np.mean(test_act[i]))
		pred_means.append(np.mean(test_pred[i]))
		
	#plot (a) all test data on single plot and (b) test cluster means
	fig3, axes3 = plt.subplots(1,2,figsize=(12,4))
	#(a) all test points
	for cluster, act, pred in zip(unique_clusters,test_act,test_pred):
		axes3[0].scatter(act,pred,label=cluster,s=6)
	axes3[0].legend()	
	axes3[0].set_xlabel('Actual')
	axes3[0].set_ylabel('Predicted')
	axmin = multi_min((multi_min(test_act),multi_min(test_pred)))
	axmax = multi_max((multi_max(test_act),multi_max(test_pred)))
	axes3[0].plot([axmin,axmax],[axmin,axmax],'g-',label='Ideal')
	
	#(b) cluster means
	for cluster, act, pred in zip(unique_clusters,act_means,pred_means):
		axes3[1].scatter(act,pred,label=cluster)
	axes3[1].legend()
	axes3[1].set_xlabel('Actual Cluster Mean')
	axes3[1].set_ylabel('Predicted Cluster Mean')
	axmin = multi_min((act_means,pred_means))
	axmax = multi_max((act_means,pred_means))
	axes3[1].plot([axmin,axmax],[axmin,axmax],'g-',label='Ideal')
	
	fig3.tight_layout()
	
	
class LeaveOneClusterOut():
	"""
	Wrapper for sklearn LeaveOneGroupOut splitter
	Stores clusters as attribute (rather than fit param) as a workaround to enable LOCO-CV in mlxtend SequentialFeatureSelector
	
	Args:
		clusters: list of cluster labels for observations (n-vector)
	"""
	def __init__(self,clusters):
		self.logo = LeaveOneGroupOut()
		self.clusters = clusters
	
	def get_n_splits(self,X=None,y=None,groups=None):
		return self.logo.get_n_splits(groups=self.clusters)
	
	def split(self,X,y=None,groups=None):
		return self.logo.split(X,y,groups=self.clusters)
		

def single_cluster_val_with_selection(cv_estimator,select_estimator,X,y,clusters,test_cluster,k_features,selector_cv,standardize=False,**selector_params):
	"""
	Predict a single test cluster using remaining clusters for training and feature selection (runs a single fold of LOCO-CV)
	
	Args:
		cv_estimator: sklearn estimator instance to use for training and predicting response
		select_estimator: sklearn estimator instance to use for feature selection
		X: data matrix (nxm). numpy array or pandas dataframe
		y: response (n-vector)
		clusters: list of cluster labels for observations (n-vector)
		test_cluster: cluster to use as test set
		k_features: number of features to select
		selector_cv: int, sklearn splitter instance (e.g. KFold()), or LeaveOneClusterOut class for feature selection cross validation.
		standardize: if True, standardize inputs (center and whiten)
		selector_params: keywords for SequentialFeatureSelector
		
	Returns: 
		train_scores: training set scores 
		test_scores: test set scores
		test_pred: test set predicted values
		test_act: test set actual values
		train_pred: test set predicted values 
		train_act: train set predicted values
		selected_features: selected features
	"""
	
	if type(X)==pd.core.frame.DataFrame:
		X_df = X
		X = X.values
		X_is_df=True
	else:
		X_is_df=False
	
	train_idx = np.where(clusters!=test_cluster)
	test_idx = np.where(clusters==test_cluster)
	X_train = X[train_idx]
	y_train = y[train_idx]
	
	if standardize==True:
		#scale training data to zero mean and unit variance
		ss = StandardScaler()
		X_train = ss.fit_transform(X_train)
	
	#select features based on training data
	if selector_cv==LeaveOneClusterOut:
		cv = selector_cv(clusters[train_idx])
		sfs = SFS(select_estimator,k_features=k_features,cv=cv,**selector_params)
	else:
		sfs = sfs = SFS(select_estimator,k_features=k_features,cv=selector_cv,**selector_params)
	
	if X_is_df:
		sfs.fit(X_train,y_train,custom_feature_names=X_df.columns.values)
		selected_features = list(sfs.k_feature_names_)
		X_select = X_df.loc[:,list(sfs.k_feature_names_)].values
	else:
		sfs.fit(X_train,y_train)
		selected_features = list(sfs.k_feature_idx_)
		X_select = X[:,list(sfs.k_feature_idx_)]
	
	#train model on training data with selected features only
	X_train_sel = X_select[train_idx]
	if standardize==True:
		#retrain scaler to selected features
		X_train_sel = ss.fit_transform(X_train_sel)
	cv_estimator.fit(X_train_sel,y_train)
	train_pred = cv_estimator.predict(X_train_sel)
	train_act = y_train
	train_scores = cv_estimator.score(X_train_sel,y_train)
	
	#predict test set with selected features
	X_test_sel = X_select[test_idx]
	if standardize==True:
		#apply scaler fitted to training data
		X_test_sel = ss.transform(X_test_sel)
	y_test = y[test_idx]
	test_pred = cv_estimator.predict(X_test_sel)
	test_act = y_test
	test_scores = cv_estimator.score(X_test_sel,y_test)
	
	return train_scores, test_scores, test_pred, test_act, train_pred, train_act, selected_features
		
def loco_cv_with_selection(cv_estimator,select_estimator,X,y,clusters,k_features,selector_cv,standardize=False,**selector_params):
	"""
	Use mlxtend sequential feature selection to choose k_features that produce best test score using LOCO-CV
	
	Args:
		cv_estimator: sklearn estimator instance to use for training and predicting response
		select_estimator: sklearn estimator instance to use for feature selection
		X: data matrix (nxm). numpy array or pandas dataframe
		y: response (n-vector)
		clusters: list of cluster labels for observations (n-vector)
		k_features: number of features to select
		selector_cv: int, sklearn splitter instance (e.g. KFold()), or LeaveOneClusterOut class for feature selection cross validation.
		standardize: if True, standardize inputs (center and whiten)
		selector_params: keywords for SequentialFeatureSelector
		
	Returns: 
		train_scores: training set scores 
		test_scores: test set scores (one score per cluster)
		test_pred: test set predicted values
		test_act: test set actual values
		train_pred: test set predicted values 
		train_act: train set predicted values
		selected_features: selected features for each training fold
	"""
	#for now - standardize once. Later should change to standardize as part of training
	# if standardize==True:
		# ss = StandardScaler()
		# X = ss.fit_transform(X)
	
	unique_clusters = np.unique(clusters)
	num_clusters = len(unique_clusters)
	
	train_scores = np.empty(num_clusters)
	test_scores = np.empty(num_clusters)
	train_act = []
	train_pred = []
	test_act = []
	test_pred = []
	selected_features = []
	for i, test_cluster in enumerate(unique_clusters):
		print(f'Test set: {test_cluster}')
		#for now - standardize once. Later should change to standardize as part of training
		train_s, test_s, test_p, test_a, train_p, train_a, sel_feat = single_cluster_val_with_selection(cv_estimator,select_estimator,X,y,clusters,test_cluster,k_features,selector_cv,standardize=False,**selector_params)
		train_scores[i],test_scores[i] = train_s, test_s
		test_pred.append(test_p)
		test_act.append(test_a)
		train_pred.append(train_p)
		train_act.append(train_a)
		selected_features.append(sel_feat)
	
	return train_scores, test_scores, test_pred, test_act, train_pred, train_act, selected_features
	
	
def plot_loco_cv_with_selection(cv_estimator,select_estimator,X,y,clusters,k_features,selector_cv,standardize=False,ncol=4,sharex=False,sharey=False,**selector_params):
	train_scores, test_scores, test_pred, test_act, train_pred, train_act, selected_features = loco_cv_with_selection(cv_estimator,select_estimator,X,y,clusters,k_features,selector_cv,standardize,**selector_params)
	print("Train scores: ", train_scores, "\n\tMean train score: ", np.mean(train_scores))
	print("Test scores: ", test_scores, "\n\tMean test score: ", np.mean(test_scores))	
	print("Selected features:")
	unique_clusters = np.unique(clusters)
	for cluster, features in zip(unique_clusters,selected_features):
		print(cluster,features)

	#plot predicted vs. actual	
	num_clusters = len(unique_clusters)
	nrow = int(np.ceil(num_clusters/ncol))
	#axes for train data
	fig1, axes1 = plt.subplots(nrow,ncol,figsize=(ncol*3,nrow*3),sharex=sharex,sharey=sharey)
	#axes for test data
	fig2, axes2 = plt.subplots(nrow,ncol,figsize=(ncol*3,nrow*3),sharex=sharex,sharey=sharey)
	act_means = []
	pred_means = []
	for i, (c,ax1,ax2) in enumerate(zip(unique_clusters,axes1.ravel(),axes2.ravel())):
		for ax in ax1, ax2:
		#plot test data
			if ax==ax1:
				act, pred = train_act[i], train_pred[i]
			elif ax==ax2:
				act, pred = test_act[i], test_pred[i]
			ax.scatter(act,pred,s=6)
			ax.set_title(c)
			axmin = multi_min((act,pred))
			axmax = multi_max((act,pred))
			ax.plot([axmin,axmax],[axmin,axmax],'g-',label='Ideal')
			ax.axhline(np.mean(act),ls='--',c='k',label='Actual Cluster Mean')
			ax.axhline(np.mean(pred),ls='--',c='r',label='Predicted Cluster Mean')
			ax.set_xlabel('Actual')
			ax.set_ylabel('Predicted')
		#ax.legend()
		
		fig1.suptitle('Train',size=14)
		fig2.suptitle('Test',size=14)
		for fig in fig1,fig2:
			fig.tight_layout()
			fig.subplots_adjust(top=0.8)
		
		act_means.append(np.mean(test_act[i]))
		pred_means.append(np.mean(test_pred[i]))
		
	#plot (a) all test data on single plot and (b) test cluster means
	fig3, axes3 = plt.subplots(1,2,figsize=(12,4))
	#(a) all test points
	for cluster, act, pred in zip(unique_clusters,test_act,test_pred):
		axes3[0].scatter(act,pred,label=cluster,s=6)
	axes3[0].legend()	
	axes3[0].set_xlabel('Actual')
	axes3[0].set_ylabel('Predicted')
	axmin = multi_min((multi_min(test_act),multi_min(test_pred)))
	axmax = multi_max((multi_max(test_act),multi_max(test_pred)))
	axes3[0].plot([axmin,axmax],[axmin,axmax],'g-',label='Ideal')
	
	#(b) cluster means
	for cluster, act, pred in zip(unique_clusters,act_means,pred_means):
		axes3[1].scatter(act,pred,label=cluster)
	axes3[1].legend()
	axes3[1].set_xlabel('Actual Cluster Mean')
	axes3[1].set_ylabel('Predicted Cluster Mean')
	axmin = multi_min((act_means,pred_means))
	axmax = multi_max((act_means,pred_means))
	axes3[1].plot([axmin,axmax],[axmin,axmax],'g-',label='Ideal')
	
	fig3.tight_layout()
	
	return dict(zip(unique_clusters,selected_features))
