import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold

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
	
def KFold_cv(estimator,X,y,k=5):
	"""
	Perform k-fold cross-validation
	Returns: actual and predicted (for test set), train scores, test scores
	
	Args:
		estimator: sklearn estimator instance
		X: data matrix (nxm)
		y: response (n-vector)
		k: number of folds. Default 5
	"""
	kf = KFold(k,shuffle=True,random_state=7)
	train_scores = np.empty(k)
	test_scores = np.empty(k)
	actual = np.array([])
	pred = np.array([])
	for i, (train_index,test_index) in enumerate(kf.split(X)):
		X_train, X_test = X[train_index], X[test_index]
		y_train, y_test = y[train_index], y[test_index]
		
		estimator.fit(X_train,y_train)
		train_scores[i] = estimator.score(X_train,y_train)
		test_scores[i] = estimator.score(X_test,y_test)
		actual = np.concatenate((actual,y_test))
		pred = np.concatenate((pred, estimator.predict(X_test)))
	
	return actual, pred, train_scores, test_scores
	
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
	"""
	y, y_pred, train_scores, test_scores = KFold_cv(estimator,X,y,k=k)
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
	
	return train_scores, test_scores

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
	
def plot_loco_cv(estimator_class,X,y,clusters,standardize=True,ncol=3,sharex=False,sharey=False,**params):
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
	
	est = estimator_class(**params)
	
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
		
	#plot test cluster means
	fig3, ax3 = plt.subplots()
	for cluster, act, pred in zip(unique_clusters,act_means,pred_means):
		ax3.scatter(act,pred,label=cluster)
	ax3.legend()
	ax3.set_xlabel('Actual Cluster Mean')
	ax3.set_ylabel('Predicted Cluster Mean')
	axmin = min(min(act_means),min(pred_means))
	axmax = max(max(act_means),max(pred_means))
	ax3.plot([axmin,axmax],[axmin,axmax],'g-',label='Ideal')
	ax3.legend()
	

	
