import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import KernelPCA
import matplotlib.pyplot as plt
from .plotting import scatter_slices, add_colorbar
from .quaternary_plt import QuaternaryAxes
import pymatgen as mg
import matplotlib as mpl

def z_score(x):
	mu = np.mean(x)
	std = np.std(x)
	return (x-mu)/std

class DataCleaner():
	"""
	Class for outlier detection and data cleaning in preprocessing
	Implements sklearn.cluster.DBSCAN for compositional clustering, 
	and sklearn.ensemble.IsolationForest for greedy outlier flagging.
	Applies z-score threshold within composition clusters to screen 
	IsolationForest flags.
	
	Parameters
	----------
	data: dataset to process (pandas DataFrame)
	prop_dim: property dimension to screen for outliers
	comp_dims: composition dimensions for clustering and IsolationForest
	DB_kw: kwargs to pass to DBSCAN instantiation
	IF_kw: kwargs to pass to IsolationForest instantiation
	"""
	def __init__(self, data, prop_dim, comp_dims=None, DB_kw={},IF_kw={}):
		self.data = data
		self.set_prop_dim(prop_dim)
		self.set_comp_dims(comp_dims)
		self.random_state = np.random.RandomState(17)
		self.db = DBSCAN(**DB_kw)
		self.clf = IsolationForest(random_state=self.random_state,**IF_kw)
				 
	def set_prop_dim(self,prop_dim):
		"set property dimension"
		self._prop_dim = prop_dim
	
	def get_prop_dim(self):
		"get property dimension"
		return self._prop_dim
	
	prop_dim = property(get_prop_dim,set_prop_dim)
	
	def set_comp_dims(self,comp_dims=None):
		"""
		set composition dimensions used for clustering. Defaults to all valid elemental symbols in data columns
		
		Parameters
		----------
		comp_dims: list of columns in data to use as composition dimensions
		"""
		#if no comp dims specified, use all columns that are valid element symbols
		if comp_dims==None:
			comp_dims = []
			for col in self.data.columns:
				try: 
					mg.Element(col)
					comp_dims.append(col)
				except ValueError:
					pass
		self._comp_dims = comp_dims
		
	def get_comp_dims(self):
		"get composition dimensions"
		return self._comp_dims
	
	comp_dims = property(get_comp_dims,set_comp_dims)
	
	@property 
	def comp_data(self):
		"composition data"
		return self.data[self.comp_dims]
	
	@property
	def iso_dims(self):
		"dimensions used by IsolationForest"
		return self.comp_dims + [self.prop_dim]
	
	@property
	def iso_data(self):
		"data used by IsolationForest"
		return self.data[self.iso_dims]
	
	def scale_composition(self,max_var=1):
		"""
		scale composition dimensions such that largest-variance dimension has variance max_var
		"""
		ss = StandardScaler()
		#get dimension with largest variance
		ref_dim = np.var(self.comp_data).idxmax()
		ss.fit(self.comp_data[ref_dim].values[:,None])
		#scale all comp dims with same scaler such that refdim has variance max_var
		self.scaled_comp_data = pd.DataFrame((max_var**0.5)*ss.transform(self.comp_data),columns=self.comp_dims)
		
	def fit(self, comp_scale=1):
		"""
		fit DBSCAN and IsolationForest to data
		
		Parameters
		----------
		comp_scale: maximum compositional variance set by scale_composition
		"""
		self.scale_composition(max_var=comp_scale)
		self.db.fit(self.scaled_comp_data) #fit DBSCAN to comp data for compositional clustering
		self.clf.fit(self.iso_data) #fit IsolationForest to iso data for greedy outlier flagging
				 
	def predict(self,z_thresh=2):
		"""
		predict outliers in data
		
		Parameters
		----------
		z_thresh: z-score threshold for intra-cluster outlier identification
		"""
		self.z_thresh = z_thresh
		self.pred = pd.DataFrame()
		self.pred[self.prop_dim] = self.data[self.prop_dim]
		self.pred.loc[:,'cluster'] = self.db.fit_predict(self.scaled_comp_data) #db has no pure predict function
		
		self.pred.loc[:,'isolation_flag'] = self.clf.predict(self.iso_data)
		self.pred.loc[:,'isolation_score'] = self.clf.decision_function(self.iso_data)

		#get z-scores for each cluster and cross-ref with isolation forest
		for i, cluster in enumerate(self.pred['cluster'].unique()):
			df = self.pred.loc[self.pred['cluster']==cluster,:]
			self.pred.loc[self.pred['cluster']==cluster,'cluster_zscore'] = z_score(df[self.prop_dim])

		#set final outlier flag - if flagged by isolation forest and cluster z-score is outside z_thresh
		self.pred.loc[:,'outlier_flag'] = np.where(
											(self.pred['isolation_flag']==-1) & (np.abs(self.pred['cluster_zscore']) > z_thresh),
											-1, 0)
		
		return self.pred
		
	def fit_predict(self,comp_scale=1,z_thresh=2):
		"""combine fit and predict functions"""
		self.fit(comp_scale=comp_scale)
		self.predict(z_thresh=z_thresh)
		return self.pred
	
	def remove_outliers(self):
		"""remove outliers identified by fit_predict"""
		self.clean_data = self.data[self.pred['outlier_flag']!=-1]
		return self.clean_data
	
	@property
	def data_pred(self):
		"data joined with prediction results"
		return self.data.join(self.pred.drop(labels=self.prop_dim, axis=1))
	
	@property
	def outliers(self):
		"outlier data rows"
		return self.data_pred[self.data_pred['outlier_flag']==-1]
	
	@property
	def inliers(self):
		"inlier data rows"
		return self.data_pred[self.data_pred['outlier_flag']!=-1]
	
	def set_DB_params(self,**params):
		"""set DBSCAN parameters"""
		self.db.set_params(**params)
	
	def set_IF_params(self,**params):
		"""set IsolationForest parameters"""
		self.clf.set_params(**params)
		
	def highlight_outliers(self, slice_axis, slice_starts, slice_widths, tern_axes,cmap=plt.cm.viridis,**scatter_kwargs):
		"""
		plot all data points with outliers highlighted in red. color determined by value of prop_dim
		
		Parameters
		----------
		
		slice_axis: composition dimension on which to slice
		slice_starts: values of slice_axis at which to start slices
		slice_widths: widths of slices in slice_axis dimension. Single value or list
		tern_axes: composition dimensions for ternary plot axes (order: right, top, left)
		cmap: colormap for prop_dim values
		scatter_kwargs: kwargs to pass to helpers.plotting.scatter_slices
		"""
		#get vmin and vmax
		vmin = self.data[self.prop_dim].min()
		vmax = self.data[self.prop_dim].max()
		#plot inliers
		axes = scatter_slices(self.inliers,self.prop_dim,slice_axis,slice_starts,slice_widths,tern_axes,cmap=cmap,
							vmin=vmin,vmax=vmax,**scatter_kwargs)
		#plot outliers
		scatter_slices(self.outliers,self.prop_dim,slice_axis,slice_starts,slice_widths,tern_axes,cmap=cmap,axes=axes,
					   vmin=vmin,vmax=vmax, colorbar=False, ptsize=20,scatter_kw=dict(marker='d',edgecolors='r',linewidths=0.8),**scatter_kwargs)
		
	def plot_clusters(self, slice_axis, slice_starts, slice_widths, tern_axes, cmap=plt.cm.jet,**scatter_kwargs):
		"""
		plot all data points with cluster shown by color
		
		Parameters
		----------
		slice_axis: composition dimension on which to slice
		slice_starts: values of slice_axis at which to start slices
		slice_widths: widths of slices in slice_axis dimension. Single value or list
		tern_axes: composition dimensions for ternary plot axes (order: right, top, left)
		cmap: colormap for cluster values
		scatter_kwargs: kwargs to pass to helpers.plotting.scatter_slices
		"""
		scatter_slices(self.data_pred,'cluster',slice_axis,slice_starts,slice_widths,tern_axes,cmap=cmap,**scatter_kwargs)
		
	def plot_outliers(self, slice_axis, slice_starts, slice_widths, tern_axes, cmap=plt.cm.viridis,**scatter_kwargs):
		"""
		plot outliers only
		
		Parameters
		----------
		slice_axis: composition dimension on which to slice
		slice_starts: values of slice_axis at which to start slices
		slice_widths: widths of slices in slice_axis dimension. Single value or list
		tern_axes: composition dimensions for ternary plot axes (order: right, top, left)
		cmap: colormap for prop_dim values
		scatter_kwargs: kwargs to pass to helpers.plotting.scatter_slices
		"""
		axes = scatter_slices(self.outliers,self.prop_dim,slice_axis,slice_starts,slice_widths,tern_axes,cmap=cmap,**scatter_kwargs)
		return axes
		
	def plot_inliers(self, slice_axis, slice_starts, slice_widths, tern_axes, cmap=plt.cm.viridis,**scatter_kwargs):
		"""
		plot inliers only
		
		Parameters
		----------
		slice_axis: composition dimension on which to slice
		slice_starts: values of slice_axis at which to start slices
		slice_widths: widths of slices in slice_axis dimension. Single value or list
		tern_axes: composition dimensions for ternary plot axes (order: right, top, left)
		cmap: colormap for prop_dim values
		scatter_kwargs: kwargs to pass to helpers.plotting.scatter_slices
		"""
		axes = scatter_slices(self.inliers,self.prop_dim,slice_axis,slice_starts,slice_widths,tern_axes,cmap=cmap,**scatter_kwargs)
		return axes
	
	def cluster_hist(self,ncols=2):
		
		clusters = self.pred['cluster'].unique()
		print(clusters)
		nrows = int(np.ceil(len(clusters)/ncols))
		print(nrows)
		fig, axes = plt.subplots(nrows,ncols,figsize=(ncols*6,nrows*4))
		for i, cluster in enumerate(clusters):
			df = self.data_pred.loc[self.data_pred['cluster']==cluster,:]

			num_outliers = len(df[df['isolation_flag']==-1])
			try: #2d axes
				ax = axes[int(i/ncols), i%ncols]
			except IndexError: #1d axes
				ax = axes[i]
				
			dfo = df[df['isolation_flag']==-1]
			dfi = df[df['isolation_flag']==1]
			hist, bins = np.histogram(df['cluster_zscore'])
			ax.hist(dfi['cluster_zscore'],alpha=0.5,bins=bins,label='IsolationForest inliers')
			ax.hist(dfo['cluster_zscore'],alpha=0.5,bins=bins, label='IsolationForest outliers')
			ax.set_title(f'Cluster {cluster}')
			ax.set_xlabel('Cluster Z-score')
			ax.set_ylabel('Frequency')
			ax.legend()
			
			#plot z-score threshold
			ax.axvline(-self.z_thresh,ls='--',c='r')
			ax.axvline(self.z_thresh,ls='--',c='r')
		fig.tight_layout()
		
	def quat_plot(self,ax=None,quat_axes=['Co','Fe','Zr','Y'], gridlines=True, cb_label=None, s=3,**scatter_kw):
		qax = QuaternaryAxes(ax=ax)
		qax.draw_axes()
		qax.label_corners(quat_axes,offset=0.11,size=14)
		vmin = self.data[self.prop_dim].min()
		vmax = self.data[self.prop_dim].max()
		
		if cb_label is None:
			cb_label = self.prop_dim
			
		qax.scatter(self.data[quat_axes].values,c=self.data[self.prop_dim], s=s,vmin=vmin, vmax=vmax, colorbar=True,
					cb_kwargs={'label':cb_label,'cbrect':[0.8,0.1,0.02,0.65],'labelkwargs':{'size':14},'tickparams':{'labelsize':13}}, **scatter_kw)
		
		qax.axes_ticks(size=13,corners='rbt',offset=0.08)
		if gridlines==True:
			qax.gridlines(ls=':',LW=0.6)
		qax.ax.axis('off')
	
		return qax
	
	def quat_highlight(self,ax=None,quat_axes=['Co','Fe','Zr','Y'], gridlines=True, **scatter_kw):
		qax = QuaternaryAxes(ax=ax)
		qax.draw_axes()
		qax.label_corners(quat_axes)
		vmin = min(self.inliers[self.prop_dim].min(),self.outliers[self.prop_dim].min())
		vmax = min(self.inliers[self.prop_dim].max(),self.outliers[self.prop_dim].max())
		
		qax.scatter(self.inliers[quat_axes].values,c=self.inliers[self.prop_dim],s=3, vmin=vmin, vmax=vmax, 
					colorbar=True, cb_kwargs={'label':self.prop_dim}, **scatter_kw)
		qax.scatter(self.outliers[quat_axes].values,c=self.outliers[self.prop_dim],s=6, vmin=vmin, vmax=vmax,
					edgecolors='r',linewidths=0.5, **scatter_kw)
		
		qax.axes_ticks()
		if gridlines==True:
			qax.gridlines()
		qax.ax.axis('off')
	
		return qax
	
	def quat_clusters(self,ax=None,quat_axes=['Co','Fe','Zr','Y'], gridlines=True, cmap=plt.cm.plasma,s=3, 
					  label_kw={},**scatter_kw):
		qax = QuaternaryAxes(ax=ax)
		qax.draw_axes()
		qax.label_corners(quat_axes,**label_kw)
		vmin = self.pred['cluster'].min()
		vmax = self.pred['cluster'].max()
		
		#make norm for discrete colormap
		clusters = self.pred['cluster'].unique()
		n_clusters = len(self.pred['cluster'].unique())
		bounds = np.arange(clusters.min()-0.5,clusters.max()+0.51)
		norm = mpl.colors.BoundaryNorm(bounds, cmap.N)
		
		qax.scatter(self.data[quat_axes].values,c=self.pred['cluster'],s=s, cmap=cmap, norm=norm, vmin=vmin, vmax=vmax, 
					colorbar=True, cb_kwargs={'label':'Cluster','norm':norm,'ticks':clusters}, **scatter_kw)
		
		qax.axes_ticks()
		if gridlines==True:
			qax.gridlines()
		qax.ax.axis('off')
		return qax
	
	def reduce_comp_dims(self,kernel='poly',gamma=10,**kpca_kw):
		comp_dims = self.comp_dims.copy()
		comp_dims.remove('O')
		comp_dims.remove('Ba')
		print(comp_dims)
		reconstructed = self.data[comp_dims].copy()
		kpca = KernelPCA(kernel=kernel,n_components=2,fit_inverse_transform=True,gamma=gamma,**kpca_kw)

		self.reduced = pd.DataFrame(kpca.fit_transform(self.data[comp_dims]),columns=['v1','v2'])
		reconstructed[comp_dims] = kpca.inverse_transform(self.reduced)

		self.reduced[self.prop_dim] = self.data[self.prop_dim].values
		self.reduced['outlier_flag'] = self.pred['outlier_flag'].values
		self.reduced['cluster'] = self.pred['cluster'].values
		
		error = np.linalg.norm(self.data[comp_dims] - reconstructed,ord=2)
		
		return self.reduced, error
	
	def reduced_plot(self,ax=None,cmap=plt.cm.viridis,vmin=None,vmax=None,cbar=True, cbrect=[0.88,0.12,0.02,0.75],**kwargs):
		"""
		scatter plot of prop_dim in reduced-dimension composition space
		
		Args:
		-----
		kwargs: kwargs to pass to plt.scatter
		"""
		if ax is None:
			fig, ax = plt.subplots()
		else: 
			fig = plt.gcf()
		if vmin is None:
			vmin = self.reduced[self.prop_dim].min()
		if vmax is None:
			vmax = self.reduced[self.prop_dim].max()
		
		ax.scatter(self.reduced['v1'],self.reduced['v2'],c=self.reduced[self.prop_dim],cmap=cmap,vmin=vmin,vmax=vmax,**kwargs)
		if cbar==True:
			add_colorbar(fig=fig,ax=ax,cmap=cmap,label=self.prop_dim, vmin=vmin,vmax=vmax,
				subplots_adjust=dict(left=0.1,right=0.8),cbrect=cbrect)
		ax.set_xlabel('$v_1$')
		ax.set_ylabel('$v_2$')
	
	def reduced_highlight_plot(self,ax=None,cmap=plt.cm.viridis,vmin=None,vmax=None,s=8,cbar=True, cbrect=[0.88,0.12,0.02,0.75],**kwargs):
		"""
		scatter plot of prop_dim in reduced-dimension composition space with outliers highlighted in red
		
		Args:
		-----
		ax: axis on which to plot. if None, create new axis
		cmap: colormap
		vmin: vmin for colormap
		vmax: vmax for colormap
		s: marker size
		cbar: if True, create a colorbar
		cbrect: colorbar rectangle: [left, bottom, width, height]
		kwargs: kwargs to pass to plt.scatter
		"""
		outliers = self.reduced.loc[self.reduced['outlier_flag']==-1,:]
		inliers = self.reduced.loc[self.reduced['outlier_flag']!=-1,:]
		
		if ax is None:
			fig, ax = plt.subplots()
		else: 
			fig = plt.gcf()
		if vmin is None:
			vmin = self.reduced[self.prop_dim].min()
		if vmax is None:
			vmax = self.reduced[self.prop_dim].max()
			
		ax.scatter(inliers['v1'],inliers['v2'],c=inliers[self.prop_dim],cmap=cmap,vmin=vmin,vmax=vmax,s=s,**kwargs)
		ax.scatter(outliers['v1'],outliers['v2'],c=outliers[self.prop_dim],cmap=cmap,vmin=vmin,vmax=vmax,s=s*2,
				   edgecolors='r', linewidths=0.7, **kwargs)
		if cbar==True:
			add_colorbar(fig=fig,ax=ax,cmap=cmap,label=self.prop_dim, vmin=vmin,vmax=vmax,
				subplots_adjust=dict(left=0.1,right=0.8),cbrect=cbrect)
		ax.set_xlabel('$v_1$')
		ax.set_ylabel('$v_2$')
		
		return ax
		
	def reduced_inlier_plot(self,ax=None,cmap=plt.cm.viridis,vmin=None,vmax=None,cbar=True,cbrect=[0.88,0.12,0.02,0.75],**kwargs):
		inliers = self.reduced.loc[self.reduced['outlier_flag']!=-1,:]
		
		if ax is None:
			fig, ax = plt.subplots()
		else: 
			fig = plt.gcf()
		if vmin is None:
			vmin = self.reduced[self.prop_dim].min()
		if vmax is None:
			vmax = self.reduced[self.prop_dim].max()
			
		ax.scatter(inliers['v1'],inliers['v2'],c=inliers[self.prop_dim],cmap=cmap,vmin=vmin,vmax=vmax,**kwargs)
		if cbar==True:
			add_colorbar(fig=fig,ax=ax,cmap=cmap,label=self.prop_dim, vmin=vmin,vmax=vmax,
				subplots_adjust=dict(left=0.1,right=0.8),cbrect=cbrect)
		ax.set_xlabel('$v_1$')
		ax.set_ylabel('$v_2$')
		
		return ax
		
	def reduced_outlier_plot(self,ax=None,cmap=plt.cm.viridis,vmin=None,vmax=None,cbar=True,cbrect=[0.88,0.12,0.02,0.75],**kwargs):
		"""
		scatter plot of prop_dim in reduced-dimension composition space with outliers highlighted in red
		
		Args:
		-----
		kwargs: kwargs to pass to plt.scatter
		"""
		outliers = self.reduced.loc[self.reduced['outlier_flag']==-1,:]
		
		if ax is None:
			fig, ax = plt.subplots()
		else: 
			fig = plt.gcf()
		if vmin is None:
			vmin = self.reduced[self.prop_dim].min()
		if vmax is None:
			vmax = self.reduced[self.prop_dim].max()
			
		ax.scatter(outliers['v1'],outliers['v2'],c=outliers[self.prop_dim],cmap=cmap,vmin=vmin,vmax=vmax,**kwargs)
		if cbar==True:
			add_colorbar(fig=fig,ax=ax,cmap=cmap,label=self.prop_dim, vmin=vmin,vmax=vmax,
				subplots_adjust=dict(left=0.1,right=0.8),cbrect=cbrect)
		ax.set_xlabel('$v_1$')
		ax.set_ylabel('$v_2$')
		
		return ax
	
	def reduced_cluster_plot(self, ax=None,cmap=plt.cm.plasma, cbar=True,cbrect = [0.88,0.12,0.02,0.75], **kwargs):
				
		vmin = self.pred['cluster'].min()
		vmax = self.pred['cluster'].max()
		
		#make norm for discrete colormap
		clusters = self.pred['cluster'].unique()
		n_clusters = len(self.pred['cluster'].unique())
		bounds = np.arange(clusters.min()-0.5,clusters.max()+0.51)
		norm = mpl.colors.BoundaryNorm(bounds, cmap.N)
		
		if ax is None:
			fig, ax = plt.subplots()
		else: 
			fig = plt.gcf()
		ax.scatter(self.reduced['v1'],self.reduced['v2'],c=self.reduced['cluster'],cmap=cmap,norm=norm, **kwargs)
		ax.set_xlabel('$v_1$')
		ax.set_ylabel('$v_2$')
		if cbar==True:
			add_colorbar(fig=fig,ax=ax,cmap=cmap,norm=norm,label='Cluster',ticks=clusters,
					 subplots_adjust=dict(left=0.1,right=0.8),cbrect=cbrect)
		
		return ax
		