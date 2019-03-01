# Convenience functions for ternary plotting

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import ternary
from . import fileload as fl
from .calc_chemfeat import perovskite, formula_redfeat
from .predict import formula_input
from ternary.helpers import simplex_iterator

def quat_slice_scatter(data, z, slice_start, slice_width=0, slice_axis='Y', tern_axes=['Co','Fe','Zr'], tern_labels=None, labelsize=14,
					tick_kwargs={'axis':'lbr', 'linewidth':1, 'tick_formats':'%.1f','offset':0.03}, nticks=5, add_labeloffset=0,
					cmap=plt.cm.viridis, ax=None,vmin=None,vmax=None,ptsize=8, figsize=None, scatter_kw={}):
	if slice_width==0:
		df = data[data[slice_axis] == slice_start]
	else:
		df = data[(data[slice_axis] >= slice_start) & (data[slice_axis] < slice_start +	 slice_width)]

	points = df.loc[:,tern_axes].values
	colors = df.loc[:,z].values
	if len(df[pd.isnull(df[z])]) > 0:
		print('Warning: null values in z column')
	if vmin==None:
		vmin = np.min(colors)
	if vmax==None:
		vmax = np.max(colors)

	scale = 1 - (slice_start + slice_width/2) #point coords must sum to scale
	print('Scale: {}'.format(scale))
	#since each comp has different slice_axis value, need to scale points to plot scale
	ptsum = np.sum(points,axis=1)[np.newaxis].T
	scaled_pts = points*scale/ptsum

	if ax==None:
		fig, ax = plt.subplots(figsize=figsize)
		tfig, tax = ternary.figure(scale=scale,ax=ax)
	else:
		tax = ternary.TernaryAxesSubplot(scale=scale,ax=ax)
	
	if len(points) > 0:
		tax.scatter(scaled_pts,s=ptsize,cmap=cmap, vmin=vmin,vmax=vmax,
				colorbar=False,c=colors,**scatter_kw)
				
	tax.boundary(linewidth=1.0)
	tax.clear_matplotlib_ticks()
	
	multiple = scale/nticks
	#manually set ticks and locations - default tax.ticks behavior does not work
	ticks = list(np.arange(0,scale + 1e-6,multiple))
	locations = ticks
	
	tax.ticks(multiple=multiple,ticks=ticks,locations=locations,**tick_kwargs)
	tax.gridlines(multiple=multiple,linewidth=0.8)
	
	if tern_labels is None:
		tern_labels = tern_axes
	tax.right_corner_label(tern_labels[0],fontsize=labelsize,offset=0.08+add_labeloffset)
	tax.top_corner_label(tern_labels[1],fontsize=labelsize,offset=0.2+add_labeloffset)
	tax.left_corner_label(tern_labels[2],fontsize=labelsize,offset=0.08+add_labeloffset)

	tax._redraw_labels()
	ax.axis('off')
	  
	return tax, vmin, vmax
	
def scatter_slices(data, z, slice_axis, slice_starts, slice_widths, tern_axes, axes=None, ncols=2, figsize=None, colorbar=True,
				cmap = plt.cm.viridis, cb_kwargs = {}, vmin=None, vmax=None,
				 titles=True, titlesize=14, titlebox_props=dict(boxstyle='round', facecolor='wheat', alpha=0.5),**slice_scatter_kw):
	
	if axes is None:
		nrows = int(np.ceil(len(slice_starts)/ncols))
		if figsize==None:
			figsize = [ncols*6.5,nrows*4.8]
		fig, axes = plt.subplots(nrows,ncols,figsize=figsize)
	else:
		nrows = axes.shape[0]
	
	#if single value given for slice_widths, make list of len matching slice_starts
	if type(slice_widths)!=list:
		slice_widths = [slice_widths]*len(slice_starts)
	
	#get vmin and vmax
	if vmin is None or vmax is None:
		vmins = []
		vmaxs = []
		
		for slice_start,slice_width in zip(slice_starts,slice_widths):
			if slice_width==0:
				df = data[data[slice_axis] == slice_start]
			else:
				df = data[(data[slice_axis] >= slice_start) & (data[slice_axis] < slice_start +	 slice_width)]

			vmins.append(df.loc[:,z].min())
			vmaxs.append(df.loc[:,z].max())
		
		if vmin is None:
			vmin = min(vmins)
		if vmax is None:
			vmax = max(vmaxs)
	
	for i, (slice_start,slice_width) in enumerate(zip(slice_starts,slice_widths)):
		try: #2d axes
			ax = axes[int(i/ncols), i%ncols]
		except IndexError: #1d axes
			ax = axes[i]
		tax, vmin, vmax = quat_slice_scatter(data, z, slice_start, slice_width=slice_width, slice_axis=slice_axis, tern_axes=tern_axes, ax=ax, 
										cmap=cmap,vmin=vmin, vmax=vmax,**slice_scatter_kw)

		if titles==True:
			ax.set_title('{:.2g} < {} < {:.2g}'.format(slice_start,slice_axis,slice_start+slice_width),
					  size=titlesize,x=0.1,bbox=titlebox_props)
	
	if colorbar==True:
		cb_defaults = dict(label=z,labelkwargs={'size':16,'labelpad':10},
				tickparams={'labelsize':12}, tickformat='%.1f',
				subplots_adjust={'left':0.05,'wspace':0.35, 'hspace':0.25, 'right':0.8})
		cb_defaults.update(cb_kwargs)
		add_colorbar(vmin=vmin,vmax=vmax,ax=axes, cmap=cmap,**cb_defaults)
	
	return axes

def add_colorbar(fig=None, ax=None, cbrect=[0.9,0.15,0.02,0.75], label=None, tickformat=None, 
				 cmap = plt.cm.viridis, vmin=None, vmax=None, logscale=False, norm=None,
				 tickparams={}, labelkwargs={},subplots_adjust={'left':0.05,'wspace':0.35, 'hspace':0.25, 'right':0.8},
				 **cb_kwargs):
	#add a single colorbar
	if fig is None:
		fig = plt.gcf()
	if ax is None:
		ax = plt.gca()
	#make an axis for colorbar to control position/size
	cbaxes = fig.add_axes(cbrect) #[left, bottom, width, height]
	#code from ternary.colormapping.colorbar_hack
	
	if norm==None:
		if logscale==True:
			norm = colors.LogNorm(vmin=vmin,vmax=vmax)
		else:
			norm = plt.Normalize(vmin=vmin,vmax=vmax)
	sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
	sm._A = []
	cb = fig.colorbar(sm, ax=ax, cax=cbaxes, format=tickformat,**cb_kwargs)
	cb.ax.tick_params(**tickparams)
	if label is not None:
		cb.set_label(label, **labelkwargs)
	
	fig.subplots_adjust(**subplots_adjust)
	
def sliceformula_from_tuple(tup,slice_val,slice_axis='Y',tern_axes=['Co','Fe','Zr'],Ba=1,ndigits=5):
	tern_tot = 1 - slice_val
	tuple_scale = np.sum(tup)
	formula = 'Ba{}'.format(Ba)
	el_amts = []
	for el,amt in zip(tern_axes,tup):
		el_amt = amt*tern_tot/tuple_scale
		el_amts.append(el + str(round(el_amt,ndigits)))
	#put in alphabetical order - important for pickling and looking up previously calculated features
	el_amts.sort()
	formula = formula + ''.join(el_amts)
	
	formula = formula + slice_axis + str(slice_val)
	formula = formula + 'O3'
	return formula

def rescale_ticks(tax,new_scale,multiple,axis='lbr',**kwargs): #for redrawing ticks based on desired scale - doesn't scale actual data
	scale = tax._scale
	ticks = np.arange(0,new_scale + 1e-6,multiple)
	locations = np.arange(0,scale + 1e-6,multiple*scale/new_scale)
	tax.ticks(ticks=list(ticks), locations=list(locations),axis=axis,**kwargs)
	
def quat_slice_heatmap(tuple_scale, zfunc, slice_val, zfunc_kwargs={}, style='triangular', tern_labels=['Co','Fe','Zr'], 
					   labelsize=14, add_labeloffset=0, cmap=plt.cm.viridis, ax=None,figsize=None, vmin=None, vmax=None,
					   multiple=0.1, tick_kwargs={'tick_formats':'%.1f','offset':0.02}):

	axis_scale = 1 - slice_val
	tuples = []
	zvals = []
	for tup in simplex_iterator(scale=tuple_scale):
		tuples.append(tup)
		zvals.append(zfunc(tup,**zfunc_kwargs))
	if vmin==None:
		vmin = min(zvals)
	if vmax==None:
		vmax = max(zvals)
	
	d = dict(zip([t[0:2] for t in tuples],zvals))

	if ax==None:
		fig, ax = plt.subplots(figsize=figsize)
		tfig, tax = ternary.figure(scale=tuple_scale,ax=ax)
	else:
		tax = ternary.TernaryAxesSubplot(scale=tuple_scale,ax=ax)
	
	tax.heatmap(d,style=style,colorbar=False,cmap=cmap,vmin=vmin,vmax=vmax)
	rescale_ticks(tax,new_scale=axis_scale,multiple = multiple, **tick_kwargs)
	tax.boundary()
	tax.ax.axis('off')
	if tern_labels==None:
		tern_labels=tern_axes
	
	tax.right_corner_label(tern_labels[0],fontsize=labelsize,offset=0.08+add_labeloffset)
	tax.top_corner_label(tern_labels[1],fontsize=labelsize,offset=0.2+add_labeloffset)
	tax.left_corner_label(tern_labels[2],fontsize=labelsize,offset=0.08+add_labeloffset)
	
	tax._redraw_labels()
	
	return tax, vmin, vmax
	
def quat_slice_heatmap2(tuple_scale, zfunc, slice_val, zfunc_kwargs={}, style='triangular', slice_axis='Y', tern_axes=['Co','Fe','Zr'], 
					   labelsize=14, add_labeloffset=0, cmap=plt.cm.viridis, ax=None,figsize=None, vmin=None, vmax=None, Ba=1,
					   multiple=0.1, tick_kwargs={'tick_formats':'%.1f','offset':0.02}):
	"""
	get zvals from formula instead of tup
	"""
	axis_scale = 1 - slice_val
	tuples = []
	zvals = []
	for tup in simplex_iterator(scale=tuple_scale):
		tuples.append(tup)
		formula = sliceformula_from_tuple(tup,slice_val=slice_val,slice_axis=slice_axis,tern_axes=tern_axes,Ba=Ba)
		zvals.append(zfunc(formula,**zfunc_kwargs))
	if vmin==None:
		vmin = min(zvals)
	if vmax==None:
		vmax = max(zvals)
	
	d = dict(zip([t[0:2] for t in tuples],zvals))

	if ax==None:
		fig, ax = plt.subplots(figsize=figsize)
		tfig, tax = ternary.figure(scale=tuple_scale,ax=ax)
	else:
		tax = ternary.TernaryAxesSubplot(scale=tuple_scale,ax=ax)
	
	tax.heatmap(d,style=style,colorbar=False,cmap=cmap,vmin=vmin,vmax=vmax)
	rescale_ticks(tax,new_scale=axis_scale,multiple = multiple, **tick_kwargs)
	tax.boundary()
	tax.ax.axis('off')
	
	tax.right_corner_label(tern_axes[0],fontsize=labelsize,offset=0.08+add_labeloffset)
	tax.top_corner_label(tern_axes[1],fontsize=labelsize,offset=0.2+add_labeloffset)
	tax.left_corner_label(tern_axes[2],fontsize=labelsize,offset=0.08+add_labeloffset)
	
	tax._redraw_labels()
	
	return tax, vmin, vmax
	
def heatmap_inputs(cat_ox_lims,tuple_scale, pkl_dict, slice_axis, slice_vals, tern_axes,conditions,Ba = 0.9):
	inputs = {}
	red_feat = {}
	for slice_val in slice_vals:
		for tup in simplex_iterator(scale=tuple_scale):
			formula = sliceformula_from_tuple(tup=tup,slice_val=slice_val,slice_axis=slice_axis,tern_axes=tern_axes,Ba=Ba)
			try:
				rf = pkl_dict.dict[formula]
			except KeyError:
				rf = formula_redfeat(formula,cat_ox_lims=cat_ox_lims)
				pkl_dict.dict[formula] = rf
			red_feat[formula] = rf
			inp_dict = formula_input(formula,cat_ox_lims,conditions,red_feat=rf)
			inputs[formula] = inp_dict
			
	return inputs
			
def dict_lookup(key,kdict):
	"""
	dict lookup function to feed to quat_slice_heatmap2
	"""
	return kdict[key]

def log_dict_lookup(key,kdict):
	"""
	log10 dict lookup function to feed to quat_slice_heatmap2
	"""
	return np.log10(dict_lookup(key,kdict))