import numpy as np
import pandas as pd
import os
import warnings
import glob

def get_file_info(file,sample=None):
	'''
	extract info from filename (usually measurement conditions)
	'''
	filename = os.path.basename(file)
	if sample==None:
		sample = filename[0:15]
	#remove sample and file extension
	infostr = filename.replace(sample,'').replace('.txt','').replace('.csv','')
	#get column names and remove from filename
	cols = pd.read_csv(file,sep='\t',nrows=0).columns
	for col in cols:
		infostr = infostr.replace(col,'')
	#remove separating ;s and leading/trailing _
	infostr = infostr.replace(';','').strip('_')
	return infostr.split('_')
	
def read_datafile(file,append_file_info=True,info_cols=['T_set','atm']):
	'''
	load a single data file
	----------------------
	file: file to load
	append_file_info: if True, add columns for info fields extracted from filename
	info_cols: ordered list of info fields contained in filename
	'''
	df = pd.read_csv(file,sep='\t',index_col=0)
	if append_file_info==True:
		info = get_file_info(file)
		if len(info)!=len(info_cols): 
			raise Exception('file_info mismatch with info_cols for {}.\nfile_info: {}\ninfo_cols: {}'.format(
								os.path.basename(filename),info,info_cols))
		
		df.loc[:,'Point'] = df.index
		for col,val in zip(info_cols,info):
			df.insert(len(df.columns),col,val)
			df.index = df.index.astype(str) + '_' + df[col].astype(str)
		df.index.rename('Point_info',inplace=True)
	
	return df

def load_sample_files(sampledir,info_filter=None,append_file_info=True,info_cols=['T_set','atm']):
	'''
	load all files in sample directory using read_data_file
	----------------------
	sampledir: sample directory to load
	info_filter: dict of allowed values for file info (e.g. {'T_set':['400C','500C']} will load only files for 400C and 500C)
	append_file_info: if True, add columns for info fields extracted from filename
	info_cols: ordered list of info fields contained in filename
	'''
	orig = os.getcwd()
	os.chdir(sampledir)
	
	df = pd.DataFrame()
	if info_filter is None:
		for datafile in glob.glob('*.txt'):
			tdf = read_datafile(datafile)
			df = df.append(tdf,sort=True)
	else:
		for datafile in glob.glob('*.txt'):
			info = get_file_info(datafile)
			if len(info)!=len(info_cols): 
				raise Exception('file_info mismatch with info_cols for {}.\nfile_info: {}\ninfo_cols: {}'.format(
									os.path.basename(filename),info,info_cols))
			info_dict = dict(zip(info_cols,info))
			filter_violations = 0
			for k,v in info_filter.items():
				if info_dict[k] not in v:
					filter_violations += 1
			if filter_violations==0:
				tdf = read_datafile(datafile)
				df = df.append(tdf,sort=True)
	
	os.chdir(orig)
	
	return df
	
def concat_nameswvalues(df,cols=None,roundvals=False):
	'''
	convenience function for concatenating dataframe column names and values. Used to create chemical formula
	'''
	tdf = df.copy()
	if cols==None:
		cols = tdf.columns
	if roundvals!=False:
		tdf = tdf.round(roundvals)
	for col in cols:
		tdf[col] = col + tdf[col].astype(str)
	tdf['concat'] = tdf.apply(lambda x: ''.join(x),axis=1)
	return tdf['concat']

def get_formula(df,overwrite=False,round=5):
	'''
	derive chemical formula from X_at columns of dataframe
	----------------
	df: dataframe containing atomic fractions
	overwrite: if True, add formula column to df and remove X_at columns in place. If False, don't modify df, and simply return formulas as Series
	'''
	#list all possible A and B site elements
	Asite = ['Ba']
	Bsite = ['Co','Fe','Y','Zr']

	#get elements actually present
	at_col = df.columns[df.columns.str.contains('_at')]
	el_col = at_col.str.replace('_at','')
	A_col = np.intersect1d(el_col,Asite)
	B_col = np.intersect1d(el_col,Bsite)
	if len(A_col) + len(B_col) < len(el_col):
		raise Exception('Not all elements are captured in A- and B-site lists')

	comp = df.loc[:,at_col]
	coldict = dict(zip(at_col,el_col))
	comp.rename(columns=coldict,inplace=True) #remove _at from column names
	
	#normalize for higher-occupancy site - assume other site is deficient
	sites = pd.DataFrame()
	sites['A_sum'] = comp[A_col].sum(axis=1)
	sites['B_sum'] = comp[B_col].sum(axis=1)
	sites['sitemax'] = sites.max(axis=1)
	sites['sitesum'] = sites['A_sum'] + sites['B_sum']
	badsum = sites[sites['sitesum'].round(5)!=1].index
	if len(badsum)>0:
		raise Exception('Atomic fractions do not sum to 1 for following points: {}'.format(badsum.values))
	normcomp = comp.divide(sites['sitemax'],axis=0)

	#concatenate elements & fractions to get formula string
	normcomp['formula'] = concat_nameswvalues(normcomp,roundvals=round)
	normcomp['formula'] = normcomp['formula'] + 'O3'
	
	if overwrite==False:
		return normcomp['formula']
	elif overwrite==True:
		df['formula'] = normcomp['formula']
		df.drop(columns=at_col,inplace=True)
		
def lin_reg(x,y):
	'''
	convenience function for linear regression. returns slope, intercept, and R^2
	'''
	#y = mx + b
	model, resid = np.polyfit(x,y,deg=1,full=True)[:2]
	if len(x) > 2:
		R2 = float(1-resid/(y.size*y.var())) #calculate R^2
	elif len(x)==2:
		R2 = 1.5
	m = model[0]
	b = model[1]
	return m, b, R2
		
def fit_by_group(df, x, y, groupby):
	'''
	apply linear fit to dataframe by groups. returns new dataframe with fit parameters
	-------------------
	df: dataframe to fit
	x: x data - must specify as column of df, e.g. df['T'] or 1/df['T']
	y: y data - must specify as column of df
	groupby: list of columns to group by
	'''
	cdf = df.copy()
	fit = pd.DataFrame()
	cdf['x'] = x
	cdf['y'] = y
	cdf = cdf.loc[(pd.isnull(cdf['x'])==False) & (pd.isnull(cdf['y'])==False),:]
	grouped = cdf.groupby(groupby)
	for name, group in grouped:
		gdf = grouped.get_group(name)
		if len(gdf) > 1:
			x = np.array(gdf['x'])
			y = np.array(gdf['y']) 
			m, b, R2 = lin_reg(x,y)
			#include columns grouped by
			row = pd.Series(name, index=groupby)
			row = row.append(pd.Series([m,b,R2], index=['m','b','R2']))
			fit = fit.append(row, ignore_index=True)
			#print('Group {} fitted with {} data points'.format(name,len(gdf)))
			if len(x)==2:
				warnings.warn('Group {} fitted with only 2 data points. R^2 set to 1.5 to flag'.format(name))
		else:
			print('Group {} has too few points to fit'.format(name))
		
	return fit