import numpy as np
import pandas as pd
import warnings

def get_file_info(file,sample):
    infostr = file.replace(sample,'').replace('.txt','').replace('.csv','').strip('_')
    info = infostr.split('_')
    return info

def concat_nameswvalues(df,cols=None,roundvals=False):
    tdf = df.copy()
    if cols==None:
        cols = tdf.columns
    if roundvals!=False:
        tdf = tdf.round(roundvals)
    for col in cols:
        tdf[col] = col + tdf[col].astype(str)
    tdf['concat'] = tdf.apply(lambda x: ''.join(x),axis=1)
    return tdf['concat']

def get_formula(df,overwrite=False):
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
    normcomp['formula'] = concat_nameswvalues(normcomp,roundvals=5)
    normcomp['formula'] = normcomp['formula'] + 'O3'
    
    if overwrite==False:
        return normcomp['formula']
    elif overwrite==True:
        df['formula'] = normcomp['formula']
        df.drop(columns=at_col,inplace=True)
        
def lin_reg(x,y):
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