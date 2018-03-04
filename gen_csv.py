
# coding: utf-8

# In[140]:


import pandas as pd
import numpy as np
import os
import glob
from datetime import datetime
from shutil import copyfile

datadir = os.path.join(os.environ['USERPROFILE'],'OneDrive - Colorado School of Mines/Research/MIDDMI/TCO/Data')
indir = os.path.join(datadir,'in') #libraries to be processed
outdir = os.path.join(datadir,'out') #output directory for csvs and logs
procdir = os.path.join(datadir,'processed') #destination dir for processed libraries
errdir = os.path.join(datadir,'error') #destination dir for libraries with errors
config = os.path.join(datadir,'config') #config/parameter file location

############################
"read config files"
############################
os.chdir(config)

#load variable definitions
#----------------------------
vardictdf = pd.read_excel('VariableDict.xlsx')

#put standard variables into dict
stdvardf = vardictdf[vardictdf.Keyword.str[0] != '!']
stdvars = list(stdvardf.VarName)
headers = list(stdvardf.Keyword + ': ' + stdvardf.DisplayName)
stdvardict = dict(zip(stdvars,headers))

#composition variables
compvardf = vardictdf[vardictdf.Keyword == '!composition']
compvars = list(compvardf.VarName)

#load PLD parameters
#----------------------------
prepstep = 'Pulsed Laser Deposition'
params = pd.read_excel('PLD_params.xlsx',skiprows=3,index_col=0)
paramdictdf = pd.read_excel('PLD_ParamDict.xlsx')
#put standard params into dict
stdparamdf = paramdictdf[paramdictdf.Keyword.str[0] != '!']
stdparams = list(stdparamdf.ParamName)
parheaders = list(stdparamdf.Keyword + ': ' + stdparamdf.DisplayName)
stdparamdict = dict(zip(stdparams,parheaders))
params = params.rename(index=str, columns = stdparamdict) #rename headers to ingestable format using paramdict
params.insert(1,'PREPARATION STEP NAME',prepstep)
#end config

#library prefix to strip out
libprefix = 'PDAC_COM3_'

########################################
"loop through all libraries in indir"
########################################
os.chdir(indir)
nsucc = 0
nerr = 0
errsumname = os.path.join(errdir, 'gen_csv_' + datetime.now().strftime('%Y%m%d_%H%M%S') + '.err')
errsum = open(errsumname,'w')

for lib in next(os.walk(indir))[1]:
    os.chdir(lib)
    
    #log file
    logname = lib + '_gen_csv.log'
    log = open(logname, 'w')
    log.write('Processing library ' + lib + '\n')
    
    #error file
    errtxt = ''
    
    #csv
    csvname = lib + '_AllVar.csv'
    
    all_var = pd.DataFrame([])
    
    #read the Points file to determine file length
    pts = pd.read_csv(lib + '_Points.txt',sep='\t',usecols = [0]) 
    
    #dataframes to store composition data
    chm = pd.DataFrame()#np.zeros((len(pts),1),dtype='str'),columns=['formula'])
    sitesum = pd.DataFrame(np.zeros((len(pts),2)), columns = ['A','B'])
    
    #track found variables
    foundvars = ['Point','Row','Column']
    
    for fname in glob.glob(lib + '*.txt'):
        vname = fname[len(lib)+1:fname.find('.txt')]
        #regular variables
        if vname in stdvars:
            if len(all_var) == 0:
                cols = None
            else:
                cols = [3]
            df = pd.read_csv(fname,sep='\t',usecols = cols)
            all_var = pd.concat([all_var,df], axis=1)
        #composition variables
        elif vname in compvars:
            dfc = pd.read_csv(fname,sep='\t',usecols = [3])
            elmnt = vname[:vname.find('_at')]
            chm[elmnt] = dfc[vname] 
            sitesum.A = sitesum.A + dfc[vname]
        
        foundvars.append(vname)
    
    #identify ignored and missing variables
    ignoredvars = np.setdiff1d(foundvars,stdvars+compvars, assume_unique=True)
    missingstd = np.setdiff1d(stdvars,foundvars, assume_unique=True)
    missingcomp = np.setdiff1d(compvars,foundvars, assume_unique=True)
    if len(ignoredvars) > 0:
        log.write('Ignored variables:\n\t' + '\n\t'.join(ignoredvars) + '\n')
    if len(missingstd) > 0:
        log.write('*Warning: missing standard variables:\n\t' + '\n\t'.join(missingstd) + '\n')
    if len(missingcomp) > 0:
        log.write('*Warning: missing composition variables:\n\t' + '\n\t'.join(missingcomp) + '\n')
        errtxt += 'Missing composition variables: ' + ', '.join(missingcomp) + '\n'
        
    #determine composition
    sitesum.B = 1 - sitesum.A
    sitemax = sitesum.max(axis=1)
    sitenorm = sitesum.divide(sitemax,axis=0) #normalize for the higher occupancy site
    chmnorm = chm.divide(sitemax,axis=0)
    chmnorm.insert(0,'Ba',list(sitenorm.B))
    chmnorm = chmnorm.round(5)
    for elmnt in list(chmnorm.columns):
        chmnorm[elmnt] = elmnt + chmnorm[elmnt].map(str) 
    chmnorm['formula'] = chmnorm.apply(lambda x: ''.join(x),axis=1)
    chmnorm['formula'] = chmnorm['formula'] + 'O3'

    sample = lib[len(libprefix):]

    all_var = all_var.rename(index=str, columns = stdvardict) #rename headers to ingestable format using vardict
    all_var.insert(0,'IDENTIFIER: Sample number',sample) #add sample column        
    all_var.insert(0,'FORMULA',list(chmnorm['formula'])) #add formula column
    
    #add PLD parameters corresponding to sample number
    if sample in list(params.index) and not(pd.isnull(params.loc[sample,'IDENTIFIER: Date'])):
        #identify targets used and reformat accordingly
        sp = params.loc[sample,:]
        for tgti in tgtinf:
            tgt = tgti[0:8]
            disp = np.asscalar(paramdictdf[paramdictdf['ParamName']==tgti].DisplayName)
            if sp[tgti] == ' - ' or sp[tgti] == '':
                sp = sp.drop(tgti)
            else:
                sp = sp.rename({tgti: 'PREPARATION STEP DETAIL: ' + sp[tgt] + ' ' + disp})
        for tgt in tgtcols:
            sp = sp.drop(tgt)

        all_var = all_var.join(sp.to_frame().T,on='IDENTIFIER: Sample number')
    else:
        log.write('*Warning: could not locate PLD parameters\n')
        errtxt += 'Missing PLD parameters\n'
        
    all_var.to_csv(csvname,index=False)
    log.write('Wrote ' + csvname + '\n')
    
    #handle errors - these files should be sent to errdir rather than procdir, and err file should be written
    if len(errtxt) > 0:
        #write err file
        errname = lib + '_gen_csv.err' #os.path.join(errdir, lib + '_gen_csv.err')
        err = open(errname,'w')
        err.write(errtxt)
        err.close()
        
        #write to errsummary
        errsum.write(lib + ': ' + errtxt)
        
        #make a note in log
        log.write('Error(s) detected, moving to ' + errdir)
        log.close()
        
        #move to errdir
        os.chdir('..')
        os.rename(os.path.join(indir,lib), os.path.join(errdir,lib))
        
        #increment nerr
        nerr += 1
    #if no errors - move to procdir and copy outputs to outdir
    else:
        #move processed library to procdir
        log.write('Finished processing library ' + lib)
        log.close()

        #copy csv and log to outdir
        copyfile(csvname, os.path.join(outdir,csvname))
        copyfile(logname, os.path.join(outdir,logname))
        os.chdir('..')
        os.rename(os.path.join(indir,lib), os.path.join(procdir,lib))
        
        #increment nsucc
        nsucc += 1


errsum.close()

print('Processed ' + str(nsucc) + ' sample(s) successfully')
print('Processed ' + str(nerr) + ' sample(s) with errors')

