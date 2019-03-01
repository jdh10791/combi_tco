# Functions for predicting using Citrination API

import pypif.pif
from citrination_client import CitrinationClient
import pandas as pd
import numpy as np
import os
from .calc_chemfeat import formula_redfeat

def predict_from_pifs(view_id,pifs,predict,condition={},exclude=[],client=None):
	'''
	predict properties using inputs from pifs. returns dataframe with actual and predicted values
	------------------
	view_id: dataview id containing model
	pifs: list of pifs to predict
	predict: list of properties to predict
	condition: dict of conditions and values
	exclude: properties in pif to exclude from inputs (besides properties to predict)
	client: CitrinationClient instance
	'''
	if client is None:
		client = CitrinationClient(os.environ['CITRINATION_API_KEY'],'https://citrination.com')
	
	ids = []
	inputs = []
	predict = predict
	actuals = []
	for pif in pifs:
		pids = {p.name:p.value for p in pif.ids}
		ids.append(pids)
		props = {p.name:p.scalars for p in pif.properties}
		props['formula'] = pif.chemical_formula
		inp = {'Property {}'.format(k):v for (k,v) in props.items() if k not in predict + exclude}
		inp.update(condition)
		inputs.append(inp)
		actuals.append({k:v for (k,v) in props.items() if k in predict})
		
	modelout = client.predict(view_id,inputs)
	predictions = []
	for r in modelout:
		pred = {'pred_{}'.format(p):r.get_value('Property {}'.format(p)).value for p in predict}
		predictions.append(pred)
		
	dicts = []
	for i,a,p in zip(ids,actuals,predictions):
		td = {**i,**a,**p}
		dicts.append(td)
	result = pd.DataFrame(dicts)
	
	return result
	
def formula_input(formula,cat_ox_lims,conditions,red_feat=None):
	if red_feat is None:
		red_feat = formula_redfeat(formula,cat_ox_lims=cat_ox_lims)
	red_inp = {'Property {}'.format(k):v for (k,v) in red_feat.items()}
	inp_dict = {**conditions,**red_inp}
	return inp_dict
	
