
import numpy as np
import pandas as pd
import pymatgen as mg
from pymatgen.ext.matproj import MPRester
import os
import warnings
import itertools
from pypif.obj import ChemicalSystem, Property

class matproj_calc:
	def __init__(self,oxide_dict={}):
		#dict to specify which oxide to use for metal
		self.oxide_dict = oxide_dict
		#dict for materials known to have incorrect formation enthalpy in mat proj
		self.fH_corrections = {
				('Ce','gas'):417.1 #fH for Ce gas is negative in MP
						  }
		#dict to store oxygen bond energies after calculation. Avoid repeated lookups in MP
		self.calc_obe = {} 
		self.DO2 = 498.34 #O2 dissociation energy
		self.mp = MPRester(os.environ['MATPROJ_API_KEY'])
		
	def get_fH(self,formula, phase='solid'): #get exp formation enthalpy
		"Get average experimental formation enthalpy for formula and phase"
		#first check for corrected data in fH_corrections
		try:
			fH = self.fH_corrections[(formula,phase)]
		#if no correction exists, look up in MP
		except KeyError:
			results = self.mp.get_exp_thermo_data(formula)
			if phase=='solid':
				phase_results = [r for r in results if r.type=='fH' and r.phaseinfo not in ('liquid','gas')]
			else:
				phase_results = [r for r in results if r.type=='fH' and r.phaseinfo==phase]
			phases = np.unique([r.phaseinfo for r in phase_results])
			fH = [r.value for r in phase_results]
			if len(fH)==0:
				raise LookupError('No thermo data for {} in {} phase'.format(formula,phase))
			maxdiff = np.max(fH) - np.min(fH)
			if maxdiff > 15:
				warnings.warn('Max discrepancy of {} in formation enthalpies for {} exceeds limit'.format(maxdiff,formula))
			print('Formation enthalpy for {} includes data from phases: {}'.format(formula,', '.join(phases)))
			#print('Max difference: {}'.format(maxdiff))
		return np.mean(fH)

	def oxide_formula(self,metal,ox_state,return_mn=False):
		"Get metal oxide formula with integer units"
		#formula MmOn
		OM_ratio = ox_state/2
		if OM_ratio%1 == 0.5:
			m = 2
		else:
			m = 1
		n = m*OM_ratio
		formula = '{}{}O{}'.format(metal,m,n)
		if return_mn==False:
			return formula
		else:
			return formula, m, n

	def oxide_obe(self,formula): #M-O bond energy per mol of oxide
		"Get molar metal-oxygen bond energy (OBE) for simple metal oxide"
		try:
			obe = self.calc_obe[formula]
		except KeyError:
			comp = mg.Composition(formula)
			cd = comp.get_el_amt_dict()
			for el, amt in cd.items():
				if el=='O':
					n = amt
				else:
					metal = el
					m = amt
			fH = self.get_fH(comp.reduced_formula) #oxide formation enthalpy
			H_sub = self.get_fH(metal, phase='gas') #metal sublimation enthalpy
			obe = fH - m*H_sub - (n/2)*self.DO2 #M-O bond energy
			self.calc_obe[formula] = obe
		return obe

	def get_ABE(self,formula,A_site,B_site,verbose=False):
		"""
		Estimate average metal-oxygen bond energy for complex perovskite oxide from simple oxide thermo data
		Formula from Sammells et al. (1992), Solid State Ionics 52, 111-123.
		
		Parameters:
		-----------
		formula: oxide formula
		A_site: list of A-site elements
		B_site: list of B-site elements
		verbose: if True, print info about which simple oxides used in calculation
		"""
		#validated on compounds in Sammells 1992 - all but CaTi0.7Al0.3O3 agree
		#validated on (La,Sr)(Cr,Co,Fe)O3 compounds in https://pubs.acs.org/doi/suppl/10.1021/acs.jpcc.6b10571/suppl_file/jp6b10571_si_001.pdf
			#works if Co3O4 specified in oxide_dict
		comp = mg.Composition(formula)
		cd = comp.get_el_amt_dict()
		metals = [x for x in cd.keys() if x!='O']
		abe = 0
		if verbose==True:
			print('Oxides used for ABE calculation:')
		for metal in metals:
			amt = cd[metal]
			met_mg = mg.Element(metal)

			try: #oxide_dict specifies which oxide to use
				oxide = self.oxide_dict[metal]
				oxide_mg = mg.Composition(oxide)
				m = oxide_mg.get(metal)
				n = oxide_mg.get('O')
				obe = self.oxide_obe(oxide)
			except KeyError: #if no oxide indicated in oxide_dict
				"placeholder - for now, take the lowest common oxidation state with a corresponding stable oxide"
				i = 0
				while i != -1: 
					ox = met_mg.common_oxidation_states[i]
					oxide, m ,n = self.oxide_formula(metal,ox,return_mn=True)
					try:
						obe = self.oxide_obe(oxide)
						#print(obe)
						i = -1
					except LookupError as err:
						i += 1 #try the next oxidation state

			if verbose==True:
				print(oxide)
			#print('m: {}, n: {}'.format(m,n))
			if metal in A_site:
				abe += amt*obe/(12*m)
			elif metal in B_site:
				abe += amt*obe/(6*m)
			else:
				raise KeyError('{} is not assigned to A or B site'.format(metal))
			#print(abe)

		return abe
		



def bond_IC(a,b): 
	"""
	ionic character of bond between elemenets a and b based on Pauling electronegativities
	"""
	a_ = mg.Element(a)
	b_ = mg.Element(b)
	return 1-np.exp(-0.25*(a_.X-b_.X)**2)


class perovskite:
	def __init__(self, 
				 formula,
				 cation_site={'Ba':'A','Co':'B','Fe':'B','Zr':'B','Y':'B'},
				 site_ox_lim={'A':[2,4],'B':[2,4]},
				 site_base_ox={'A':2,'B':4}):
		self.formula = formula
		
		#remove cations not in formula from cation_site dict
		rem = [c for c in cation_site.keys() if c not in self.cations]
		cs = cation_site.copy()
		for r in rem:
			del cs[r]
		self.cation_site = cs
		
		#create A_site and B_site lists for convenience
		self.A_site = [c for c in self.cations if self.cation_site[c]=='A']
		self.B_site = [c for c in self.cations if self.cation_site[c]=='B']
		
		self.site_ox_lim = site_ox_lim
		
		#initialize cation_ox_lim dict
		self._cat_ox_lim = {}
		for c in self.cations:
			self.set_cat_ox_lim(c,self.site_ox_lim[self.cation_site[c]])
		
		self.site_base_ox = site_base_ox
		
		#initialize _ox_combos
		#self.set_ox_combos()
		
		#check if any cations in formula not assigned to a site
		unassigned = [c for c in self.cations if c not in self.A_site + self.B_site]
		if len(unassigned) > 0:
			raise Exception('Cation(s) not assigned to a site: {}'.format(unassigned))
	
	@property
	def site_cn(self):
		'''
		standard coordination numbers for each site
		'''
		return {'A':12,'B':6}
	
	def get_cat_ox_lim(self):
		'''
		getter for cat_ox_lim
		'''
		return self._cat_ox_lim
	
	def set_cat_ox_lim(self,cat,lim):
		'''
		setter for cat_ox_lim
		'''
		self._cat_ox_lim[cat] = lim
		#print('Set lim')
		
	cat_ox_lim = property(get_cat_ox_lim,set_cat_ox_lim, 
						  doc='''
						  oxidation state limits for each cation 
						  (doesn\'t reflect physically allowed oxidation states,
						  just user-defined limits on the range of physical oxidation states that will be considered)''')
	
	
	@property
	def composition(self):
		'''
		pymatgen.Composition object
		'''
		return mg.Composition(self.formula)
	
	@property
	def el_amts(self):
		'''
		dict of formula units for each element
		'''
		return self.composition.get_el_amt_dict()
	
	@property
	def cations(self):
		'''
		cations in formula
		'''
		return [x for x in self.el_amts.keys() if x!='O']
	
	@property
	def A_sum(self):
		'''
		total A-site formula units
		'''
		return np.sum([self.el_amts[c] for c in self.A_site])
	
	@property
	def B_sum(self):
		'''
		total A-site formula units
		'''
		return np.sum([self.el_amts[c] for c in self.B_site])
	
	@property
	def norm_cat_wts(self):	 
		'''
		weights for A and B site cations, normalized to each site total
		'''
		nwa = [self.el_amts[m]/self.A_sum for m in self.A_site]
		nwb = [self.el_amts[m]/self.B_sum for m in self.B_site]
		return dict(zip(self.A_site+self.B_site,nwa+nwb))
	
	def siteavg_mg_prop(self,property_name): 
		'''
		general function for averaging properties in mg.Element data dict across A and B sites
		'''
		p_a = np.sum([self.norm_cat_wts[m]*mg.Element(m).data[property_name] for m in self.A_site])
		p_b = np.sum([self.norm_cat_wts[m]*mg.Element(m).data[property_name] for m in self.B_site])
		p_tot = (p_a*self.A_sum + p_b*self.B_sum)/(self.A_sum + self.B_sum)
		return p_a, p_b, p_tot
	
	def siteavg_func(self,func,**kwargs):
		'''
		general function for averaging the value of a function "func" across A and B sites
		'''
		f_a = np.sum([self.norm_cat_wts[m]*func(m,**kwargs) for m in self.A_site])
		f_b = np.sum([self.norm_cat_wts[m]*func(m,**kwargs) for m in self.B_site])
		f_tot = (f_a*self.A_sum + f_b*self.B_sum)/(self.A_sum + self.B_sum)
		return f_a, f_b, f_tot
			
	@property
	def allowed_ox_states(self):
		"""
		returns allowed cation oxidation states as dict of tuples
		"""
		ox_states = dict()
		for c in self.cations:
			el = mg.Element(c)
			if len(el.common_oxidation_states)==1: #if only 1 commmon ox state, choose that
				ox_states[c] = el.common_oxidation_states
			else: #otherwise take ox states corresponding to Shannon radii
				oxlim = self.cat_ox_lim[c]
				ox_states[c] = tuple([int(x) for x in el.data['Shannon radii'].keys() 
									  if oxlim[0] <= int(x) <= oxlim[1]])
		return ox_states
	
	@property 
	def multivalent_cations(self):
		'''
		cations with multiple allowed oxidation states
		'''
		return [c for c in self.cations if len(self.allowed_ox_states[c]) > 1]
	
	@property
	def _rom_to_num(self): 
		'''
		roman numeral dict - needed for Shannon radii coordination numbers
		'''
		return {'I':1,'II':2,'III':3,'IV':4,'V':5,'VI':6,'VII':7,'VIII':8,'IX':9,'X':10,'XI':11,'XII':12}
	
	@property
	def _num_to_rom(self):
		'''
		reverse lookup roman numeral dict
		'''
		return {v:k for k,v in self._rom_to_num.items()}
	
	def _closest_CN(self, el, ox, target): 
		'''
		get closest coordination number (in roman numerals) to target
		convenience function for choosing appropriate Shannon radius - used in ox_combos
		'''
		cn_rom = mg.Element(el).data['Shannon radii']['{}'.format(ox)].keys()
		cn_rom = [rn for rn in cn_rom if rn in self._rom_to_num.keys()] #remove any roman numerals not in dict (eg "IVSQ")
		cn = [self._rom_to_num[rn] for rn in cn_rom]
		idx = np.argmin(np.abs(np.array(cn)-target))
		rom = self._num_to_rom[cn[idx]]
		return rom
	
	#@property
	def set_ox_combos(self):
		"""
		considers all possible combinations of oxidation states for multivalent cations
		returns dict of oxidation state-dependent properties for each combination
		"""
		#anion radius - 6-coordinated
		rx = mg.Element('O').data['Shannon radii']['-2']['VI']['']['ionic_radius']
		
		#fixed-valence cations
		fixed = [c for c in self.cations if len(self.allowed_ox_states[c]) == 1]
		fixed_dict = dict()
		for f in fixed:
			ox = self.allowed_ox_states[f][0]
			el = mg.Element(f)
			#get closest CN to CN for site
			cn = self.site_cn[self.cation_site[f]]
			rom = self._closest_CN(f,ox,cn)
			print('fixed:',f,ox,rom)
			#should only be one spin state
			fixed_dict[f] = {'r':el.data['Shannon radii']['{}'.format(ox)][rom]['']['ionic_radius'],
							 'n':ox}
		#multivalent cations
		multival = [c for c in self.cations if len(self.allowed_ox_states[c]) > 1]
		multi_combos = []
		multi_dict = dict()
		for m in multival:
			multi_combos.append(self.allowed_ox_states[m])
			md = {}
			#get Shannon radius for each ox state
			for ox in self.allowed_ox_states[m]:
				el = mg.Element(m)
				#get closest CN to CN for site
				cn = self.site_cn[self.cation_site[m]]
				rom = self._closest_CN(m,ox,cn)
				print('multi:',m,ox,rom)
				try: #assume transition metal assumes high-spin state if available (assumption made by Bartel)
					md[ox] = {'r':el.data['Shannon radii']['{}'.format(ox)][rom]['High Spin']['ionic_radius'],
								 'n':ox}
				except KeyError:
					md[ox] = {'r':el.data['Shannon radii']['{}'.format(ox)][rom]['']['ionic_radius'],
								 'n':ox}
			multi_dict[m] = md
		
		dicts = []
		for tup in itertools.product(*multi_combos): #get all combinations of oxidation states for multivalents
			cat_dict = fixed_dict.copy()
			for m, ox in zip(multival,tup):
				#print(m,ox)
				cat_dict.update({m:multi_dict[m][ox]})
			#print(cat_dict)
			#get site averaged radii and oxidation states
			ra = np.sum([self.norm_cat_wts[c]*cat_dict[c]['r'] for c in self.A_site])
			rb = np.sum([self.norm_cat_wts[c]*cat_dict[c]['r'] for c in self.B_site])
			na = np.sum([self.norm_cat_wts[c]*cat_dict[c]['n'] for c in self.A_site])
			nb = np.sum([self.norm_cat_wts[c]*cat_dict[c]['n'] for c in self.B_site])
			#"cation electronegativity" (as defined in Zohourian 2018: z/r^2)
			X_cat_a = np.sum([self.norm_cat_wts[c]*cat_dict[c]['n']/cat_dict[c]['r']**2 for c in self.A_site])
			X_cat_b = np.sum([self.norm_cat_wts[c]*cat_dict[c]['n']/cat_dict[c]['r']**2 for c in self.B_site])
			
			#B-site radius deviation
			#std = sum_i(wt_i*(r_i-ravg)^2)^1/2
			rb_std = np.sum([self.norm_cat_wts[b]*(cat_dict[b]['r'] - rb)**2 for b in self.B_site])**0.5
			
			#tolerance factors
			goldschmidt = (ra+rx)/((2**0.5)*(rb+rx))
			tau = (rx/rb)-na*(na-(ra/rb)/np.log(ra/rb)) #Bartel 2018 improved tolerance factor
			
			#total cation charge & oxygen delta
			tot_cat_charge = na*self.A_sum + nb*self.B_sum
			O_delta = (6 - tot_cat_charge)/2 #oxygen non-stoich
			
			#unit cell volume and free volume (assume cubic)
			if goldschmidt > 1: #A-O bonds are close packed
				alat = (2**0.5)*(ra + rx)
			elif goldschmidt <= 1: #B-O bonds are close packed
				alat = 2*(rb + rx)
			vol = alat**3
			volf = vol - (4*np.pi/3)*(ra**3+rb**3+(3-O_delta)*rx**3)
			
			#critical radius of saddle point - Liu 2011
			rc = (-ra**2 + (3/4)*alat**2 - (2**0.5)*alat*rb + rb**2) / (2*ra + (2**0.5)*alat - 2*rb)
			
			
			#put all features in dict for current ox combo
			dicts.append({'cations':cat_dict,
						  'r_a':ra,
						  'r_b':rb,
						  'n_a':na,
						  'n_b':nb,
						  'X_cat_a':X_cat_a,
						  'X_cat_b':X_cat_b,
						  'r_b_std':rb_std,
						  'goldschmidt':goldschmidt,
						  'tau':tau,
						  'tot_cat_charge':tot_cat_charge,
						  'O_delta':O_delta,
						  'alat':alat,
						  'uc_vol':vol,
						  'uc_vol_free':volf,
						  'r_crit':rc})
		
		#print('called set_ox_combos') #used to verify that set_ox_combos runs only once initially
		self._ox_combos = dicts
		
	def get_ox_combos(self):
		return self._ox_combos
	
	ox_combos = property(get_ox_combos,set_ox_combos)
	
	#functions for ox_state-dependent properties
	def oxprop_range(self,prop):
		'''
		range (min and max as list) for oxidation state-dependent property
		'''
		val = [d[prop] for d in self.ox_combos]
		return min(val), max(val)
	
	def oxprop_avg(self,prop):
		'''
		average value for oxidation state-dependent property
		'''
		val = [d[prop] for d in self.ox_combos]
		return np.mean(val)
				
	@property
	def ABE(self):
		'''
		average M-O bond energy
		'''
		#if matproj_calc instance already exists, use it; otherwise create global instance (to be used in other perovskite instances)
		global mpcalc
		try:
			mpcalc
		except NameError:
			mpcalc = matproj_calc(oxide_dict={'Co':'Co3O4'})
			print('Created MPRester instance')
		return mpcalc.get_ABE(self.formula,self.A_site,self.B_site)
	
	@property
	def aliovalents(self):
		'''
		dict of aliovalent cations and their valence differences
		'''
		#only consider single-valence ions
		#don't treat transition metals as aliovalent - assume they take on base oxidation state for site
		alio = {}
		for c,site in self.cation_site.items():
			if len(self.allowed_ox_states[c])==1:
				if self.allowed_ox_states[c][0] != self.site_base_ox[site]:
					alio[c] = self.allowed_ox_states[c][0] - self.site_base_ox[site]
		return alio
	
	@property
	def acceptors(self):
		'''
		list of acceptor dopants
		'''
		return [k for k,v in self.aliovalents.items() if v < 0]
	
	@property
	def donors(self):
		'''
		list of donor dopants
		'''
		return [k for k,v in self.aliovalents.items() if v > 0]
		
	@property
	def acceptor_mag(self): 
		'''
		magnitude of acceptor doping (acceptor amt*valence delta)
		'''
		return np.sum([self.el_amts[a]*self.aliovalents[a] for a in self.acceptors])
	
	@property
	def donor_mag(self): 
		'''
		magnitude of donor doping (donor amt*valence delta)
		'''
		return np.sum([self.el_amts[a]*self.aliovalents[a] for a in self.donors])
	
	@property
	def alio_net_mag(self): 
		'''
		net magnitude of aliovalent doping (amt*valence delta)
		'''
		return np.sum([self.el_amts[a]*self.aliovalents[a] for a in self.aliovalents])
	
	def set_ox_feats(self,feature):
		'''
		convenience function for setting oxidation-state-dependent features
		'''
		self.features['{}_oxavg'.format(feature)] = self.oxprop_avg(feature)
		self.features['{}_oxmin'.format(feature)], self.features['{}_oxmax'.format(feature)] = self.oxprop_range(feature)
			
	def featurize(self):
		'''
		generate chemical features as dict
		'''
		self.features = {}
		self.set_ox_combos()
		
		#oxidation-state-dependent features
		self.set_ox_feats('uc_vol')
		self.set_ox_feats('uc_vol_free')
		self.set_ox_feats('r_crit')
		self.set_ox_feats('n_a')
		self.set_ox_feats('n_b')
		self.set_ox_feats('goldschmidt')
		self.set_ox_feats('tau')
		self.set_ox_feats('r_a')
		self.set_ox_feats('r_b')
		self.set_ox_feats('r_b_std')
		self.set_ox_feats('X_cat_a')
		self.set_ox_feats('X_cat_b')
		self.set_ox_feats('alat')
		self.set_ox_feats('O_delta')
		self.set_ox_feats('tot_cat_charge')
		
		#other features
		self.features['MO_ABE'] = self.ABE
		self.features['MO_IC_a'], self.features['MO_IC_b'], self.features['MO_IC_avg'] = self.siteavg_func(bond_IC,b='O')
		self.features['acceptor_magnitude'] = self.acceptor_mag
		self.features['trans_met_amt'] = np.sum([self.el_amts[c] for c in self.multivalent_cations])
		self.features['X_a'],self.features['X_b'], self.features['X_avg'] = self.siteavg_mg_prop('X')
		self.features['mass_a'],self.features['mass_b'], self.features['mass_avg'] = self.siteavg_mg_prop('Atomic mass')
		try:
			self.features['Co:Fe_ratio'] = self.el_amts['Co']/self.el_amts['Fe']
		except ZeroDivisionError:
			self.features['Co:Fe_ratio'] = 1000
			#print('Set Co:Fe_ratio to 1000 due to ZeroDivisionError (no Fe)')
		self.features['Ba_amt'] = self.el_amts['Ba']
		self.features['Co_amt'] = self.el_amts['Co']
		self.features['Fe_amt'] = self.el_amts['Fe']
		self.features['Zr_amt'] = self.el_amts['Zr']
		self.features['Y_amt'] = self.el_amts['Y']
		self.features['A_sum'] = self.A_sum
		self.features['B_sum'] = self.B_sum
		self.features['A:B_ratio'] = self.A_sum/self.B_sum
		
		return self.features
		
def formula_redfeat(formula,cat_ox_lims={}):
    pvskt = perovskite(formula,site_ox_lim={'A':[2,4],'B':[2,4]},site_base_ox={'A':2,'B':4})
    for k,v in cat_ox_lims.items():
        pvskt.set_cat_ox_lim(k,v)
    pvskt.featurize()
    red_feat = {'{}'.format(k):v for (k,v) in pvskt.features.items() 
                if k[-5:] not in ['oxmin','oxmax'] and k[0:7]!='O_delta'}
    return red_feat
	
def formula_pif(formula,cat_ox_lims={},red_feat=None):
    '''
    create pif with formula and chemical feature properties
    '''
    fpif = ChemicalSystem()
    fpif.chemical_formula = formula
    if red_feat is None:
        red_feat = formula_readfeat(formula,cat_ox_lims)
    
    props = []
    for feat, val in red_feat.items():
        prop = Property(name=feat,scalars=val)
        props.append(prop)
    fpif.properties=props
    
    return fpif, red_feat

