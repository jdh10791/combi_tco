import pickle

def save_pickle(obj, file):
	with open(file,'wb') as f:
		pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)
	print('Dumped pickle to {}'.format(file))
	
def load_pickle(file):
	with open(file,'rb') as f:
		return pickle.load(f)
		
class pickle_dict():
	def __init__(self,file):
		self.set_file(file)
		try:
			self.dict = load_pickle(file)
		except FileNotFoundError:
			self.dict = {}

	def reload(self):
		'''
		reload dict from file. Overwrites dict
		'''
		reloaded_dict = load_pickle(self.file)
		self.dict = reloaded_dict
		
	def update_dict(self):
		'''
		update dict from file
		'''
		updated_dict = load_pickle(self.file)
		self.dict.update(updated_dict)
		
	def save(self,file=None):
		'''
		save dict to file. Overwrites file
		'''
		if file is None:
			file = self.file
		save_pickle(self.dict,file)
	
	def update_file(self):
		'''
		update file from dict
		'''
		file_dict = load_pickle(self.file)
		file_dict.update(self.dict)
		self.save()
	
	def set_file(self,file):
		'''
		change file that dict is saved to/loaded from
		'''
		self._file = file
	
	def _get_file(self):
		return self._file
		
	file = property(_get_file,set_file)
	