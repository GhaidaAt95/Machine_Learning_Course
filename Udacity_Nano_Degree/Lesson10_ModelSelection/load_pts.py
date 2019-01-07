import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from pandas import DataFrame, Series

def load_pts(csv_name):
	data = np.asarray(pd.read_csv(csv_name, header=None))
	X = data[:,0:2]
	y = data[:,2]
	#plt.scatter(x,y,....)
	plt.scatter(X[np.argwhere(y==0).flatten(),0], X[np.argwhere(y==0).flatten(),1],s=50, color = 'blue', edgecolor='k')
	plt.scatter(X[np.argwhere(y==1).flatten(),0], X[np.argwhere(y==1).flatten(),1],s=50, color = 'red', edgecolor='k')

	plt.xlim(-2.05,2.05)
	plt.ylim(-2.05,2.05)
	plt.grid(False)
	plt.tick_params(
		axis = 'x',
		which= 'both',
		bottom= False,
		top= False)
		
	return X,y
	

"""
	# argwhere returns a ndarray of indeces grouped by element
	# flatten : returns a copy pf the array collapsed into one dimension
"""
