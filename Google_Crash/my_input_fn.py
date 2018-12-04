import numpy as np
import pandas as pd
from sklearn import metrics
import tensorflow as tf
from tensorflow.python.data import Dataset

"""
	* features   : DataFrame of features
	* targets    : DataFrame of labels/tagrets
	* batch_size : Size of batches to be passed to the model
					Default = 1
	* shuffle    : Whether to shuffle the data (passed to the model 
						randomly during training)
	* num_epochs : Number of epochs for which data should be repeated. 
						None = repeat indefinitely
	
	Returns a tuple of features and labels for the next data batch
"""

def my_input_fn(features, targets, batch_size=1,shuffle=True, num_epochs=None):
	#### Step1 : Convert Pandas Data into a dict of np arrays
	features = {key:np.array(value) for key, value in dict(features).items()}
	
	### Step2 : Construct a dataset object, then configure batching and repeating
	dataset = Dataset.from_tensor_slices((features,targets))
	dataset = dataset.batch(batch_size).repeat(num_epochs)
	
	### Step 3: Shuffle
	if shuffle:
		# buffer_size specifies the size of the dataset from which shufle will randomly sample
		dataset = dataset.shuffle(buffer_size=10000)
		
	features, labels, = dataset.make_one_shot_iterator().get_next()
	return features, labels
