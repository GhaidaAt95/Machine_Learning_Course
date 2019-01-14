import pandas as pd
import numpy as np
from mlens.visualization import corrmat
import seaborn as sns

sns.set()
sns.diverging_palette(240, 10, n=9)
corrmat(P.corr(), inflate=False)