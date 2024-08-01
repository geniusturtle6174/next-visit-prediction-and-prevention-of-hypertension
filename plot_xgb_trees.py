import matplotlib.pyplot as plt
from xgboost import plot_tree

import util

model = util.load_pkl('models/240714_xgb-top-01/model_0.pkl')

plot_tree(model, num_trees=0)
plt.show()
