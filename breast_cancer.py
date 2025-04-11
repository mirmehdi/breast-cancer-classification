import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt 
import seaborn as sns
import io
from utils import Classification


from sklearn.datasets import load_breast_cancer

# Load data
cancer = load_breast_cancer()
cancer_df = pd.DataFrame(np.c_[cancer['data'], cancer['target']], columns = np.append(cancer['feature_names'], ['target']))
cancer_df


# Open a text file to write descriptive statistic report
with open("breast_cancer_descriptive_analysis.txt", "w") as f: 
    # Dataset info
    f.write('===info===\n')
    buffer2 = io.StringIO()
    cancer_df.info(buf=buffer2)
    f.write(buffer2.getvalue() + "\n\n")

    # Pearson correlation
    x = cancer_df['mean radius']
    y = cancer_df['mean texture']
    f.write('===pearson value===\n')
    f.write("correlation_coef(mean radius, mean texture) = " + str(pearsonr(x, y)[0]) + "\n\n")
    f.write("p_value(mean radius, mean texture)  = " + str(pearsonr(x, y)[1]) + "\n\n")

cancer_df.columns
################ visualization 

sns.pairplot(cancer_df, hue = 'target', vars = ['mean radius', 'mean texture', 'mean area', 'mean perimeter', 'mean smoothness'] )

plt.savefig('pairplot')
plt.close()


sns.scatterplot(data = cancer_df, x = 'mean area', y = 'mean smoothness', hue = 'target' )
plt.savefig('scatter_meanarea_meansmoothness')
plt.close()

sns.heatmap(cancer_df.corr(), annot = True)
plt.savefig('heatmap_cc')
plt.close()

############## classification 

 # Split into features and target
X = cancer_df.drop(columns=['target'])
y = cancer_df['target']

# use utils model and train/test clasification 
# Run classification pipeline
clf = Classification(X, y)
clf.preprocess()
clf.train_evaluate()
clf.cross_validate()
y_pred, best_params = clf.optimization_best_model()




