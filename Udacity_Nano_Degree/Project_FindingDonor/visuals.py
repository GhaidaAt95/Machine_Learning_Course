import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
'''
    Built upon Code from Udacity ML class
'''
def distribution(data,features,transformed= False):
    # Visualizing skewed distributions of features

    fig = plt.figure()

    n_features = len(features)
    n_rows, n_cols = 0, 0
    if n_features > 6:
        print("6 features Max at a time")
        return
    elif n_features > 3:
         n_rows = 2
         n_cols = 3
    else:
        n_rows = 1
        n_cols = n_features
    colors = ['#00A0A0','#0030A0','#004DDD','#D900DD','#00DD72','#12DD00']
    for i, feature in enumerate(features):
        # print("i: {}\nfeature: {}".format(i,feature))
        ax = fig.add_subplot(n_rows, n_cols, i+1)
        ax.hist(data[feature], bins = 25, color = colors[i])
        ax.set_title("'%s' Feature Distribution"%(feature), fontsize=14)
        ax.set_xlabel("Value")
        ax.set_ylabel("Number of records")
        ax.set_ylim((0, 2000))
        ax.set_yticks([0, 500, 1000, 1500, 2000])
        ax.set_yticklabels([0, 500, 1000, 1500, ">2000"])

    # Plot aesthetics
    if transformed:
        fig.suptitle("Log-transformed Distributions of Continuous Census Data Features", \
            fontsize = 16)
    else:
        fig.suptitle("Skewed Distributions of Continuous Census Data Features", \
            fontsize = 16)

    fig.tight_layout()
    fig.show()


def evaluate(results, accuracy, f1):
    fig , ax = plt.subplots(2,4, figsize=(11,7))

    bar_width = 0.3
    colors = ['#A00000','#00A0A0','#00A000']

    for k, learner in enumerate(results.keys()):
        # print("k = {} and learner is {}".format(k, learner))
        for j, metric in enumerate(['train_time', 'acc_train', 'f_train', 'pred_time', 'acc_test', 'f_test']):
            # print("j = {} and metric = {}".format(j, metric))
            for i in np.arange(3):
                ax[j//3, j%3].bar(i+k*bar_width, results[learner][i][metric], width = bar_width, color = colors[k])
                ax[j//3, j%3].set_xticks([0.45, 1.45, 2.45])
                ax[j//3, j%3].set_xticklabels(["1%", "10%", "100%"])
                ax[j//3, j%3].set_xlabel("Training Set Size")
                ax[j//3, j%3].set_xlim((-0.1, 3.0))
    
    # Add unique y-labels
    ax[0, 0].set_ylabel("Time (in seconds)")
    ax[0, 1].set_ylabel("Accuracy Score")
    ax[0, 2].set_ylabel("F-score")
    ax[1, 0].set_ylabel("Time (in seconds)")
    ax[1, 1].set_ylabel("Accuracy Score")
    ax[1, 2].set_ylabel("F-score")
    
    # Add titles
    ax[0, 0].set_title("Model Training")
    ax[0, 1].set_title("Accuracy Score on Training Subset")
    ax[0, 2].set_title("F-score on Training Subset")
    ax[1, 0].set_title("Model Predicting")
    ax[1, 1].set_title("Accuracy Score on Testing Set")
    ax[1, 2].set_title("F-score on Testing Set")
    
    # Add horizontal lines for naive predictors
    ax[0, 1].axhline(y = accuracy, xmin = -0.1, xmax = 3.0, linewidth = 1, color = 'k', linestyle = 'dashed')
    ax[1, 1].axhline(y = accuracy, xmin = -0.1, xmax = 3.0, linewidth = 1, color = 'k', linestyle = 'dashed')
    ax[0, 2].axhline(y = f1, xmin = -0.1, xmax = 3.0, linewidth = 1, color = 'k', linestyle = 'dashed')
    ax[1, 2].axhline(y = f1, xmin = -0.1, xmax = 3.0, linewidth = 1, color = 'k', linestyle = 'dashed')
    
    # Set y-limits for score panels
    ax[0, 1].set_ylim((0, 1))
    ax[0, 2].set_ylim((0, 1))
    ax[1, 1].set_ylim((0, 1))
    ax[1, 2].set_ylim((0, 1))

    # Set additional plots invisibles
    ax[0, 3].set_visible(False)
    ax[1, 3].axis('off')

    # Create legend
    for i, learner in enumerate(results.keys()):
        plt.bar(0, 0, color=colors[i], label=learner)
    plt.legend()
    
    # Aesthetics
    plt.suptitle("Performance Metrics for Three Supervised Learning Models", fontsize = 16, y = 1.10)
    plt.tight_layout()   
    fig.show()