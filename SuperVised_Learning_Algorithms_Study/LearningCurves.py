from sklearn.model_selection import learning_curve
import matplotlib.pyplot as plt
def plot_learning_curve(train_sizes, train_scores_mean, validation_scores_mean, name=" "):
    plt.figure()
    plt.style.use('seaborn')

    plt.plot(train_sizes, train_scores_mean, label='Training Error')
    plt.plot(train_sizes, validation_scores_mean, label='Validation Error')

    plt.ylabel('MSE', fontsize = 14)
    plt.xlabel('Training set size', fontsize = 14)
    title = 'Learning curves for a linear regression model ' + name
    plt.title(title, fontsize = 18, y=1.00)
    plt.legend()
    plt.ylim(0,40)
	
def learning_curve_train(clf,features, target,n_instances):
    train_sizes =[1, int(0.014 * n_instances),int(0.052 * n_instances),int(0.21 * n_instances),int(0.52*n_instances),int(0.8*n_instances) ]

    train_sizes, train_scores, validation_scores = learning_curve(estimator = clf,\
                                                                  X = features,\
                                                                  y = target,\
                                                                  train_sizes= train_sizes,\
                                                                  cv = 5,\
                                                                  shuffle=False,\
                                                                  scoring='accuracy' )
    train_scores_mean = train_scores.mean(axis = 1)
    validation_scores_mean = validation_scores.mean(axis = 1)

    plot_learning_curve(train_sizes, train_scores_mean, validation_scores_mean)