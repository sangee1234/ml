'''
PCA
It is a method of dimensionality reduction, where data with m-columns is projected to fewer columns
The data can be decomposed into different orientations with diff magnitudes, the orientations having highest contribution is talem
i.e. directions that maximizes the variance
Steps:
1.Find mean of each column and subtract from each value in column C=A-M
2.Find V=cov(C)
3.Calculate eigenvectors and eigenvalues values by eigen decomposition of V, vectors = eig(C)
4.Select top eigenvectors
5.Project data P= vectors.T.dot(C.T)

In detail:
1. Noramilze the data i.e. value-mean/std, it is done so that each variable contributes equally
2. To get all possible relationship between different dimensions is to calculate conariance among all and put them in a covariance matrix
   Sometimes vectors maybe highly correlated storing redundant info 
3. Choose the number of principal components by considering tradeoff between components and loss if information
4. Calculate eigenvectors and eigenvalues of covariane matrix, to get direction of most variance
5. Percentage of contribution of each vector is given by eigenvalue/sum(eigenvalues), can decide if need to include or not
6. These final eigenvectors give the feature vectors, can cast the dataset by feature_vector^t*dataset^t
7. PCA can be applied to reduce training time

'''
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

class IrisClassification:

    def __init__(self):
        self.url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
        self.n_components = 2

    def preparedata(self):
        self.df = pd.read_csv(self.url, names=['sepal length','sepal width','petal length','petal width','target'])
        features = ['sepal length','sepal width','petal length','petal width']
        x = self.df.loc[:, features].values
        self.y = self.df.loc[:,['target']].values
        self.x = StandardScaler().fit_transform(x)

    def applyPCA(self):
        pca = PCA(n_components=self.n_components)
        principalComponents = pca.fit_transform(self.x)
        principalDf = pd.DataFrame(data = principalComponents, columns = ['principal component 1', 'principal component 2'])
        self.finalDf = pd.concat([principalDf, self.df[['target']]], axis = 1)

    def visulizeResult(self):
        fig = plt.figure(figsize=(8,8))
        ax = fig.add_subplot(1,1,1)
        ax.set_xlabel('Principal Component 1')
        ax.set_ylabel('Principal Component 2')
        ax.set_title('2 component PCA')
        targets = ['Iris-setosa', 'Iris-versicolor', 'Iris-virginica']
        colors = ['r', 'g', 'b']
        for target, color in zip(targets,colors):
            indicesToKeep = self.finalDf['target'] == target
            ax.scatter(self.finalDf.loc[indicesToKeep, 'principal component 1'], self.finalDf.loc[indicesToKeep, 'principal component 2'], c = color, s = 50)
        ax.legend(targets)
        ax.grid()
        plt.show()

iris = IrisClassification()
iris.preparedata()
iris.applyPCA()
iris.visulizeResult()
