import pandas as pd
import numpy as np
import random as rd
import matplotlib.pyplot as plt

class KMeans:

    def __init__(self):
        self.K = 3
        self.data = pd.read_csv('clustering.csv')
        self.color=['red','yellow','blue']

    def prepareData(self):
        self.X = self.data[["LoanAmount","ApplicantIncome"]]
        self.Centroids = (self.X.sample(n=self.K))

    def plotClusters(self):
        for k in range(self.K):
            data=self.X[self.X["Cluster"]==k+1]
            plt.scatter(data["ApplicantIncome"],data["LoanAmount"],c=self.color[k])
        plt.scatter(self.Centroids["ApplicantIncome"],self.Centroids["LoanAmount"],c='green')
        plt.xlabel('Income')
        plt.ylabel('Loan Amount (In Thousands)')
        plt.show()

    def kmeans(self):
        diff = 1
        j = 0
        while(diff!=0): #no difference in cluster centers
            XD = self.X
            i = 1
            for index1, row_c in self.Centroids.iterrows():
                ED = []
                for index2, row_d in XD.iterrows():
                    d1 = (row_c["ApplicantIncome"]-row_d["ApplicantIncome"])**2
                    d2=(row_c["LoanAmount"]-row_d["LoanAmount"])**2
                    d=np.sqrt(d1+d2)
                    ED.append(d)
                self.X[i] = ED
                i = i+1
            C=[]
            for index,row in self.X.iterrows():
                min_dist=row[1]
                pos=1
                for i in range(self.K):
                    if row[i+1] < min_dist:
                        min_dist = row[i+1]
                        pos=i+1
                C.append(pos)
            self.X["Cluster"]=C
            Centroids_new = self.X.groupby(["Cluster"]).mean()[["LoanAmount","ApplicantIncome"]]
            if j == 0:
                diff=1
                j=j+1
            else:
                diff = (Centroids_new['LoanAmount'] - self.Centroids['LoanAmount']).sum() + (Centroids_new['ApplicantIncome'] - self.Centroids['ApplicantIncome']).sum()
                print(diff)
            self.Centroids = self.X.groupby(["Cluster"]).mean()[["LoanAmount","ApplicantIncome"]]
            self.plotClusters()

kmeans = KMeans()
kmeans.prepareData()
kmeans.kmeans()