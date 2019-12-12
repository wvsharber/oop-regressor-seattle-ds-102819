class MeanRegressor:
    """
    This is a class that implements a similar interface to a 'sklearn' regressor. Except it only predicts the mean of the target variable.
    """

    
    mean = None

    def fit(self,X,y):
        # method to fit the mean
        self.mean = sum(y)/len(y)
    
    def predict(self,X):
        # 'predicts' the mean value for each element of X
        return [self.mean for i in range(len(X))]
               
    def score(self,X,y):
        # this returns the R^2 value, comparing the provided test data vs fitted training data
        predictions = self.predict(X)
        resids_sq = []
        mean_dif = []
        zipped = zip(predictions,y)
        mapped = set(zipped)
        for i in mapped:
            resids_sq.append((i[1]-i[0])**2)
            mean_dif.append((i[1]-(sum(y)/len(y)))**2)
        tss = sum(mean_dif)  
        rss = sum(resids_sq)
        return 1 - (rss/tss)
        

         

    