import numpy as np
import util

from linear_model import LinearModel


def main(train_path, eval_path, pred_path):
    # Load dataset
    x_train, y_train = util.load_dataset(train_path, add_intercept=False)
    # *** START CODE HERE ***
    
    
    # Train GDA
    model = GDA()
    model.fit(x_train, y_train)

    # Plot data and decision boundary
    util.plot(x_train, y_train, model.theta, 'output/p01e_{}.png'.format(pred_path[-5]))

    # Save predictions
    x_eval, y_eval = util.load_dataset(eval_path, add_intercept=True)
    y_pred = model.predict(x_eval)
    np.savetxt(pred_path, y_pred > 0.5, fmt='%d')

    # *** END CODE HERE ***


class GDA(LinearModel):
    def fit(self, x, y):
        """Fit a GDA model to training set given by x and y.

        Args:
            x: Training example inputs. Shape (m, n).
            y: Training example labels. Shape (m,).

        Returns:
            theta: GDA model parameters.
        """
        # *** START CODE HERE ***
        
        # Init theta
        m, n = x.shape
        self.theta = np.zeros(n+1) # one being the bias term

        # Compute phi, mu_0, mu_1, sigma
        y_1 = sum(y == 1) # y==1 returns an array of same shape as y with true in all places where y==1 and then sum sums them all up i.e the line of code keeps count of all the +ve instances.
        phi = y_1 / m
        mu_0 = np.sum(x[y == 0], axis=0) / (m - y_1) #x[y==0] highlights all the rows where y = 0. then all elements in one column are added togather 
        #x = [[1, 2],    if y = 0 for 1st two rows: np.sum(x[y == 0], axis=0) = [1 + 3, 2 + 4] = [4, 6]
            #[3, 4],
            #[5, 6]]   the sum itself is a matrix with each column = sum of all examples of a single feature. then divided by # of -ve examples
        mu_1 = np.sum(x[y == 1], axis=0) / y_1
        sigma = ((x[y == 0] - mu_0).T.dot(x[y == 0] - mu_0) + (x[y == 1] - mu_1).T.dot(x[y == 1] - mu_1)) / m

        # Compute theta
        sigma_inv = np.linalg.inv(sigma)
        self.theta[0] = 0.5 * (mu_0 + mu_1).T.dot(sigma_inv).dot(mu_0 - mu_1) - np.log((1 - phi) / phi)
        self.theta[1:] = sigma_inv.dot(mu_1 - mu_0)

        # theta is a (n+1,) shaped array.
        
        # Return theta
        return self.theta

        # *** END CODE HERE ***

    def predict(self, x):
        """Make a prediction given new inputs x.

        Args:
            x: Inputs of shape (m, n).

        Returns:
            Outputs of shape (m,).
        """
        # *** START CODE HERE ***
        
        return 1 / (1 + np.exp(-x.dot(self.theta)))

        # *** END CODE HERE
