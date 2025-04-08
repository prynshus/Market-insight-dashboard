import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import t

class linreg:
    def __init__(self,a,b):
        self.a = a
        self.b = b

    def Sxx(self):
        x_mean = np.mean(self.a)
        result = 0
        for x in self.a:
            result += (x - x_mean)**2
        return result

    def Syy(self):
        y_mean = np.mean(self.b)
        result = 0
        for y in self.b:
            result += (y - y_mean)**2
        return result

    def Sxy(self):
        x_mean = np.mean(self.a)
        y_mean = np.mean(self.b)
        result = 0
        for x,y in np.column_stack((self.a,self.b)):
            result += (x - x_mean)*(y - y_mean)

        return result

    def LSF_estimators(self):
        x_mean = np.mean(self.a)
        y_mean = np.mean(self.b)
        beta = self.Sxy()/self.Sxx()
        alpha = y_mean - beta * x_mean
        return alpha.item(), beta.item()

    def point_estimation(self,x):
        alpha, beta = self.LSF_estimators()
        return beta*x + alpha

    def plot(self):
        plt.scatter(self.a,self.b)
        x = np.arange(0,np.max(self.a)+1)
        alpha, beta = self.LSF_estimators()
        plt.plot(beta*x + alpha, label = f"y = {alpha:.3f}x + {beta:.3f}")
        plt.legend()
        plt.show()

    def t_dist_plot(self, alpha_ = 0, beta_ = 0, confidence = 0.95, choice=0):
        df = self.a.shape[0] - 2
        x = np.linspace(-4,4, 1000)
        y = t.pdf(x, df)
        t_alpha, t_beta = self.t_values(alpha_,beta_)
        if choice == 0:
            alp = (1-confidence)/2
        else:
            alp = 1-confidence
        plt.plot(x,y)
        if choice == 0 or choice == 1:
            plt.axvline(t.ppf(1-alp, df), color="red", label=f"Critical Value:{t.ppf(1-alp, df):.3f} ({confidence*100}%)")
        if choice == 0 or choice == 2:
            plt.axvline(t.ppf(alp, df), color="red")
        plt.axvline(t_alpha, color="green", linestyle="dashed",
                    label=f"t_alpha: {t_alpha:.3f}")
        plt.axvline(t_beta, color="blue", linestyle="dashed",label=f"t_beta: {t_beta:.3f}")
        plt.xlabel("t-value")
        plt.ylabel("Probability Density")
        plt.title(f"H0: alpha = {alpha_} | H0: beta = {beta_}")
        plt.legend()
        plt.show()

    def standard_error(self):
        return (self.Syy() - (self.Sxy()**2)/self.Sxx()) / (self.a.shape[0] -2)

    def t_values(self,alpha_ = 0,beta_ = 0):
        alpha, beta = self.LSF_estimators()
        se = self.standard_error() ** (1/2)
        n = self.a.shape[0]
        t_alpha = ((alpha - alpha_)/se) * (n*self.Sxx()/(self.Sxx() + n*np.mean(a)**2))**(1/2)
        t_beta = ((beta - beta_) * self.Sxx()**(1/2)) / se
        return t_alpha.item(), t_beta.item()

    def test_hypothesis(self, confi = 0.95):

        crit_val = t.ppf(1 - ((1-confi)/2),self.a.shape[0] - 2)
        t_alpha, t_beta = self.t_values()
        print(f"T alpha: {t_alpha:.3f} | critical value: {crit_val:.3f}")
        print(f"T beta: {t_beta:.3f} | critical value: {crit_val:.3f}")

    def confidence_interval_alpha(self,confidence = 0.95):
        alpha, beta = self.LSF_estimators()
        t_alpha = t.ppf((1-confidence)/2, self.a.shape[0] - 2)
        var = t_alpha * (self.standard_error())**(1/2) * ((1/self.a.shape[0]) + ((np.mean(self.a)**2) / (self.Sxx()))) ** (1/2)
        return alpha - var, alpha + var

    def expo_transformation(self):
        self.b = np.log(self.b)
        log_alpha, log_beta = self.LSF_estimators()
        return log_alpha, log_beta

    def sample_corr_coeff(self):
        r = self.Sxy()/(self.Sxx() * self.Syy())**(1/2)
        return r


