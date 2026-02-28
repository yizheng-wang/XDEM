
# Defines material properties
class MaterialProperties:
    def __init__(self, mat_E, mat_nu, w1, l0):
        self.mat_E = mat_E
        self.mat_nu = mat_nu
        self.w1 = w1
        self.l0 = l0
        self.mat_lmbda = self.mat_E*self.mat_nu/(1+self.mat_nu)/(1-2*self.mat_nu)
        self.mat_mu = self.mat_E/(1+self.mat_nu)/2.0

    def __call__(self):
        return self.mat_lmbda, self.mat_mu, self.w1, self.l0
    