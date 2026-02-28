import warnings

# Defines phase field fracture model
class PFFModel:
    def __init__(self, PFF_model = 'AT1', se_split = 'volumetric', tol_ir = 5e-3):
        self.PFF_model = PFF_model
        self.se_split = se_split
        self.tol_ir = tol_ir
        
        if self.se_split != 'volumetric':
            warnings.warn('Prescribed strain energy split is not volumetric. No strain energy split will be applied.')
        
        if self.PFF_model not in ['AT1', 'AT2']:
            raise ValueError('PFF_model must be AT1 or AT2')

    # degradation function for Young's modulus and its derivative w.r.t. \alpha: g(\alpha) and g'(\alpha)
    def Edegrade(self, alpha):
        return (1 - alpha)**2, 2*(alpha - 1)

    # damage function and its derivative w.r.t. \alpha: w(\alpha) and w'(\alpha) and c_w
    def damageFun(self, alpha):
        if self.PFF_model == 'AT1':
            return alpha, 1.0, 8.0/3.0
        elif self.PFF_model == 'AT2':
            return alpha**2, 2*alpha, 2.0
    
    # Irreversibility penalty
    def irrPenalty(self):
        if self.PFF_model == 'AT1':
            return 27/64/self.tol_ir**2
        elif self.PFF_model == 'AT2':
            return 1.0/self.tol_ir**2-1.0