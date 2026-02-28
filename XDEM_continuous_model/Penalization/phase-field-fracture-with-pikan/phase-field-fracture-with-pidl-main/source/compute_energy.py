import torch
import torch.nn as nn

# Computes the total strain energy, damage energy and irreversibility penalty
def compute_energy(inp, u, v, alpha, hist_alpha, matprop, pffmodel, area_elem, T_conn=None):
    E_el, E_d, E_hist_penalty = compute_energy_per_elem(inp, u, v, alpha, hist_alpha, matprop, pffmodel, area_elem, T_conn)
    E_el_sum = torch.sum(E_el)
    E_d_sum = torch.sum(E_d)
    E_hist_sum = torch.sum(E_hist_penalty)

    return E_el_sum, E_d_sum, E_hist_sum


def compute_energy_per_elem(inp, u, v, alpha, hist_alpha, matprop, pffmodel, area_elem, T_conn=None):
    '''
    Computes the energies in each element.
    T_conn = None: an indicator that the input points are the Gauss points of elements and 
    autodiff should be used for gradient computation
    '''

    strain_11, strain_22, strain_12, grad_alpha_x, grad_alpha_y = gradients(inp, u, v, alpha, area_elem, T_conn)

    if T_conn == None:
        alpha_elem = alpha
    else:
        alpha_elem = (alpha[T_conn[:, 0]] + alpha[T_conn[:, 1]] + alpha[T_conn[:, 2]])/3
    
    damageFn, _, c_w = pffmodel.damageFun(alpha_elem)
    weight_penalty = pffmodel.irrPenalty()

    E_el_elem, _ = strain_energy_with_split(strain_11, strain_22, strain_12, alpha_elem, matprop, pffmodel)
    E_el = area_elem*E_el_elem
    E_d = (matprop.w1/c_w*(damageFn + matprop.l0**2*(grad_alpha_x**2+grad_alpha_y**2)))*area_elem

    dAlpha = alpha - hist_alpha
    if T_conn == None:
        dAlpha_elem = dAlpha
    else:
        dAlpha_elem = (dAlpha[T_conn[:, 0]] + dAlpha[T_conn[:, 1]] + dAlpha[T_conn[:, 2]])/3
    hist_penalty = nn.ReLU()(-dAlpha_elem)
    E_hist_penalty = 0.5*matprop.w1*weight_penalty*hist_penalty**2 * area_elem

    return E_el, E_d, E_hist_penalty


# Computes the components of strain and gradients of alpha
def gradients(inp, u, v, alpha, area_elem, T_conn = None):
    grad_u_x, grad_u_y = field_grads(inp, u, area_elem, T_conn)
    grad_v_x, grad_v_y = field_grads(inp, v, area_elem, T_conn)
    grad_alpha_x, grad_alpha_y = field_grads(inp, alpha, area_elem, T_conn)

    strain_11 = grad_u_x
    strain_22 = grad_v_y
    strain_12 = 0.5*(grad_u_y + grad_v_x)

    return strain_11, strain_22, strain_12, grad_alpha_x, grad_alpha_y
    

# Computes the gradient of fields using the shape functions of a triangular element in FEA
def field_grads(inp, field, area_elem, T = None):
    if T == None:
        grad_field = torch.autograd.grad(field.sum(), inp, create_graph=True)[0]
        grad_x = grad_field[:, 0]
        grad_y = grad_field[:, 1]
    else:
        grad_x = (inp[T[:, 1], -1]-inp[T[:, 2], -1])*field[T[:, 0]] + \
                (inp[T[:, 2], -1]-inp[T[:, 0], -1])*field[T[:, 1]] + \
                (inp[T[:, 0], -1]-inp[T[:, 1], -1])*field[T[:, 2]]
        grad_y = (inp[T[:, 2], -2]-inp[T[:, 1], -2])*field[T[:, 0]] + \
                (inp[T[:, 0], -2]-inp[T[:, 2], -2])*field[T[:, 1]] + \
                (inp[T[:, 1], -2]-inp[T[:, 0], -2])*field[T[:, 2]]
        grad_x = grad_x/area_elem/2
        grad_y = grad_y/area_elem/2

    return grad_x, grad_y


# Computes the element-wise strain energy density after applying the prescribed split
def strain_energy_with_split(strain_11, strain_22, strain_12, alpha, matprop, pffmodel):
    fun_EDegrade, _ = pffmodel.Edegrade(alpha)

    if pffmodel.se_split == 'volumetric':
        mat_K = matprop.mat_lmbda + 2.0/3.0*matprop.mat_mu
        strain_k = (strain_11+strain_22)/3.0
        strain_deviatoric_11 = strain_11 - strain_k
        strain_deviatoric_22 = strain_22 - strain_k
        strain_deviatoric_33 =  0 - strain_k
        E_elV_p = 0.5*mat_K*(nn.ReLU()(3.0*strain_k))**2
        E_elV_n = 0.5*mat_K*(-nn.ReLU()(-3.0*strain_k))**2

        E_el_dev = matprop.mat_mu*(strain_deviatoric_11**2+strain_deviatoric_22**2+strain_deviatoric_33**2+2*strain_12**2)
        E_el_p = E_elV_p + E_el_dev
        E_el = fun_EDegrade*(E_el_p) + E_elV_n

    else:
        E_el = fun_EDegrade*(0.5*matprop.mat_lmbda*(strain_11+strain_22)**2 + matprop.mat_mu*(strain_11**2+strain_22**2+2*strain_12**2))
        E_el_p = E_el

    return E_el, E_el_p


# Computes stress in each element
def stress(strain_11, strain_22, strain_12, alpha, matprop, pffmodel):
    fun_EDegrade, _ = pffmodel.Edegrade(alpha)

    if pffmodel.se_split == 'volumetric':
        mat_K = matprop.mat_lmbda + 2.0/3.0*matprop.mat_mu
        strain_k = (strain_11+strain_22)/3.0
        strain_deviatoric_11 = strain_11 - strain_k
        strain_deviatoric_22 = strain_22 - strain_k
        stress_11 = fun_EDegrade*(mat_K*(nn.ReLU()(3.0*strain_k)) + 2*matprop.mat_mu*strain_deviatoric_11) + mat_K*(-nn.ReLU()(-3.0*strain_k))
        stress_22 = fun_EDegrade*(mat_K*(nn.ReLU()(3.0*strain_k)) + 2*matprop.mat_mu*strain_deviatoric_22) + mat_K*(-nn.ReLU()(-3.0*strain_k))
        stress_12 = fun_EDegrade*(2*matprop.mat_mu*strain_12)

    else:
        stress_11 = fun_EDegrade*(matprop.mat_lmbda*(strain_11+strain_22) + 2*matprop.mat_mu*strain_11)
        stress_22 = fun_EDegrade*(matprop.mat_lmbda*(strain_11+strain_22) + 2*matprop.mat_mu*strain_22)
        stress_12 = fun_EDegrade*(2*matprop.mat_mu*strain_12)

    return stress_11, stress_22, stress_12
