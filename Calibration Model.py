#!/usr/bin/env python
# coding: utf-8

#  # Calibration Model

# In[2]:


import numpy as np
import csv
import pandas as pd


# In[18]:


## whole program

def calibrate(LX0, LY0, LG0,
             KX0, KY0, KG0, T0,
             F0, E0, sig_G, sig_X, 
            sig_Y, sig_Z, sig_U,
             w, r, v,
             taxes):
    
    rho_g, rho_x, rho_y, rho_z, rho_u = rhos(sig_G, sig_X, sig_Y, sig_Z, sig_U)
    
    ## the above code calibrates the rho parameters from the input sigma parameters
    
    tl = taxes[0] ## extracts labor tax from input array of taxes
    tk = taxes [1] ## extracts capital tax from input array of taxes
    
    
    
    alg, akg = alpha_G(w, r, rho_g, LG0, KG0, tk, tl)
    
    
    alx, akx = alpha_X(w, r, rho_x, LX0, KX0, tk, tl)
    
    aky, aly, afy = alpha_Y(w, r, v, LY0, KY0, F0, rho_y, tk, tl)
    
    shares_g = [alg, akg]
    shares_x = [alx, akx]
    shares_y = [aly, aky, afy]
    
    ## the above code calibrates the share parameters for goods X, Y and G
    
    
    
    X0 = LX0 + KX0
    gamma_x = gamma_X(w, r, tl, tk, akx, alx, sig_X)
    
    G0 = LG0 + KG0
    gamma_g = gamma_G(w, r, tl, tk, akg, alg, sig_G)
    
    Y0 = LY0 + KY0 + F0
    gamma_y = gamma_Y(w, r, tl, tk, aky, aly, afy, sig_Y, v)
    
    ## the above code calibrates the gamma factors for goods X, Y and G
    
    
    
    p_X = px(gamma_x, sig_X, akx, alx, w, r, tk, tl)
    
    p_G = pg(gamma_g, sig_G, akg, alg, w, r, tk, tl)
    
    p_Y = py(gamma_y, sig_Y, aky, aly, afy, w, r, v, tk, tl)
    
    
    
    ## the above code calibrates the prices for goods X, Y, and G
    
    axz, ayz = alpha_Z(p_Y, p_X, X0, Y0, rho_z)
    
    shares_z = [axz, ayz]
    
    gamma_z = gamma_Z(axz, ayz, p_X, p_Y, sig_Z)
    
    p_Z = pz(gamma_z, sig_Z, axz, ayz, p_X, p_Y)
    
    ## the above code calculates the share parametes, gamma factor, and price for composite good Z
    prices = [p_X, p_G, p_Y, p_Z]
    azu, agu = alpha_U(p_G, p_Z, X0, Y0, G0, rho_u)
    
    share_parameters = [np.array([alg, akg]), np.array([alx, akx]), np.array([aly, aky, afy]),
                       np.array([axz, ayz]), np.array([azu, agu])]
    
    cal_prices = [w,r,v]
    
    sigmas = [sig_X, sig_G, sig_Y, sig_Z]
    gammas = [gamma_x, gamma_g, gamma_y, gamma_z]
    
    initials = [LX0, LY0, LG0,
                KX0, KY0, KG0, F0, E0, T0, X0, Y0, G0]
    
    b = E0/F0
    print(f'The emissions factor is  b = {b}')
    print(f"The share parameters for good G are a_lg = {alg} and a_kg = {akg}")
    print(f"The share parameters for good X are a_lx = {alx} and a_kx = {akx}")
    print(f"The share parameters for good Y are a_ly = {aly}, a_ky = {aky}, and a_fy = {afy}")
    print(f"The share parameters for composite good Z are a_xz = {axz}, a_yz = {ayz}")
    print(f"The share parameters for U are a_zu = {azu}, a_gu = {agu}")
    
    print(f"The prices are p_X = {p_X}, p_Y = {p_Y}, p_G = {p_G}, and p_Z = {p_Z}")
    print(f"The gammas are gamma_x = {gamma_x}, gamma_g = {gamma_g}, gamma_y = {gamma_y}, and gamma_z = {gamma_z}")
    
    data = share_parameters, prices, gammas, T0, taxes, rho_y, rho_x, rho_g, X0, b
    with open('data_for_sim.csv', 'w') as f:
        
        writer = csv.writer(f)
        writer.writerow(initials)
        writer.writerow(shares_x)
        writer.writerow(shares_g)
        writer.writerow(shares_y)
        writer.writerow(sigmas)
        writer.writerow(gammas)
        writer.writerow(taxes)
    
    with open('data_key.csv', 'w') as f:
        
        writer = csv.writer(f)
        writer.writerow(['LX0', 'LY0', 'LG0','KX0', 'KY0', 'KG0', 'F0', 'E0', 'T0',
                        'X0', 'Y0', 'G0'])
        writer.writerow(['alx', 'akx'])
        writer.writerow(['alg', 'akg'])
        writer.writerow(['aly', 'aky', 'afy'])
        writer.writerow(['sig_X', 'sig_G', 'sig_Y', 'sig_Z'])
        writer.writerow(['gamma_x', 'gamma_g', 'gamma_y', 'gamma_z'])
        
        writer.writerow(['t_l','t_k', 't_f'])
      
    return data
    
 
    
calibrate(100, 20, 60,
             50, 15, 20, -14.4048,
             5,79.5, 0.55, 0.65, 
            0.75, 0.85, 0.80,
             1, 1, 1,
             np.array([0.4,0.2,0]))


# In[19]:


test = pd.read_csv('data_for_sim.csv', header = None,dtype=np.float64)
print(test)
test2 = pd.read_csv('data_key.csv', header = None)
test2


# In[4]:


## calculating rhos from sigmas

def rhos(sig_G, sig_X, sig_Y, sig_Z, sig_U):
    rho_g = 1 - (1/sig_G) #calculates rho g from input sigma G
    rho_x = 1 - (1/sig_X) #calculates rho x from input sigma X
    rho_y = 1 - (1/sig_Y) #calculates rho y from input sigma Y
    rho_z = 1 - (1/sig_Z) #calculates rho z from input sigma Z
    rho_u = 1 - (1/sig_U) #calculates rho u from input sigma U
    
    return rho_g, rho_x, rho_y, rho_z, rho_u


# In[5]:


### Calculating share parameters for G



def alpha_G(w, r, rho_g, LG, KG, tk, tl):
    
    LG_q = LG/(1 + tl)
    KG_q = KG/(1 + tk)
    
    r_tax = r*(1 + tk)
    w_tax = w*(1 + tl)
    alg = 1/(1 + (r_tax/w_tax)*((LG_q/KG_q)**(rho_g - 1))) #calculates the share parameter a_lg from FoC
    akg = 1 - alg
    return alg,  akg

##Note: the double asterisk (**) denotes exponentiation

## alg is the share parameter for labor into good G
## akg is the share parameter for capital into good G


# In[6]:


### Calculating share parameters for X

def alpha_X(w, r, rho_x, LX, KX, tk, tl):
    
    LX_q = LX/(1 + tl)
    KX_q = KX/(1 + tk)
    
    r_tax = r*(1 + tk)
    w_tax = w*(1 + tl)
    alx = 1/(1 + (r_tax/w_tax)*((LX_q/KX_q)**(rho_x - 1))) #calculates the share parameter a_lx from FoC
    akx = 1 - alx
    return alx, akx


##Note: the double asterisk (**) denotes exponentiation

## alx is the share parameter for labor into good X
## akx is the share parameter for capital into good X


# In[7]:


### Calculating share parameters for Y
#0.3 kg/$ GDP https://data.worldbank.org/indicator/EN.ATM.CO2E.PP.GD
##GDP is 265 in test data
## F = 5
### Note that a value of eta  = 0.2 gives a values of alpha_EY ~ 0.12
def alpha_Y(r, w, v, LY, KY, FY, rho_y, tk, tl):
    r_tax = r*(1 + tk)
    w_tax = w*(1 + tl)
    LY_q = LY/(1 + tl)
    KY_q = KY/(1 + tk)
    
    a = r_tax/w_tax
    b = (LY_q/KY_q)**(rho_y - 1)
    c = (a*b)
    
    d = v/w_tax
    e = (LY_q/FY)**(rho_y - 1)
    f = (d*e)
    
    aly = (1 + c + f)**(-1)
    aky = aly*c
    afy = aly*f
    
    return aky, aly, afy

## aly is the share parameter for labor into good Y
## aky is the share parameter for capital into good Y
## aey is the share parameter for capital into good Y


##Note: the double asterisk (**) denotes exponentiation


# In[8]:


## calculating share parameters for Z

def alpha_Z(py, px, X0, Y0, rho_Z):
    
    
    axz = 1/(1 + (py/px)*((X0/Y0)**(rho_Z - 1)))
    ayz = 1 - axz
    
    return axz, ayz


## axz is the share parameter for X into good Z
## ayz is the share parameter for Y into good Y

##Note: the double asterisk (**) denotes exponentiation


# In[9]:


## calculating share parameters for U

def alpha_U(pg, pz, X0, Y0, G0, rho_U):
    Z0 = X0 + Y0
    azu = 1/(1 + (pg/pz)*((Z0/G0)**(rho_U - 1)))
    agu = 1 - azu
    
    return np.round(azu,4), np.round(agu, 4)


## azu is the share parameter for Z into U
## agu is the share parameter for G into U

##Note: the double asterisk (**) denotes exponentiation


# In[10]:


## calculating gamma factor X

def gamma_X(w, r, tl, tk, alpha_k, alpha_l, sigma):
    
    
    #LX_q = LX/(1 + tl)
    #KX_q = KX/(1 + tk)
    
    r_tax = r*(1 + tk)
    w_tax = w*(1 + tl)
    """num = w_tax*LX_q + r_tax*KX_q
    first_denom = (alpha_k**sigma)*(r_tax**(1-sigma))
    second_denom = (alpha_l**sigma)*(w_tax**(1-sigma))
    other_term = alpha_l*(LX_q**rho) + alpha_k*(KX_q**rho)
    full_denom = (other_term**(1/rho))*((first_denom + second_denom)**(1/(1-sigma)))
    gammax = (num/full_denom)**((sigma - 1)/(2*sigma - 1))"""

    first = (alpha_k**sigma)*(r_tax**(1-sigma)) + (alpha_l**sigma)*(w_tax**(1-sigma))
    inner = 1/ (first**(1/(1-sigma)))
    
    gammax = inner**((sigma - 1)/sigma)
    return gammax


# In[11]:


## calculating gamma factor G

def gamma_G(w, r, tl, tk, alpha_k, alpha_l, sigma):
    
    #LG_q = LG/(1 + tl)
    #KG_q = KG/(1 + tk)
    
    
    r_tax = r*(1 + tk)
    w_tax = w*(1 + tl)
    """num = w_tax*LG_q + r_tax*KG_q
    first_denom = (alpha_k**sigma)*(r_tax**(1-sigma))
    second_denom = (alpha_l**sigma)*(w_tax**(1-sigma))
    other_term = alpha_l*(LG_q**rho) + alpha_k*(KG_q**rho)
    full_denom = (other_term**(1/rho))*((first_denom + second_denom)**(1/(1-sigma)))
    gammag = (num/full_denom)**((sigma - 1)/(2*sigma - 1))"""
    first = (alpha_k**sigma)*(r_tax**(1-sigma)) + (alpha_l**sigma)*(w_tax**(1-sigma))
    inner = 1/ (first**(1/(1-sigma)))
    
    gammag = inner**((sigma - 1)/sigma)
    
    return gammag


# In[12]:


## calculating gamma factor Z

def gamma_Z(axz, ayz, px, py, sigma):
    
    
    first = (axz**sigma)*(px**(1-sigma)) + (ayz**sigma)*(py**(1-sigma))
    inner = 1/(first**(1/(1-sigma)))
    
    gammaz = inner**((sigma - 1)/sigma)
    
    return gammaz


# In[13]:


## calculating gamma factor Y

def gamma_Y(w, r, tl, tk, alpha_k, alpha_l, alpha_f, sigma, v):
    
    #LY_q = LY/(1 + tl)
    #KY_q = KY/(1 + tk)
    #Y0_q = LY_q + KY_q
    
    
    r_tax = r*(1 + tk)
    w_tax = w*(1 + tl)
    """num = w_tax*LY_q + r_tax*KY_q
    first_denom = (alpha_k**sigma)*(r_tax**(1-sigma))
    second_denom = (alpha_l**sigma)*(w_tax**(1-sigma))
    other_term = alpha_l*(LY_q**rho) + alpha_k*(KY_q**rho) #+  alpha_e*(79.5**rho)
    full_denom = (other_term**(1/rho))*((first_denom + second_denom)**(1/(1-sigma)))
    gammay = (num/full_denom)**((sigma - 1)/(2*sigma - 1))"""
    first = (alpha_k**sigma)*(r_tax**(1-sigma)) + (alpha_l**sigma)*(w_tax**(1-sigma)) + (alpha_f**sigma)*(v**(1-sigma))
    inner = 1/ (first**(1/(1-sigma)))
    
    gammay = inner**((sigma - 1)/sigma)
  
    return gammay




# In[14]:


## calculating price of good X

def px(gamma_x, sigma_x, akx, alx, w, r, tk, tl):
    r_tax = r*(1 + tk)
    w_tax = w*(1 + tl)
    
    
    first_term = gamma_x**(sigma_x/(sigma_x-1))
    second_term = akx**(sigma_x)*(r_tax**(1-sigma_x)) + alx**(sigma_x)*(w_tax**(1-sigma_x))
    p_x = first_term*(second_term**(1/(1-sigma_x)))
    
    return round(p_x, 4)
    


# In[15]:


## calculating price of good G

def pg(gamma_g, sigma_g, akg, alg, w, r, tk, tl):
    r_tax = r*(1 + tk)
    w_tax = w*(1 + tl)
    
    first_term = gamma_g**(sigma_g/(sigma_g-1))
    second_term = akg**(sigma_g)*(r_tax**(1-sigma_g)) + alg**(sigma_g)*(w_tax**(1-sigma_g))
    p_g = first_term*(second_term**(1/(1-sigma_g)))
    
    return round(p_g, 4)
    


# In[16]:


## calculating price of good Y

def py(gamma_y, sigma_y, aky, aly, afy, w, r, v, tk, tl):
    r_tax = r*(1 + tk)
    w_tax = w*(1 + tl)
    
    first_term = gamma_y**(sigma_y/(sigma_y-1))
    
    second_term_1 = aky**(sigma_y)*(r_tax**(1-sigma_y)) 
    second_term_2 = aly**(sigma_y)*(w_tax**(1-sigma_y)) 
    second_term_3 = afy**(sigma_y)*(v**(1-sigma_y))
    second_term = second_term_1 + second_term_2 + second_term_3
    p_y = first_term*(second_term**(1/(1-sigma_y)))
   
    return round(p_y, 4)



# In[17]:


## calculating price of composite good Z

def pz(gamma_z, sigma_z, axz, ayz, px, py):
    
    first_term = gamma_z**(sigma_z/(sigma_z-1))
    
    second_term = axz**(sigma_z)*(px**(1-sigma_z)) + ayz**(sigma_z)*(py**(1-sigma_z))
    
    p_z = first_term*(second_term**(1/(1-sigma_z)))
    #p_z = (second_term**(1/(1-sigma_z)))
    
    return round(p_z, 4)
    


# In[ ]:


##baseline emissins factor

b = 

