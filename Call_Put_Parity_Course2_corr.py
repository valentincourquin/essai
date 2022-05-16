# -*- coding: utf-8 -*-
"""
Call - Put Parity

@author: arthur.tardy@essca.eu
@corrections: miia.chabot@essca.fr

"""

#PACKAGES
import numpy as np
import scipy.stats as si
import sympy as sy
from sympy.stats import Normal, cdf
from sympy import init_printing
init_printing()

# CALL/PUT PARITY
    # Portefeuille A : une option d'achat européenne (call) et une trésorerie égale à Ke-rT (valeur actualisé au taux d'interet r, K= strike)
    # Portefeuille B : une option de vente européenne (put) et une action sous-jacente (SO)
    # Puisque les options sont européennes, elles ne peuvente être exercés avant la date d'échéance.
    # Les portefeuilles doivent par conséquent avoir la même valeur aujourd'hui, ce qui s'écrit :
        # Portefeuille A = Portefeuille B
        # c + Ke-rT = p + S0  (Call/Put Parity)
    # Si l'équation n'est pas vérifié, il y a alors des opportunités d'arbitrage
    
# 1) BS MODEL 
    # S = 40
    # K = 40
    # T = 1
    # r = 0.05
    # sigma = 0.4
        # On cherche c, p, Ke-rT (S0 = 40)
class cp :

    def call(S, K, T, r, sigma):
            d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma* np.sqrt(T))
            d2 = (np.log(S / K) + (r - 0.5 * sigma ** 2) * T) / (sigma* np.sqrt(T))
            call = (S * si.norm.cdf(d1, 0.0, 1.0) - K * np.exp(-r * T) * si.norm.cdf(d2, 0.0, 1.0))            
            return call
            # call = 7.209
           
            
    def put(S, K, T, r, sigma):
            d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
            d2 = (np.log(S / K) + (r - 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
            put = (K * np.exp(-r * T) * si.norm.cdf(-d2, 0.0, 1.0) - S * si.norm.cdf(-d1, 0.0, 1.0))
            return put
            # put = 5.258
            
    def k(S, K, T, r, sigma):
            k = K * np.exp(-r * T)    
            return k
            # k = 38.049
            
    def s(S, K, T, r, sigma):
            s=S
            return s
            # s = 40
        
print(call(40, 40, 1, 0.05, 0.4))
print (put(40, 40, 1, 0.05, 0.4))    
print(k(40, 40, 1, 0.05, 0.4))  
print(s(40, 40, 1, 0.05, 0.4))

# On vérife la parité call-put de BS   
A= (call(40, 40, 1, 0.05, 0.4) + k(40, 40, 1, 0.05, 0.4))
print(A)
# Portefeuille A = 45.258
B= (put(40, 40, 1, 0.05, 0.4) + s(40, 40, 1, 0.05, 0.4))
# Portefeuille B = 45.258
print(B)  
print(A-B)                   
round(A-B,3)

------------------------------------------------------------------------------

# CALL/PUT PARITY DIVIDEND MODEL
    # Portefeuille C : une option d'achat européenne et une trésorerie égale à Ke-rT (valeur actualisé du strike) 
    # Portefeuille D : une option de vente européenne et une action sous-jacente actualisée du dividende se-qt
    # Puisque les options sont européennes, elles ne peuvente être exercés avant la date d'échéance.
    # Les portefeuilles doivent par conséquent avoir la même valeur aujourd'hui, ce qui s'écrit :
        # Portefeuille C = Portefeuille D
        # c + Ke-rT = p + Se-qt  (Call/Put Parity)

# 2) BS MODEL Dividendes 
    # S = 40
    # K = 40
    # T = 1
    # r = 0.05
    # q = 0.1 (Dividende)
    # sigma = 0.4
        # On cherche c, p, Ke-rT, Se-qt, 
        
class cp2: 

    def call_div(S, K, T, r, q, sigma):            
        d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
        d2 = (np.log(S / K) + (r - 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))        
        call_div = (S * np.exp(-q * T) * si.norm.cdf(d1, 0.0, 1.0) - K * np.exp(-r * T) * si.norm.cdf(d2, 0.0, 1.0))        
        return call_div
        # call = 4.821     
    
    def put_div(S, K, T, r, q, sigma):        
        d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
        d2 = (np.log(S / K) + (r - 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))        
        put_div = (K * np.exp(-r * T) * si.norm.cdf(-d2, 0.0, 1.0) - S * np.exp(-q * T) * si.norm.cdf(-d1, 0.0, 1.0))        
        return put_div
        # put = 6.676    
    
    def k(S, K, T, r, q, sigma):        
        k = K * np.exp(-r * T)            
        return k
        # Ke-rt = 38.049
    
    def s1(S, K, T, r, q, sigma):        
        s1= S * np.exp(-q * T)            
        return s1
       # Se-qt = 36.193
    
print(call_div(40,40,1,0.05,0.1,0.4))
print(put_div(40,40,1,0.05,0.1,0.4))
print(k(40,40,1,0.05,0.1,0.4))
print(s1(40,40,1,0.05,0.1,0.4))
       
C= (call_div(40, 40, 1, 0.05, 0.1, 0.4) + k(40, 40, 1, 0.05, 0.1, 0.4))
print(C)
D= (put_div(40, 40, 1, 0.05, 0.1, 0.4) + s1(40, 40, 1, 0.05, 0.1, 0.4))
print(D)  
print(C-D)                   
round(C-D,3)

-------------------------------------------------------------------------------
# CALL/PUT PARITY CURRENCY MODEL GARMAN KOHLAGEN
    # A appliquer sur devises S = valeur du taux de change à échéance
    # Portefeuille E : une option d'achat européenne et une trésorerie égale à Ke-rT (valeur actualisé du strike) 
    # Portefeuille F : une option de vente européenne et une action sous-jacente actualisé du taux de rendement étranger Se-rft
    # Puisque les options sont européenes, elles ne peuvente être exercés avant la date d'échéance.
    # Les portefeuilles doivent par conséquent avoir la même valeur aujourd'hui, ce qui s'écrit :
        # Portefeuille E= Portefeuille F
        # c + Ke-rT = p + Se-rft  (Call/Put Parity)

# 3) GARMAN KOHLAGEN 
    # S = 1.22
    # K = 1.23
    # T = 1
    # r = 0.05
    # rf = 0.1 (risk free-rate étranger)
    # sigma = 0.4
        # On cherche c, p, Ke-rT, Se-rft,   
       
class cp3:
    def currency_call(s, k, r, rf, sigma, t): 
        d1 = (np.log(s / k) + t * (r - rf + sigma ** 2 / 2)) / (sigma * np.sqrt(t))
        d2 = d1 - sigma * np.sqrt(t)     
        currency_call = s * np.exp(-rf * t) *si.norm.cdf(d1) - k * np.exp(- r * t) *si.norm.cdf(d2)     
        return currency_call
        # le call vaut 0.149 par unité de la devise étrangère
    
    def currency_put(s, k, r, rf, sigma, t):
        d1 = (np.log(s / k) + t * (r - rf + sigma ** 2 / 2)) / (sigma * np.sqrt(t))
        d2 = d1 - sigma* np.sqrt(t)
        currency_put = k * np.exp(-r * t) * si.norm.cdf(-d2) - s * np.exp(-rf * t) * si.norm.cdf(-d1)
        return currency_put
        # le put vaut 0.051 par unité de devise étrangère
    
    def k(s, k, r, rf, sigma, t):        
        k = k * np.exp(-r * t)            
        return k
    
    def s2(s, k, r, rf, sigma, t):        
        s2= s * np.exp(-rf * t)            
        return s2
    
print(currency_call(1.22, 1.23, 0.05, 0.10, 0.4, 1))  
print(currency_put(1.22, 1.23, 0.05, 0.05, 0.10, 1))
print(k(1.22, 1.23, 0.05, 0.10, 0.4, 1)) 
print(s2(1.22, 1.23, 0.05, 0.10, 0.4, 1))      

E= (currency_call(1.22, 1.23, 0.05, 0.10, 0.4, 1) + k(1.22, 1.23, 0.05, 0.10, 0.4, 1))
print(E)
F= (currency_put(1.22, 1.23, 0.05, 0.10, 0.4, 1) + s2(1.22, 1.23, 0.05, 0.10, 0.4, 1))
print(F)  
print(E-F)                   
round(E-F,3)
#TEST
