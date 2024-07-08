import numpy as np
from numpy import sqrt
from scipy.linalg import cholesky


def solve_ARC_subproblem(solver, grad, Hv, hessian, sigma, w, 
                         successful_flag,lambda_k,exact_tol,krylov_tol,solve_each_i_th_krylov_space, keep_Q_matrix_in_memory, **kwargs):
    
    if solver == 'cauchy_point':
        (s, lambda_k) = solve_ARC_subproblem_cauchy_pt(grad, Hv, sigma)
    
    elif solver == 'exact':
        (s, lambda_k) = exact_ARC_suproblem_solver(grad,hessian,sigma, 
                                                  eps_exact=exact_tol,successful_flag=successful_flag,lambda_k=lambda_k)
        
    
    elif solver == 'lanczos':
 
        y=grad
        grad_norm=np.linalg.norm(grad)
        gamma_k_next=grad_norm
        delta=[] 
        gamma=[] # save for cheaper reconstruction of Q

        dimensionality = len(w)
        if keep_Q_matrix_in_memory: 
            q_list=[]    

        k=0
        T = np.zeros((1, 1)) #Building up tri-diagonal matrix T

        while True:
            if gamma_k_next==0: #From T 7.5.16 u_k was the minimizer of m_k. But it was not accepted. Thus we have to be in the hard case.
                H =hessian(w)
    #             H =hessian(w, X, Y, **kwargs)
                (s, lambda_k) = exact_ARC_suproblem_solver(grad,H, sigma, exact_tol,successful_flag,lambda_k)
                return (s,lambda_k)

            #a) create g
            e_1=np.zeros(k+1)
            e_1[0]=1.0
            g_lanczos=grad_norm*e_1
            #b) generate H
            gamma_k = gamma_k_next
            gamma.append(gamma_k)

            if not k==0:
                q_old=q
            q=y/gamma_k

            if keep_Q_matrix_in_memory:
                q_list.append(q)    

            Hq=Hv(q)
    #         Hq=Hv(w, X, Y, q, **kwargs) #matrix free            
            delta_k=np.dot(q,Hq)
            delta.append(delta_k)
            T_new = np.zeros((k + 1, k + 1))
            if k==0:
                T[k,k]=delta_k
                y=Hq-delta_k*q
            else:
                T_new[0:k,0:k]=T
                T_new[k, k] = delta_k
                T_new[k - 1, k] = gamma_k
                T_new[k, k - 1] = gamma_k
                T = T_new
                y=Hq-delta_k*q-gamma_k*q_old

            gamma_k_next=np.linalg.norm(y)
            #### Solve Subproblem only in each i-th Krylov space          
            if k %(solve_each_i_th_krylov_space) ==0 or (k==dimensionality-1) or gamma_k_next==0:
                (u,lambda_k) = exact_ARC_suproblem_solver(g_lanczos,T, sigma, exact_tol,successful_flag,lambda_k)
                e_k=np.zeros(k+1)
                e_k[k]=1.0
                if np.linalg.norm(y)*abs(np.dot(u,e_k))< min(krylov_tol,np.linalg.norm(u)/max(1, sigma))*grad_norm:
                    break

            if k==dimensionality-1: 
                print ('Krylov dimensionality reach full space!')
                break      

            successful_flag=False     


            k=k+1

        # Recover Q to compute s
        n=np.size(grad) 
        Q=np.zeros((k + 1,n))  #<--------- since numpy is ROW MAJOR its faster to fill the transpose of Q
        y=grad

        for j in range (0,k+1):
            if keep_Q_matrix_in_memory:
                Q[j,:]=q_list[j]
            else:
                if not j==0:
                    q_re_old=q_re
                q_re=y/gamma[j]
                Q[:,j]=q_re

                Hq=Hv(q_re)
    #             Hq=Hv(w, X, Y, q_re, **kwargs) #matrix free

                if j==0:
                    y=Hq-delta[j]*q_re
                elif not j==k:
                    y=Hq-delta[j]*q_re-gamma[j]*q_re_old

        s=np.dot(u,Q)
        del Q
        
    else:
        raise ValueError('solver unknown')
        
        
    return (s,lambda_k)


def exact_ARC_suproblem_solver(grad,H,sigma, eps_exact,successful_flag,lambda_k):
    from scipy import linalg
    s = np.zeros_like(grad)

    #a) EV Bounds
    gershgorin_l=min([H[i, i] - np.sum(np.abs(H[i, :])) + np.abs(H[i, i]) for i in range(len(H))]) 
    gershgorin_u=max([H[i, i] + np.sum(np.abs(H[i, :])) - np.abs(H[i, i]) for i in range(len(H))]) 
    H_ii_min=min(np.diagonal(H))
    H_max_norm=sqrt(H.shape[0]**2)*np.absolute(H).max() 
    H_fro_norm=np.linalg.norm(H,'fro') 

    #b) solve quadratic equation that comes from combining rayleigh coefficients
    (lambda_l1,lambda_u1)=mitternachtsformel(1,gershgorin_l,-np.linalg.norm(grad)*sigma)
    (lambda_u2,lambda_l2)=mitternachtsformel(1,gershgorin_u,-np.linalg.norm(grad)*sigma)
    
    lambda_lower=max(0,-H_ii_min,lambda_l2)  
    lambda_upper=max(0,lambda_u1)            #0's should not be necessary


    if successful_flag==False and lambda_lower <= lambda_k <= lambda_upper: #reinitialize at previous lambda in case of unscuccesful iterations
        lambda_j=lambda_k
    else:
        lambda_j=np.random.uniform(lambda_lower, lambda_upper)

    no_of_calls=0 
    for v in range(0,50):
                
        no_of_calls+=1
        lambda_plus_in_N=False
        lambda_in_N=False

        B = H + lambda_j * np.eye(H.shape[0], H.shape[1])
        
        if lambda_lower==lambda_upper==0 or np.any(grad)==0:
            lambda_in_N=True
        else:
            try: # if this succeeds lambda is in L or G.
                # 1 Factorize B
                L = np.linalg.cholesky(B)
                # 2 Solve LL^Ts=-g
                Li = np.linalg.inv(L)
                s = - np.dot(np.dot(Li.T, Li), grad)
                sn = np.linalg.norm(s)
               
                ## 2.1 Terminate <- maybe more elaborated check possible as Conn L 7.3.5 ??? 
                phi_lambda=1./sn -sigma/lambda_j
                if (abs(phi_lambda)<=eps_exact): #
#                     print('checkpoint 1')
                    break
                # 3 Solve Lw=s
                w = np.dot(Li, s)
                wn = np.linalg.norm(w)

                
                ## Step 1: Lambda in L and thus lambda+ in L
                if phi_lambda < 0: 
                    #print ('lambda: ',lambda_j, ' in L')
                    c_lo,c_hi= mitternachtsformel((wn**2/sn**3),1./sn+(wn**2/sn**3)*lambda_j,1./sn*lambda_j-sigma)
                    lambda_plus=lambda_j+c_hi
                    #lambda_plus = lambda_j-(1/sn-sigma/lambda_j)/(wn**2*sn**(-3)+sigma/lambda_j**2) #ARC gives other formulation of same update, faster?

                    lambda_j = lambda_plus
                    
    
                ## Step 2: Lambda in G, hard case possible
                elif phi_lambda>0:
                    #print ('lambda: ',lambda_j, ' in G')
                    #lambda_plus = lambda_j-(1/sn-sigma/lambda_j)/(wn**2*sn**(-3)+sigma/lambda_j**2) #ARC gives other formulation of same update, faster?
                    lambda_upper=lambda_j
                    _lo,c_hi= mitternachtsformel((wn**2/sn**3),1./sn+(wn**2/sn**3)*lambda_j,1./sn*lambda_j-sigma)
                    lambda_plus=lambda_j+c_hi
                    ##Step 2a: If lambda_plus positive factorization succeeds: lambda+ in L (right of -lambda_1 and phi(lambda+) always <0) -> hard case impossible
                    if lambda_plus >0:
                        try:
                            #1 Factorize B
                            B_plus = H + lambda_plus*np.eye(H.shape[0], H.shape[1])
                            L = np.linalg.cholesky(B_plus)
                            lambda_j=lambda_plus
                            #print ('lambda+', lambda_plus, 'in L')
                        except np.linalg.LinAlgError: 
                            lambda_plus_in_N=True
                    
                    ##Step 2b/c: else lambda+ in N, hard case possible
                    if lambda_plus <=0 or lambda_plus_in_N==True:
                        #print ('lambda_plus', lambda_plus, 'in N')
                        lambda_lower=max(lambda_lower,lambda_plus) #reset lower safeguard
                        lambda_j=max(sqrt(lambda_lower*lambda_upper),lambda_lower+0.01*(lambda_upper-lambda_lower))  

                        lambda_lower=np.float32(lambda_lower)
                        lambda_upper=np.float32(lambda_upper)
                        if lambda_lower==lambda_upper:
                                lambda_j = lambda_lower #should be redundant?
                                ew, ev = linalg.eigh(H, eigvals=(0, 0))
                                d = ev[:, 0]
                                dn = np.linalg.norm(d)
                                #note that we would have to recompute s with lambda_j but as lambda_j=-lambda_1 the cholesk facto may fall. lambda_j-1 should only be digits away!
                                tao_lower, tao_upper = mitternachtsformel(1, 2*np.dot(s,d), np.dot(s,s)-lambda_j**2/sigma**2)
                                s = s + tao_lower * d # both, tao_l and tao_up should give a model minimizer!
                                print ('hard case resolved') 
                                break
                    #else: #this only happens for Hg=0 -> s=(0,..,0)->phi=inf -> lambda_plus=nan -> hard case (e.g. at saddle) 
                     #   lambda_in_N = True
                ##Step 3: Lambda in N
            except np.linalg.LinAlgError:
                lambda_in_N = True
        if lambda_in_N == True:
            #print ('lambda: ',lambda_j, ' in N')
            lambda_lower = max(lambda_lower, lambda_j)  # reset lower safeguard
            lambda_j = max(sqrt(lambda_lower * lambda_upper), lambda_lower + 0.01 * (lambda_upper - lambda_lower))  # eq 7.3.1
            #Check Hardcase
            #if (lambda_upper -1e-4 <= lambda_lower <= lambda_upper +1e-4):
            lambda_lower=np.float32(lambda_lower)
            lambda_upper=np.float32(lambda_upper)

            if lambda_lower==lambda_upper:
                lambda_j = lambda_lower #should be redundant?
                ew, ev = linalg.eigh(H, eigvals=(0, 0))
                d = ev[:, 0]
                dn = np.linalg.norm(d)
                if ew >=0: #H is pd and lambda_u=lambda_l=lambda_j=0 (as g=(0,..,0)) So we are done. returns s=(0,..,0)
                    break
                #note that we would have to recompute s with lambda_j but as lambda_j=-lambda_1 the cholesk.fact. may fail. lambda_j-1 should only be digits away!
                sn= np.linalg.norm(s)
                tao_lower, tao_upper = mitternachtsformel(1, 2*np.dot(s,d), np.dot(s,s)-lambda_j**2/sigma**2)
                s = s + tao_lower * d 
                print ('hard case resolved') 
                break 

    return s,lambda_j


def solve_ARC_subproblem_cauchy_pt(grad,Hv,sigma):
    #min m(-a*grad) leads to finding the root of a quadratic polynominal

    Hg=Hv(grad)
    gHg=np.dot(grad,Hg)
    a=sigma*np.linalg.norm(grad)**3
    b=gHg
    c=-np.dot(grad,grad)
    (alpha_l,alpha_h)=mitternachtsformel(a,b,c)
    alpha=alpha_h
    s=-alpha*grad

    return (s,0)

############################
### Auxiliary Functions ###
############################
def mitternachtsformel(a,b,c):
    sqrt_discriminant = sqrt(b * b - 4 * a * c)
    t_lower = (-b - sqrt_discriminant) / (2 * a)
    t_upper = (-b + sqrt_discriminant) / (2 * a)
    return t_lower, t_upper