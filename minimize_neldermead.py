@jit(nopython=True,nogil=True)
def minimize_neldermead(func, x0):
    maxiter=800
    maxfev=800
    xatol=1e-4
    fatol=1e-4
    unknown_options = [0.0]

    maxfun = maxfev
    fcalls = [0]
    rho = 1
    chi = 2
    psi = 0.5
    sigma = 0.5
    nonzdelt = 0.05
    zdelt = 0.00025
    x0 = np.asfarray(x0).flatten()
    N = len(x0)
    sim = np.empty((N + 1, N))
    sim[0] = x0
    
    for k in range(N):
        y = x0.copy()
        if y[k] != 0:
            y[k] = (1 + nonzdelt)*y[k]
        else:
            y[k] = zdelt
        sim[k + 1] = y

    one2np1 = list(range(1, N + 1))
    fsim = np.empty((N + 1,))

    for k in range(N + 1):
        fsim[k] = func(sim[k])

    ind = np.argsort(fsim)
    fsim = fsim[ind] 
    sim = sim[ind]

#     iterations = 1

#     while (fcalls[0] < maxfun and iterations < maxiter):
    for iterations in range(maxiter):
        if (np.max(np.ravel(np.abs(sim[1:] - sim[0]))) <= xatol and
                np.max(np.abs(fsim[0] - fsim[1:])) <= fatol):
            break

        xbar = np.sum(sim[:-1], 0) / N
        xr = (1 + rho) * xbar - rho * sim[-1]
        
        fxr = func(xr)
        doshrink = 0

        if fxr < fsim[0]:
            xe = (1 + rho * chi) * xbar - rho * chi * sim[-1]
            
            fxe = func(xe)

            if fxe < fxr:
                sim[-1] = xe
                fsim[-1] = fxe
            else:
                sim[-1] = xr
                fsim[-1] = fxr
        else:  # fsim[0] <= fxr
            if fxr < fsim[-2]:
                sim[-1] = xr
                fsim[-1] = fxr
            else:  # fxr >= fsim[-2]
                # Perform contraction
                if fxr < fsim[-1]:
                    xc = (1 + psi * rho) * xbar - psi * rho * sim[-1]
                    
                    fxc = func(xc)

                    if fxc <= fxr:
                        sim[-1] = xc
                        fsim[-1] = fxc
                    else:
                        doshrink = 1
                else:
                    # Perform an inside contraction
                    xcc = (1 - psi) * xbar + psi * sim[-1]
                    
                    fxcc = func(xcc)

                    if fxcc < fsim[-1]:
                        sim[-1] = xcc
                        fsim[-1] = fxcc
                    else:
                        doshrink = 1

                if doshrink:
                    for j in one2np1:
                        sim[j] = sim[0] + sigma * (sim[j] - sim[0])
                        fsim[j] = func(sim[j])

        ind = np.argsort(fsim)
        sim = sim[ind] #np.take(sim, ind, 0)
        fsim = fsim[ind] #np.take(fsim, ind, 0)
        
#         iterations += 1

    x = sim[0]
    fval = np.min(fsim)

    return x, fval