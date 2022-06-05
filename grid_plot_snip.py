def gridplotfit_v2(res_list,sym,save=False):
    plt.close()
    g0fac,alpha,T,fac = res_list
    dset_list = dset_list2
    allweightset = [total_weight_s5(dset,10**g0fac,alpha,T/100,10**fac,sym)[0] for dset in dset_list]
    scale = 4
    fig, (axes1, axes2, axes3, axes4, axes5, axes6) = plt.subplots(6,9,figsize=(9.5*scale,6*scale))

    for gamma, dset,axis1,axis2,axis3,axis4,axis5,axis6 in zip(allweightset,dset_list,axes1,axes2,axes3,axes4,axes5,axes6):
        expV, expC, expIR = expvals(dset, sym)
        theoC, theoIR = fitting_s5(gamma,10**g0fac,alpha,T/100,10**fac,expV)
        
        axis1.plot(expV, expC,label=f'data set = {dset}')
        axis1.plot(expV, theoC,label=f'$\Gamma_t = ${gamma}')
        axis1.legend()
        axis1.set_ylim(0.0,None)
        
        axis2.plot(expV, expIR,label=f'data set = {dset}')
        axis2.plot(expV, theoIR,label=f'$\Gamma_t = ${gamma}')
        axis2.legend()
#         axis1.set_ylim(0.0,None)
        
        v0 = expV*alpha
        nu = (1 - (v0/0.333))
        op1 = optical_func(gamma)
        s_mg = np.interp(nu, op1[:,0], op1[:,1])
        axis3.plot(expV, s_mg,label=f'$\Gamma_t = ${gamma}')
#         axis2.legend()
        axis3.set_ylim(0.0,2.05)
        
        n1 = n1_func(gamma)
        n2 = n2_func(gamma)
        n_g = np.interp(nu, n1[:,0], n1[:,1])
        n_e = np.interp(nu, n2[:,0], n2[:,1])

        axis4.plot(expV, n_g,label=f'$n_g$')
        axis4.plot(expV, n_e,label=f'$n_e$')
        axis4.legend()
        axis4.set_ylim(0.0,2.1)
        
        C_tot = fitting_s4(gamma,10**g0fac,alpha,T/100,10**fac,expV)
        C_qn = fitting_s3(gamma,alpha,T/100,10**fac,expV)
        C_q0 = fitting_s2(gamma,alpha,T/100,10**fac,expV)
        
        P0g = p0(s_mg*Delta, T/100, 10**fac)
        axis5.plot(expV,P0g,label=f'$\Gamma_t = ${gamma}')
        axis5.legend()
        
        axis6.plot(expV,C_tot,'k',label=r'$C_{total}$')
        axis6.plot(expV,C_q0,'--',label=r'$C_{q,g}$')
        axis6.plot(expV,C_qn - C_q0,'--',label=r'$C_{q,e}$')
        axis6.plot(expV,C_tot - C_qn,'--',label=r'$C_t$')
        axis6.legend()
        axis6.set_ylim(0.0,None)

    axes1[0].set_ylabel(r'Capacitance, C (fF)',fontsize=15)
    axes2[0].set_ylabel(r'Conductance, S ($10^{-8} \Omega^{-1}$)',fontsize=15)
    axes3[0].set_ylabel(r'Shiba mini-gap, $\delta / \Delta $',fontsize=15)
    axes4[0].set_ylabel(r'Occupation, $n$',fontsize=15)
    axes5[0].set_ylabel(r'P_{g,0}',fontsize=15)
    axes6[0].set_ylabel('Capacitance, C (fF)',fontsize=15)
    
    axes6[4].set_xlabel(r'$V_{P0} (mV) $',fontsize=20)
    
                   
    parameter_string = f"$T = {T*10:.3f} mK $\
            $log_{{10}} \\Gamma^0 = {g0fac:.3f}$\
            $\\alpha        = {alpha:.3f} $\
            $log_{{10}} N   = {fac:.3f}$"
    
    fig.text(
        0.5, 0.95, parameter_string, ha='center', va='center', wrap=False,
        bbox=dict(ec='black', alpha=1, color='azure'), fontsize=20)
    if sym:
        fig.text(
        0.5, 0.902, f"Stage 5 (Symmetrized) TW = {total_set_weight_s5(res_list,sym):.1f}", ha='center', va='center', wrap=False,
        bbox=dict(ec='black', alpha=0, color='azure'), fontsize=15)

    if save:
        if sym:
            plt.savefig(f"2D grid sym {[f'{member[0]}{member[1]:.2f}' for member in zip(['g0','a','T','N'],res_list)]}.pdf", format='pdf')
        else:
            plt.savefig(f"2D grid Not-sym {[f'{member:.2f}' for member in res_list]}.pdf", format='pdf')
      