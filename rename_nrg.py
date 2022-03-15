import os

abspath = 'G:\\My Drive\\GROWTH\\Quantum computing\\Majorana Project\\Delta_dependence\\'
for uvals in U_SET:
    for gammass in GAMMA_DICT[0]:
        pathstr = f'NRG\\U={uvals}\\Gamma={gammass}\\'
        oldnames = [abspath + pathstr + x for x in os.listdir(pathstr) if x[0]=='g']
        newnamess = [x.ljust(115, '0') for x in oldnames]
        for on, nn in zip(oldnames, newnamess):
            os.rename(on,nn)