import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import scale
from sklearn.decomposition import PCA
import os
import math
import sys

# See https://gaussian.com/uvvisplot/
#
#          sqrt(pi) * sqr(e) * N
# C1 = -----------------------------
#       1000 * ln(10) * sqr(c) * m_e
#
# pi   is Pi
# e    is the Euler's number
# N    is Avogadro's number
# c    is the speed of light
# m_e  is the mass of an electron
C1 = 1.3062974E8

# E  = hc/λ
# Eλ = hc
# h    = 6.626E-34 Js
# c    = 2.99E8    m/s
# 1 J  = 1.60E-19  eV
# 1 m  = 1.00E9    nm
#
# Js * m/s * eV/J * nm/m = eV*nm
eV_dot_nm = 1239.816088
# Log file Intermediate Representation
class LogIR:

    VISIBLE_MIN = 380
    VISIBLE_MAX = 700

    def __init__(self,path, nms, fs, name=None):
        if len(nms) != len(fs):
            fatal(
                "cannot create LogIR with mismatching number of wavelengths and oscillator strengths"
            )
        self.path = path
        self.nms = nms
        self.fs = fs

        if name is None:
            file = os.path.basename(self.path)
            name, _ = os.path.splitext(file)

        self.name = name

    def transform(self, x, sigma):
        out = 0
        
        for nm, f in zip(self.nms, self.fs):
            wv_nm = eV_dot_nm / sigma
            c2 = C1 * (f / (1E7 / wv_nm))
            
            y = 1 / x
            y = y - (1 / nm)
            y = y / (1 / wv_nm)
            y = y * y
            y = -y
            y = math.exp(y)
            y = c2 * y
            out = out + y
            
        return out

    def plot_band(self, start=0.1, end=1000, sigma=0.4, ax = None, c = 'b'):
        xs = np.linspace(start, end, num=100)
        ys = [self.transform(x, sigma) for x in xs]
        
        if ax: 
            ax.plot(xs, ys, label=self.name, color = c)
        else: 
            plt.plot(xs, ys, label = self.name, color = c)
        
    def get_data(self, start = 0.1, end = 1000, sigma = 0.4): 
        xs = np.linspace(start, end, num=100)
        ys = [self.transform(x, sigma) for x in xs]
        return [xs, ys]

    def plot_stem(self):
        markers, stems, bases = plt.stem(self.nms,
                                         self.fs,
                                         linefmt='-',
                                         markerfmt=' ',
                                         basefmt=':',
                                         label=self.name)
        plt.setp(stems, 'color', plt.getp(markers, 'color'))
        plt.setp(bases, 'color', 'grey')

    def filter_visible(self, lowest=5):
        nms = []
        fs = []
        count = 0
        for nm, f in zip(self.nms, self.fs):
            if LogIR.VISIBLE_MIN <= nm <= LogIR.VISIBLE_MAX:
                nms.append(nm)
                fs.append(f)
                count += 1

            if count >= lowest:
                break

        self.nms = nms
        self.fs = fs
        
def plot_hist(values, bins, xlabel, ylabel, title, xmin = None, xmax = None): 
    values = np.array(values).flatten()
    plt.hist(values, bins = bins)
    
    #plot mean
    plt.axvline(values.mean(), color='k', linestyle='dashed', linewidth=1,
                label='Mean {:.2f}'.format(values.mean()))
    min_ylim, max_ylim = plt.ylim()
    
    #put std in the legend
    plt.axvline(values.std(), color='b', linestyle='dashed', linewidth=1, 
                label='Std {:.2f}'.format(values.std()))
    
    #plot median 
    plt.axvline(np.median(values), color='r', linestyle='dashed', linewidth=1, 
                label='Median {:.2f}'.format(np.median(values)))
    
    if xmin and xmax: 
        plt.xlim([xmin, xmax])
    else: 
        plt.xlim([values.min(), values.max()])
    
    plt.legend(loc = 0)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.show()