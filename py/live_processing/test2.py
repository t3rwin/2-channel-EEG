import numpy as np
from numpy import load
import scipy
import csv
import pandas
import matplotlib
import matplotlib.pyplot as plt
import time
from scopetest import *
import concurrent.futures
import multiprocessing
from scipy.fft import fft,fftfreq
import math

NSAMP = 333
STOP = 9
fs = 333

def fft_data(q_in, q_out):
    data_buffer = []
    read_val = 0
    size = NSAMP#2 ** round(math.log(NSAMP,2))
    F = fs*np.array(fftfreq(NSAMP))
    alpha_index = [(np.abs(F - 8)).argmin(),(np.abs(F - 13)).argmin()]
    eeg_index = [(np.abs(F - 4)).argmin(),(np.abs(F - 30)).argmin()]

    i_begin = 0
    i_end = i_begin + NSAMP
    i_mid = int((NSAMP-1)/2)
    ffts = []
    while read_val != STOP:
        # t0 = time.time()
        read_val = q_in.get()
        while read_val != STOP:
            if len(data_buffer) < NSAMP:
                data_buffer.append(read_val)
            else:
                ffts = ffts + (np.abs(np.array(fft(data_buffer,size))))
                data_buffer = []
                power_block = np.square(ffts[i_begin:i_mid+1])
            read_val = q_in.get()

#---------------
        while (len(data_buffer) < NSAMP) and (read_val != STOP):
            data_buffer.append(read_val)
            read_val = q_in.get()

        power_block = np.square(np.abs(np.array(fft(data_buffer,size))))
    
        alpha_power = np.sum(power_block[alpha_index[0]:alpha_index[1]+1])
        tot_power = np.sum(power_block[eeg_index[0]:eeg_index[1]+1])

        q_out.put(alpha_power/tot_power)

        data_buffer = []
        
    if read_val == STOP:
        q_out.put([0]*NSAMP)