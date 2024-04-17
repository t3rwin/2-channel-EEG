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
from scipy.fft import fft

def draw_signals(q_in_filt,q_in_raw,fs,f_refresh): 
    def emitter3():
        samp_per_update = int(fs/f_refresh)
        out_raw = []
        out_filt = []

        done = False
        while not done:
            
            time.sleep(.002)
            if len(out_raw)<samp_per_update:
                
                if q_in_raw.empty() is False:
                    # r1 = q_in_raw.get()
                    out_raw.append(q_in_raw.get())
                else:
                    # r1 = 0
                    out_raw.append(0)
            if len(out_filt)<samp_per_update:
                if q_in_filt.empty() is False:
                # r2 = q_in_filt.get()
                    out_filt.append(q_in_filt.get())
                else:
                    # r2 = 0
                    out_filt.append(0)
            if (len(out_raw)==samp_per_update) and (len(out_filt)==samp_per_update):
                done = True
        yield [out_raw,out_filt]
        # yield [r1,r2]
    # print('a')
    # avc()

    fig, (ax1,ax2) = plt.subplots(2,1)
    fig.set_figheight(3)
    fig.set_figwidth(14)
    plt.grid()
    scope = Scope(ax1,ax2)
    ani = animation.FuncAnimation(fig, scope.update, frames=emitter3,
                                blit=True,cache_frame_data=False, save_count=0,interval=int(1000/f_refresh))
    plt.show()

fs = 333
window_time = 1
NSAMP = int(fs * window_time)
# if __name__ == '__main__':

data = pandas.read_csv('data_inputs/1open-4reading-300.csv')
volts = data['0.026325']
volts = volts[0:fs*5]

q_in_filt = multiprocessing.Queue() #raw volts
q_in_raw = multiprocessing.Queue()

for v in volts:
    q_in_filt.put(v)
    q_in_raw.put(v)

draw_signals(q_in_filt,q_in_raw,fs,60)