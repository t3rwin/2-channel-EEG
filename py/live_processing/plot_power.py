import numpy as np
from numpy import load
import scipy
import csv
import pandas
import matplotlib.pyplot as plt
import time
from scopetest import *


fs = 1000

with np.load('40hz_lowpass_fs1k.npz') as data:
    ba_lowpass = data['ba']

b_lowpass = []
for ba in ba_lowpass:
    b_lowpass.append(float(ba[0]))
a_lowpass = [0]*len(b_lowpass)
a_lowpass[0] = 1

# sos_lowpass = scipy.signal.tf2sos(b_lowpass,a_lowpass)
data = pandas.read_csv('data_inputs/1open-4reading.csv')
volts = data['Channel 1 (V)']
# volts_filt = scipy.signal.sosfilt(sos_lowpass,volts)
volts_filt = scipy.signal.lfilter(b_lowpass,a_lowpass,volts)

def filter(data,a,b,z):
    return scipy.signal.lfilter(b,a,data,zi=z)

def filter_bysample(data,a,b):
    result = []
    z = np.zeros(len(b)-1)
    for i, x in enumerate(data):
        # t0 = time.time()
        time.sleep(1e-3)
        val,z = filter([x],a,b,z)
        result.append(val)
        # t1 = time.time()
    # print(t1-t0)
    return result

def emitter():
    b = b_lowpass
    a = a_lowpass
    data = volts
    result = []
    z = np.zeros(len(b)-1)
    for i, x in enumerate(data):
        # t0 = time.time()
        # time.sleep(100e-3)
        val,z = filter([x],a,b,z)
        yield val
        # t1 = time.time()
    # print(t1-t0)
    

# ax1 = plt.subplot(211)
# # ax1.plot(volts[1:fs*2])
# plt.plot(volts_filt[0:fs*2])

# ax2 = plt.subplot(212, sharex=ax1)
# plt.plot(filt_live[0:fs*2])
# # plt.plot(volts_filt[1:fs*2])

# plt.show()

fig, ax = plt.subplots()
scope = Scope(ax)

# pass a generator in "emitter" to produce data for the update func
ani = animation.FuncAnimation(fig, scope.update, emitter,
                              blit=True, save_count=100,interval=1)

plt.show()