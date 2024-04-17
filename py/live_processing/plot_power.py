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

def send_data(volts,q_volts,q_v):
    for v in volts:
        q_volts.put(v)
        q_v.put(v)
        # print(f'(v:{v})')
        time.sleep(.001)
    # while q_volts.empty() is False:
    #     q_volts.put(0)
    #     q_v.put(0)
    q_volts.put(STOP)
    q_v.put(STOP)

def filter(data,a,b,z):
    return scipy.signal.lfilter(b,a,data,zi=z)

def filt_data(a,b,q_volts,q_volts_filt,q_filt_for_fft):
    result = []
    z = [0]*(len(b)-1)
    x = 0
    while x != STOP:
    # while q_volts.empty() is False:
        # t0 = time.time()
        # if q_volts.empty() is False:
        val,z = filter([x],a,b,z)
        result.append(val)
        q_volts_filt.put(val[0])
        q_filt_for_fft.put(val[0])
            # print(f'(v_filt:{val})')
        # q_volts_filt.put(x)
        x = q_volts.get()
    q_volts_filt.put(STOP)
    q_filt_for_fft.put(STOP)

        # t1 = time.time()

def fft_data(q_in, q_out):
    data_buffer = []
    read_val = 0
    size = 2 ** round(math.log(NSAMP,2))
    F = fs*np.array(fftfreq(NSAMP))
    alpha_index = [(np.abs(F - 8)).argmin(),(np.abs(F - 13)).argmin()]
    eeg_index = [(np.abs(F - 4)).argmin(),(np.abs(F - 30)).argmin()]

    while read_val != STOP:
        # t0 = time.time()
        read_val = q_in.get()
        while (len(data_buffer) < NSAMP) and (read_val != STOP):
            # if read_val != STOP:
            data_buffer.append(read_val)
            read_val = q_in.get()
            # else:
            #     data_buffer.append(0)

        power_block = np.square(np.abs(np.array(fft(data_buffer,size))))
        # fft_vals = np.array(fft(data_buffer,size))
        # fft_block = np.array(fft_vals[0:int((len(fft_vals)+1)/2)])
        # power_block = np.square(np.abs(fft_vals))
    
        alpha_power = np.sum(power_block[alpha_index[0]:alpha_index[1]+1])
        tot_power = np.sum(power_block[eeg_index[0]:eeg_index[1]+1])
        # a = alpha_power/tot_power
        # print(a)
        q_out.put(alpha_power/tot_power)
        # df = pandas.DataFrame([alpha_power/tot_power])
        # df.to_csv('outputs/power.csv')
        # t1=time.time()
        # print(len(data_buffer))
        data_buffer = []
        # print(t1-t0)
        
    if read_val == STOP:
        q_out.put([0]*NSAMP)



# q_v is queue for plotting raw volts
def draw_signals(q_in_filt,q_in_raw,fs,f_refresh): 
    def emitter3():
        samp_per_update = int(fs/f_refresh)
        out_raw = []
        out_filt = []

        done = False
        while not done:
            if len(out_raw)<samp_per_update:
                while q_in_raw.empty() is True:
                    pass
                out_raw.append(q_in_raw.get())
                # if q_in_raw.empty() is False:
                    # out_raw.append(q_in_raw.get())
                # else:
                    # out_raw.append(0)
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

    # fig, (ax1,ax2) = plt.subplots(2,1)
    fig, ax1 = plt.subplots()
    fig.set_figheight(3)
    fig.set_figwidth(14)
    plt.grid()
    scope = Scope(ax1)
    ani = animation.FuncAnimation(fig, scope.update, frames=emitter3,
                                blit=True,cache_frame_data=False, save_count=0,interval=int(1000/f_refresh))
    plt.show()

def draw_fft(q_fft):
    def emittera():
        while q_fft.empty() is True:
            pass
        # if q_fft.empty() is False:
        #     fft_frame = q_fft.get()
        # else:
            # fft_frame =  0
        # yield fft_frame
        yield q_fft.get()
    
    fig, ax1 = plt.subplots()
    fig.set_figheight(5)
    fig.set_figwidth(8)
    plt.grid()
    # fig.set_figheight(3)
    # fig.set_figwidth(14)
    fft_display = Power(ax1)
    ani = animation.FuncAnimation(fig, fft_display.update, frames=emittera,
                                blit=True,cache_frame_data=False, save_count=0,interval=window_time)
    plt.show()

STOP = 400
fs = 333
window_time = 1
NSAMP = int(fs * window_time)
if __name__ == '__main__':
    # STOP = 9
    # fs = 1000


    # with np.load('40hz_lowpass_fs1k.npz') as data:
    with np.load('40-55-333hz_lowpass.npz') as data:
        ba_lowpass = data['ba']

    b_lowpass = []
    for ba in ba_lowpass:
        b_lowpass.append(float(ba[0]))
    a_lowpass = [0]*len(b_lowpass)
    a_lowpass[0] = 1

    # data = pandas.read_csv('data_inputs/1open-4reading.csv')
    data = pandas.read_csv('data_inputs/1open-4reading-300.csv')
    # volts = data['Channel 1 (V)']
    volts = data['0.026325']
    volts = volts[0:(fs*20)]
    volts_filt = 0#scipy.signal.lfilter(b_lowpass,a_lowpass,volts)

    q_volts = multiprocessing.Queue() #raw volts
    q_volts_filt = multiprocessing.Queue() # filtered volts
    q_v = multiprocessing.Queue() # raw volts for plotting
    q_filt_for_fft = multiprocessing.Queue() # filtered volts for fft
    q_fft_out = multiprocessing.Queue() # filtered volts for fft

    print('starting')
    # with concurrent.futures.ProcessPoolExecutor() as executor:
    #     send = executor.submit(send_data,volts,q_volts)
    #     # filt = executor.submit(filter_bysample(a_lowpass,b_lowpass,q_volts,q_volts_filt))
    #     # plotting = executor.submit(anim,q_volts,q_volts_filt)
    send = multiprocessing.Process(target=send_data,args=(volts,q_volts,q_v))
    filt = multiprocessing.Process(target=filt_data,args=(a_lowpass,b_lowpass,q_volts,q_volts_filt,q_filt_for_fft))
    # t0 = time.time()
    graph = multiprocessing.Process(target=draw_signals,args=(q_volts_filt,q_v,fs,60))
    fft_p = multiprocessing.Process(target=fft_data,args=(q_filt_for_fft,q_fft_out))
    graph_fft = multiprocessing.Process(target=draw_fft,args=(q_fft_out,))
    graph_fft.start()
    fft_p.start()
    send.start()
    filt.start()

    graph.start()
    test = []

    # while q_volts_filt.empty() is False:
    #     a = q_volts_filt.get()
    # while q_v.empty() is False:
    #     a = q_v.get()
    # while q_fft_out.empty() is False:
    #     vals = q_fft_out.get()
    #     test.append(vals)
    # while q_filt_for_fft.empty() is False:
    #     a = q_filt_for_fft.get()
    
    send.join()
    # print('done send')
    filt.join()
    fft_p.join()
    # print('done filt')
    graph.join()
    graph_fft.join()
    print('done graph')

    # df = pandas.DataFrame(test)
    # df.to_csv('outputs/power.csv')
    # vals = np.array(test)
    # vals = 2*np.abs(test)
    # a = 0
    # # vals = test[0].tolist()
    # plt.stem(vals[0])
    # plt.show()