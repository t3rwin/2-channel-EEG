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

def send_data(volts,q_volts,q_v):
    for v in volts:
        q_volts.put(v)
        q_v.put(v)
        # print(f'(v:{v})')
        time.sleep(.003)
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
    # while len(data_buffer) < NSAMP:
    
    while read_val != STOP:
        read_val = q_in.get()
        # t1 = time.time()
        while (len(data_buffer) < NSAMP) and (read_val != STOP):
            data_buffer.append(read_val)
            read_val = q_in.get()
        fft_vals = fft(data_buffer)
        q_out.put(fft_vals[0:int((len(fft_vals)+1)/2)])
        data_buffer = []
        # t2 = time.time()
        # print(t2-t1)
    if read_val == STOP:
        q_out.put(STOP)



# q_v is queue for plotting raw volts
def draw_signals(q_volts,q_volts_filt,q_v):
    def emitter3():
        if q_volts_filt.empty() is False:
            filt_val = q_volts_filt.get()
            if filt_val != STOP:
                r2 = filt_val
            else:
                r2 = 0
        else:
            r2 = 0
        if q_v.empty() is False:
            val = q_v.get()#q_volts.get()
            if val != STOP:
                r1 = val
            else:
                r1 = 0
        else:
            r1 = 0
        yield [r1,r2]
    
    # plt.rcParams['figure.figsize'] = [15, 4]
    fig, (ax1,ax2) = plt.subplots(2,1)
    fig.set_figheight(3)
    fig.set_figwidth(14)
    # fig.grid
    scope = Scope(ax1,ax2)
    # t0 = time.time()
    ani = animation.FuncAnimation(fig, scope.update, frames=emitter3,
                                blit=True,cache_frame_data=False, save_count=0,interval=10)
    plt.show()
    # print(scope.tdata_export)
    # df = pandas.DataFrame(scope.tdata_export)
    # df.to_csv('time.csv',index=False)
    # t1 = time.time()
    # print(t1-t0)

    # print(q_volts.get())

def draw_fft(q_fft):
    def emittera():
        if q_fft.empty() is False:
            fft_frame = np.array(q_fft.get())
            # print(len(fft_frame))
        else:
            fft_frame =  NSAMP * [8]

            # if fft_frame == STOP:
                # fft_frame = NSAMP * [0]
        # else:
            # fft_frame = NSAMP * [0]
        # print(fft_frame)
        yield np.abs(fft_frame)
    
    fig, ax1 = plt.subplots()
    fig.set_figheight(5)
    fig.set_figwidth(8)
    plt.grid()
    # fig.set_figheight(3)
    # fig.set_figwidth(14)
    fft_display = FFT_Display(ax1,NSAMP,fs)
    ani = animation.FuncAnimation(fig, fft_display.update, frames=emittera,
                                blit=True,cache_frame_data=False, save_count=0,interval=1000*window_time)
    plt.show()

STOP = 9
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
    # volts = volts[0:(fs*30)]
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
    graph = multiprocessing.Process(target=draw_signals,args=(q_volts,q_volts_filt,q_v))
    fft_p = multiprocessing.Process(target=fft_data,args=(q_filt_for_fft,q_fft_out))
    graph_fft = multiprocessing.Process(target=draw_fft,args=(q_fft_out,))
    send.start()
    filt.start()
    # graph_fft.start()
    fft_p.start()
    graph.start()
    
    
    send.join()
    # print('done send')
    filt.join()
    fft_p.join()
    # print('done filt')
    graph.join()
    # graph_fft.join()
    print('done graph')

    test = []
    # while q_fft_out.empty() is False:
    #     vals = q_fft_out.get()
    #     test.append(vals)
    # vals = np.array(test)
    # vals = 2*np.abs(test)
    # a = 0
    # # vals = test[0].tolist()
    # plt.stem(vals[0])
    # plt.show()