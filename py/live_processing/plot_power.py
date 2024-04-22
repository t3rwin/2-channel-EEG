import numpy as np
from numpy import load
import scipy
import csv
import pandas
import matplotlib.pyplot as plt
import time
from scopetest import *
import concurrent.futures
import multiprocessing
from multiprocessing import Queue
from scipy.fft import fft,fftfreq

def send_data(volts: list, q_volts: Queue, q_v: Queue):
    """
    Saves data sample by sample onto the output queue. Includes a delay to simulate sampling rate.
    Run in a parallel process. 
    Inputs:
        volts   - pre-recorded data list
    Outputs:
        q_volts - Queue for filter function
        q_v     - Queue for raw volts graphing function
    """
    for v in volts:
        q_volts.put(v)
        q_v.put(v)
        time.sleep(.003)
    q_volts.put(STOP)
    q_v.put(STOP)

def filter(data,a,b,z):
    """
    Returns filtered signal
    Inputs:
        data - signal array to filter
        a    - Ak coefficients
        b    - Bk coefficients
        z    - initial state
    Outputs:
        filtered signal array
    """
    return scipy.signal.lfilter(b,a,data,zi=z)


def filt_data(a,b,q_in,q_out,q_out2):
    # q_out - filtered data
    # q_out2 - 
    result = []
    z = [0]*(len(b)-1)
    x = 0
    while x != STOP:
        val,z = filter([x],a,b,z)
        result.append(val)
        q_out.put(val[0])
        q_out2.put(val[0])
        x = q_in.get()
    q_out.put(STOP)
    q_out2.put(STOP)

def filt_data2(a,b,q_in,q_out,q_out2):
    # q_out - filtered data
    # q_out2 - 
    result = []
    z = [0]*(len(b)-1)
    x = 0
    while x != STOP:
        val,z = filter([x],a,b,z)
        result.append(val)
        q_out.put(val[0])
        q_out2.put(x)
        x = q_in.get()
    q_out.put(STOP)
    q_out2.put(STOP)

def wait_while_empty(q):
    while q.empty() is True:
        pass
    return

def fft_data(q_in, q_out,t_window,fs):
    def fft_of_block(data,size):
        return np.abs(np.array(fft(data,size))) 
    
    if (n_window:=int(t_window*fs))%2:
        n_window = n_window + 1

    # Frequency Bands
    alpha_range = [8,12]
    beta_range = [12,30]
    theta_range = [4,6]


    n_overlap = int(n_window/2)
    data_buffer = []
    read_val = 0
    size = n_window#2 ** round(math.log(NSAMP,2))
    window = np.hanning(size)
    # window = [1]*size
    F = fs*np.array(fftfreq(size))
    alpha_index = [(np.abs(F - alpha_range[0])).argmin(),(np.abs(F - alpha_range[1])).argmin()]
    eeg_index = [(np.abs(F - theta_range[0])).argmin(),(np.abs(F - beta_range[1])).argmin()]

    i_begin = 0
    i_end = i_begin + n_window
    ffts = []
    ffts_offset = [0]*n_window
    count = 0
    
    # fill up buffer for one window lenght
    while len(data_buffer)<n_window-1:
        if q_in.empty() is False:
            read_val = q_in.get()
            data_buffer.append(read_val)

    wait_while_empty(q_in)
    # while (read_val:=q_in.get()) != STOP:
    while (data_buffer[-1]!=STOP):
        data_buffer.append(q_in.get())
        if count%n_overlap == 0: #ready for next block
            windowed_data = np.multiply(window,data_buffer[i_begin:i_end])
            if not count%2: #ffts_block - count is even
                ffts = fft_of_block(windowed_data,size)
                power_block = np.square(np.add(ffts[0:n_overlap],ffts_offset[n_overlap::]))
            else: #offset_block - count is odd
                ffts_offset = fft_of_block(windowed_data,size)
                power_block = np.square(np.add(ffts_offset[0:n_overlap],ffts[n_overlap::]))
                
                # reset variables
                data_buffer = data_buffer[i_begin:i_end]
                count = 0
                i_begin = 0
                i_end = i_begin + n_window
            
            alpha_power = np.sum(power_block[alpha_index[0]:alpha_index[1]+1])
            tot_power = np.sum(power_block[eeg_index[0]:eeg_index[1]+1])
            q_out.put(alpha_power/tot_power)
            # print(power_block)
            # q_out.put(power_block)
        i_begin +=1
        i_end +=1
        count += 1
        if data_buffer[-1]!=STOP:
            wait_while_empty(q_in)
    q_out.put(STOP)



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

def draw_fft(q_in,q_in2):
    def emittera():
        while (q_in.empty() and q_in2.empty()) is True:
            pass
        yield [q_in.get(), q_in2.get()]

    
    fig, ax1 = plt.subplots()
    fig.set_figheight(5)
    fig.set_figwidth(8)
    ax1.set_title('Alpha Relative Power')
    ax1.set_xlabel('Time (s)')
    plt.grid()
    # fig.set_figheight(3)
    # fig.set_figwidth(14)
    fft_display = Power(ax1)
    ani = animation.FuncAnimation(fig, fft_display.update, frames=emittera,
                                blit=True,cache_frame_data=False, save_count=0,interval=window_time)
    plt.show()
def draw_fft2(q_fft): # bar graph
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
    ax1.set_ylim(0,1)
    bars = ax1.bar(0,0)

    def update(y):
        for bar in bars:
            bar.set_height(y)
        return bars
    
    ani = animation.FuncAnimation(fig, update, frames=emittera,
                                blit=False,cache_frame_data=False, save_count=0,interval=window_time)
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

    b_smooth = scipy.signal.savgol_coeffs(7,2)
    a_smooth = [0]*len(b_smooth)
    a_smooth[0] = 1

    # data = pandas.read_csv('data_inputs/1open-4reading.csv')
    data = pandas.read_csv('data_inputs/1open-4reading-300.csv')
    # volts = data['Channel 1 (V)']
    volts = data['0.026325']
    # volts = volts[0:(fs*20)]
    volts_filt = 0#scipy.signal.lfilter(b_lowpass,a_lowpass,volts)

    q_volts = multiprocessing.Queue() #raw volts
    q_volts_filt = multiprocessing.Queue() # filtered volts
    q_v = multiprocessing.Queue() # raw volts for plotting
    q_filt_for_fft = multiprocessing.Queue() # filtered volts for fft
    q_fft_out = multiprocessing.Queue() # filtered volts for fft
    q_power_filt = multiprocessing.Queue() # filtered power values
    q_power_nofilt = multiprocessing.Queue() # unfiltered power values

    print('starting')
    # with concurrent.futures.ProcessPoolExecutor() as executor:
    #     send = executor.submit(send_data,volts,q_volts)
    #     # filt = executor.submit(filter_bysample(a_lowpass,b_lowpass,q_volts,q_volts_filt))
    #     # plotting = executor.submit(anim,q_volts,q_volts_filt)
    send = multiprocessing.Process(target=send_data,args=(volts,q_volts,q_v))
    filt = multiprocessing.Process(target=filt_data,args=(a_lowpass,b_lowpass,q_volts,q_volts_filt,q_filt_for_fft))
    # t0 = time.time()
    graph = multiprocessing.Process(target=draw_signals,args=(q_volts_filt,q_v,fs,60))
    fs_fft = fs
    fft_p = multiprocessing.Process(target=fft_data,args=(q_filt_for_fft,q_fft_out,1,fs_fft))
    graph_fft = multiprocessing.Process(target=draw_fft,args=(q_power_nofilt,q_power_filt))
    filt_power = multiprocessing.Process(target=filt_data2,args=(a_smooth,b_smooth,q_fft_out,q_power_filt,q_power_nofilt))
    graph_bar_power = multiprocessing.Process(target=draw_fft2,args=(q_power_filt,))
    graph_fft.start()
    # graph_bar_power.start()
    fft_p.start()
    send.start()
    filt.start()
    filt_power.start()

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
    # filt_power.join()
    # print('done filt')
    graph.join()
    graph_fft.join()
    # graph_bar_power.join()
    print('done graph')

    # df = pandas.DataFrame(test)
    # df.to_csv('outputs/power.csv')
    # vals = np.array(test)
    # vals = 2*np.abs(test)
    # a = 0
    # # vals = test[0].tolist()
    # plt.stem(vals[0])
    # plt.show()