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
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor

import sys
sys.path.append('../')
import python_rx as prx

def read_UART(q_volts, q_v):
    serialInst = prx.serial.Serial()
    prx.Port_Init(serialInst)
    packet = []
    while True: # probably need to change to while the usb port is open or something
        if (serialInst.inWaiting() > 0): 
            #add to data read
            packet.append(serialInst.read())

            #once full number is recieved print the value in: original 3 byte array, 
            #concatinated 6 digit hex, and decimal values
        if (len(packet) == 4):
            q_volts.put(prx.byte2int(packet[:3]))
            q_v.put(prx.byte2int(packet[:3]))
            #reset read value memory
            packet = []
    

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
    time.sleep(2)
    for v in volts:
        q_volts.send(v)
        q_v.send(v)
        time.sleep(.003)
    q_volts.send(STOP)
    q_v.send(STOP)

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
        q_out.send(val[0])
        q_out2.send(val[0])
        x = q_in.recv()
    q_out.send(STOP)
    q_out2.send(STOP)

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

def filt_and_interpolate(a_smooth,b_smooth,a_int,b_int,q_in,q_out):
    smooth_result = []
    smooth_z= [0]*(len(b_smooth)-1)

    int_result = []
    int_z= [0]*(len(b_int)-1)
    x = 0
    while x != STOP:
        smooth_val,smooth_z = filter([x],a_smooth,b_smooth,smooth_z)
        smooth_result = np.append(smooth_val,0)
        # smooth_result = np.append(smooth_result,0)

        for value in smooth_result:
            int_val,int_z = filter([value],a_int,b_int,int_z)
            q_out.send(int_val[0])

        x = q_in.recv()
    q_out.send(STOP)
    

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
        if q_in.poll() is True:
            read_val = q_in.recv()
            data_buffer.append(read_val)

    while q_in.poll() is False:
        pass
    while (data_buffer[-1]!=STOP):
        data_buffer.append(q_in.recv())
        if count%n_overlap == 0: #ready for next block
            windowed_data = np.multiply(window,data_buffer[i_begin:i_end])
            if not count%2: #ffts_block - count is even
                ffts = fft_of_block(windowed_data,size)
                power_block = np.square(np.add(ffts[0:n_overlap],ffts_offset[n_overlap::]))
            else: #offset_block - count is odd
                ffts_offset = fft_of_block(windowed_data,size)
                power_block = np.square(np.add(ffts_offset[0:n_overlap],ffts[n_overlap::]))
                
                data_buffer = data_buffer[i_begin:i_end]
                count = 0
                i_begin = 0
                i_end = i_begin + n_window
            
            alpha_power = np.sum(power_block[alpha_index[0]:alpha_index[1]+1])
            tot_power = np.sum(power_block[eeg_index[0]:eeg_index[1]+1])
            q_out.send(alpha_power/tot_power)
        i_begin +=1
        i_end +=1
        count += 1
        if data_buffer[-1]!=STOP:
            while q_in.poll() is False:
                pass
    q_out.send(STOP)

count = 0
t0 = 0
# q_v is queue for plotting raw volts
def draw_signals(q_in_filt,q_in_raw,fs,f_refresh): 
    def emitter3():
        global count
        global t0
        samp_per_update = int(fs/f_refresh)
        out_raw = []
        out_filt = []
        done = False
        while not done:
            if count == 333:
                t0 = time.time()
            if len(out_raw)<samp_per_update:
                while q_in_raw.poll() is False:
                    pass
                out_raw.append(q_in_raw.recv())
                count = count + 1
            if len(out_filt)<samp_per_update:
                if q_in_filt.poll() is True:
                # r2 = q_in_filt.get()
                    out_filt.append(q_in_filt.recv())
                else:
                    # r2 = 0
                    out_filt.append(0)
            if (len(out_raw)==samp_per_update) and (len(out_filt)==samp_per_update):
                done = True
            if count == 333*10:
                print(time.time()-t0)
        
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

def draw_fft(q_in):
    def emittera():
        while (q_in.empty()) is True:
            pass
        yield [q_in.get(),0]
    
    fig, ax1 = plt.subplots()
    fig.set_figheight(5)
    fig.set_figwidth(8)
    ax1.set_title('Alpha Relative Power')
    ax1.set_xlabel('Time (s)')
    plt.grid()
    fft_display = Power(ax1)

    ani = animation.FuncAnimation(fig, fft_display.update, frames=emittera,
                                blit=True,cache_frame_data=False, save_count=0,interval=.02)
    plt.show()
def draw_fft2(q_fft): # bar graph
    def emittera():

        # while q_fft.empty() is True:
        #     pass
        while q_fft.poll() is False:
            pass

        # if q_fft.empty() is False:
        #     fft_frame = q_fft.get()
        # else:
            # fft_frame =  0
        # yield fft_frame

        # yield q_fft.get()
        yield q_fft.recv()
    
    fig, ax1 = plt.subplots()
    ax1.set_ylim(0,1)
    bars = ax1.bar(0,0)

    def update(y):
        for bar in bars:
            bar.set_height(y)
        return bars
    
    ani = animation.FuncAnimation(fig, update, frames=emittera,
                                blit=False,cache_frame_data=False, save_count=0,interval=.02)
    plt.show()

STOP = 400
fs = 333
window_time = 1
NSAMP = int(fs * window_time)
if __name__ == '__main__':

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

    b_int = [-0.0928,0.0000,0.5862,1.0000,0.5862,0.0000,-0.0928] #from matlab intfilt(). p=l=2,a=0.5
    # b_int = [0.0195,0.0220,-0.0000,-0.0980,-0.1216,0.0000,0.3938,0.7858,1.0000,0.7858,0.3938,0.0000,-0.1216,-0.0980,-0.0000,0.0220,0.0195]

    # data = pandas.read_csv('data_inputs/1open-4reading.csv')
    # data = pandas.read_csv('data_inputs/1open-4reading-300.csv')
    data = pandas.read_csv('data_inputs/1open-1closed-1open-2closedmusicrelax.csv')
    # volts = data['Channel 1 (V)']
    # volts = data['0.026325']
    volts = data['0.034611']
    # volts = volts[0:(fs*20)]

    print('starting')

    power_filt_rx, power_filt_tx = multiprocessing.Pipe()
    q_volts_rx, q_volts_tx = multiprocessing.Pipe()
    q_v_rx, q_v_tx = multiprocessing.Pipe()
    q_volts_filt_rx, q_volts_filt_tx = multiprocessing.Pipe()
    q_filt_for_fft_rx, q_filt_for_fft_tx = multiprocessing.Pipe()
    q_fft_out_rx, q_fft_out_tx = multiprocessing.Pipe()

    with ProcessPoolExecutor(max_workers=None) as executor:

        graph_signals = executor.submit(draw_signals,q_volts_filt_rx,q_v_rx,fs,60)

        fs_fft = fs
        power_calc = executor.submit(fft_data,q_filt_for_fft_rx,q_fft_out_tx,1,fs_fft)

        # graph_fft = executor.submit(draw_fft,q_power_filt)

        filt_power = executor.submit(filt_and_interpolate,a_smooth,b_smooth,1,b_int,q_fft_out_rx,power_filt_tx)
        time.sleep(1)
        graph_bar_power = executor.submit(draw_fft2,power_filt_rx)

        filter_data = executor.submit(filt_data,a_lowpass,b_lowpass,    q_volts_rx,q_volts_filt_tx,q_filt_for_fft_tx)

        send = executor.submit(send_data,volts,q_volts_tx,q_v_tx)