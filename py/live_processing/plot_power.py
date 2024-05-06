import numpy as np
from numpy import load
import scipy
import csv
import pandas
import matplotlib.pyplot as plt
import time
from scopetest import *
import multiprocessing
from multiprocessing import Queue
from scipy.fft import fft,fftfreq
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime

import sys
sys.path.append('../')
import python_rx as prx

def current_timestamp():
    """
    returns curent date and time
    """
    now = datetime.now()
    timestamp = now.strftime("%Y-%m-%d %H_%M_%S")
    return timestamp

def read_UART(CH1_out1, CH1_out2, CH2_out1, CH2_out2, CH1_out_csv):
# def read_UART():
    """
    Reads data coming in from UART connection and outputs the unfiltered 
    data into q_out1, q_out2
    """
    serialInst = prx.serial.Serial()
    prx.Port_Init(serialInst)
    packet = []
    # for i in range(600):
    #     q_out1.send(-1)
    #     q_out2.send(-1)
    # time.sleep(4)
    while True: # probably need to change to while the usb port is open or something
        if (serialInst.inWaiting() > 0): 
            #add to data read
            packet.append(serialInst.read())
            #once full number is recieved print the value in: original 3 byte array, 
            #concatinated 6 digit hex, and decimal values
        if (len(packet) == 4):
            val = prx.byte2int(packet[1::])
            channel = int.from_bytes(packet[0], byteorder='big') #16 = ch 1
            volts = ((val*2.4)/8388608)-1.65
            # print(channel,volts)
            if channel == 0:
                CH1_out1.send(volts)
                CH1_out2.send(volts)
                CH1_out_csv.send(volts)
            if channel == 16:
                CH2_out1.send(volts)
                CH2_out2.send(volts)
                # print(volts)
            #reset read value memory
            packet = []
    
def csv_save(raw_rx):
    timestamp = current_timestamp()
    file_path = f'./outputs/{timestamp}.csv'

    wait_while_empty(raw_rx)
    readval = raw_rx.recv()
    # print(readval)
    while (readval:=raw_rx.recv()) != STOP:
        # with open(file_path,'w',newline='') as csv_file:
        #     csv_writer = csv.writer(csv_file)
        #     csv_writer.writerow(readval)
            # print(readval)
        wait_while_empty(raw_rx)
    print(f'Done writing to file {file_path}')


def send_data(volts, CH1_out1, CH1_out2, CH1_out_csv):
    """
    Saves data sample by sample onto the output queue. Includes a delay to simulate sampling rate.
    Run in a parallel process. 
    Inputs:
        volts   - pre-recorded data list
    Outputs:
        CH1_out1 - Queue for filter function
        CH1_out2     - Queue for raw volts graphing function
    """
    time.sleep(2)
    for v in volts:
        CH1_out1.send(v)
        CH1_out2.send(v)
        CH1_out_csv.send(v)
        time.sleep(.003)
        # time.sleep(1)
        # print(v)
        # time.sleep(1)
    CH1_out1.send(STOP)
    CH1_out2.send(STOP)
    print("STOP")

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


def filt_data(a1,b1,a2,b2,CH1_in,CH1_out,CH1_out2,CH2_in,CH2_out,CH2_out2):
    """
    Filters incoming data according to a and b coefficients.
    CH1_out - filtered volts for graphing
    CH1_out2 - filtered volts for power calculations
    """
    z = [0]*(len(b1)-1)
    x = 0
    while x != STOP:
        val,z = filter([x],a1,b1,z)
        CH1_out.send(val[0])
        CH1_out2.send(val[0])
        x = CH1_in.recv()
    CH1_out.send(STOP)
    CH1_out2.send(STOP)

def filt_and_interpolate(a_smooth,b_smooth,a_int,b_int,CH1_in,CH1_out):
    """
    Used to smooth incoming power values, upsample, and interpolate
    """
    wait_while_empty(CH1_in)
    x = CH1_in.recv()
    smooth_z = np.zeros(shape=(len(x),len(b_smooth)-1))
    int_z= np.zeros(shape=(len(x),len(b_int)-1))
    results = np.zeros(shape=(len(x),3))

    while x != STOP:
        for i in range(len(x)):
            smooth_val,smooth_z[i] = filter([x[i]],a_smooth,b_smooth,smooth_z[i])
            smooth_result = np.append(smooth_val,0)
            smooth_result = np.append(smooth_result,0)

        # -------- no interpolation -------- #
            # results[i] = smooth_val
        #     print(f'smooth_val={smooth_val}')
        #     print('')
        # CH1_out.send(results[:,0])

        # -------- interpolate -------- #
            for count, value in enumerate(smooth_result):
                int_val,int_z[i] = filter([value],a_int,b_int,int_z[i])
                results[i][count] = int_val[0]
        for i in range(len(results[0])):
            CH1_out.send(results[:,i])

        x = CH1_in.recv()
    CH1_out.send([STOP]*3)
    

def wait_while_empty(q):
    while q.poll() is False:
        pass
    return

def calculate_powers(CH1_in, CH1_out,t_window,fs):
    """
    Performs Short Time Fourier Transform and power calculations for frequency bands
    """
    def fft_of_block(data,size):
        return np.abs(np.array(fft(data,size))) 
    
    # Frequency Bands
    alpha_range = [8,12]
    beta_range = [12,30]
    theta_range = [4,6]

    if (n_window:=int(t_window*fs))%2:
        # ensures window size is even
        n_window = n_window + 1
    n_overlap = int(n_window/2)
    data_buffer = []
    read_val = 0
    size = n_window#2 ** round(math.log(NSAMP,2))
    window = np.hanning(size)
    F = fs*np.array(fftfreq(size))

    theta_index = [(np.abs(F - theta_range[0])).argmin(),(np.abs(F - theta_range[1])).argmin()]
    alpha_index = [(np.abs(F - alpha_range[0])).argmin(),(np.abs(F - alpha_range[1])).argmin()]
    beta_index = [(np.abs(F - beta_range[0])).argmin(),(np.abs(F - beta_range[1])).argmin()]
    eeg_index = [(np.abs(F - theta_range[0])).argmin(),(np.abs(F - beta_range[1])).argmin()]

    i_begin = 0
    i_end = i_begin + n_window
    ffts = []
    ffts_offset = [0]*n_window
    count = 0
    
    # fill up buffer for one window length
    while len(data_buffer)<n_window-1:
        if CH1_in.poll() is True:
            read_val = CH1_in.recv()
            data_buffer.append(read_val)

    wait_while_empty(CH1_in)
    while (data_buffer[-1]!=STOP):
        data_buffer.append(CH1_in.recv())
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
            beta_power = np.sum(power_block[beta_index[0]:beta_index[1]+1])
            theta_power = np.sum(power_block[theta_index[0]:theta_index[1]+1])
            tot_power = np.sum(power_block[eeg_index[0]:eeg_index[1]+1])

            denominator = tot_power
            CH1_out.send([alpha_power/denominator,beta_power/denominator,theta_power/denominator])
        i_begin +=1
        i_end +=1
        count += 1
        if data_buffer[-1]!=STOP:
            wait_while_empty(CH1_in)
    CH1_out.send(STOP)

count = 0
t0 = 0
def draw_signals(CH1_in_filt,CH1_in_raw,fs,f_refresh): 
    # samp_per_update = 1
    samp_per_update = int(fs/f_refresh)
    # fig, (ax1,ax2) = plt.subplots(2,1)
    fig, ax1 = plt.subplots()
    fig.set_figheight(3)
    fig.set_figwidth(14)
    plt.grid()
    scope = Scope_Signal(ax1,CH1_in_filt,CH1_in_raw,samp_per_update)
    ani = animation.FuncAnimation(fig, scope.update,
                                blit=True, cache_frame_data=False, save_count=0,interval=int(1000/(f_refresh*1.5)))
    # ani = animation.FuncAnimation(fig, scope.update,
                            # blit=False, cache_frame_data=False, save_count=0,interval=0)

    plt.show()

def draw_power(CH1_in):
    fig, ax1 = plt.subplots()
    fig.set_figheight(5)
    fig.set_figwidth(8)
    ax1.set_title('Alpha Relative Power')
    ax1.set_xlabel('Time (s)')
    # ax1.set_ylim(0, 1)
    # ax1.set_xlim(0, maxt)
    plt.grid()
    fft_display = Scope_Power(ax1,CH1_in,maxt=30,dt=.166)

    ani = animation.FuncAnimation(fig, fft_display.update, blit=False,cache_frame_data=False, save_count=0,interval=160)
    plt.show()

def draw_power_bars(CH1_fft): # bar graph
    def emittera():

        # while CH1_fft.empty() is True:
        #     pass
        wait_while_empty(CH1_fft)

        # if CH1_fft.empty() is False:
        #     fft_frame = CH1_fft.get()
        # else:
            # fft_frame =  0
        # yield fft_frame

        # yield CH1_fft.get()
        yield CH1_fft.recv()
    
    fig, ax1 = plt.subplots()
    ax1.set_ylim(0,1)
    bar_positions = [0,1,2]
    colors = ['slategrey','cornflowerblue','salmon']
    # bar_positions = [0]
    bars = ax1.bar(bar_positions,[0]*len(bar_positions),color=colors)
    labels = ['alpha', 'beta','theta']
    # labels = ['alpha']
    ax1.set_xticks(bar_positions)
    ax1.set_xticklabels(labels)
    # ax1.set_facecolor('tab:gray')
    def update(x):
        y = CH1_fft.recv()
        # bars[0].set_height(y[0])
        for count, bar in enumerate(bars):
            bar.set_height(y[count])
        return bars
    
    ani = animation.FuncAnimation(fig, update,
                                blit=True,cache_frame_data=False, save_count=0,interval=160)
    plt.show()

STOP = 400
# fs = 333
fs = 566
# fs = 324
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

    # b_int = [-0.0928,0.0000,0.5862,1.0000,0.5862,0.0000,-0.0928] #from matlab intfilt(). p=l=2,a=0.5
    b_int = [0.0195,0.0220,-0.0000,-0.0980,-0.1216,0.0000,0.3938,0.7858,1.0000,0.7858,0.3938,0.0000,-0.1216,-0.0980,-0.0000,0.0220,0.0195]

    # data = pandas.read_csv('data_inputs/1open-4reading.csv')
    # data = pandas.read_csv('data_inputs/1open-4reading-300.csv')
    data = pandas.read_csv('data_inputs/1open-1closed-1open-2closedmusicrelax.csv')
    # volts = data['Channel 1 (V)']
    # volts = data['0.026325']
    volts = data['0.034611']
    # volts = volts[0:(fs*10)]

    print('starting')

    CH1_power_filt_rx, CH1_power_filt_tx = multiprocessing.Pipe()
    CH1_volts_rx, CH1_volts_tx = multiprocessing.Pipe()
    CH1_v_rx, CH1_v_tx = multiprocessing.Pipe()
    CH1_volts_filt_rx, CH1_volts_filt_tx = multiprocessing.Pipe()
    CH1_filt_for_fft_rx, CH1_filt_for_fft_tx = multiprocessing.Pipe()
    CH1_fft_out_rx, CH1_fft_out_tx = multiprocessing.Pipe()
    
    CH2_power_filt_rx, CH2_power_filt_tx = multiprocessing.Pipe()
    CH2_volts_rx, CH2_volts_tx = multiprocessing.Pipe()
    CH2_v_rx, CH2_v_tx = multiprocessing.Pipe()
    CH2_volts_filt_rx, CH2_volts_filt_tx = multiprocessing.Pipe()
    CH2_filt_for_fft_rx, CH2_filt_for_fft_tx = multiprocessing.Pipe()
    CH2_fft_out_rx, CH2_fft_out_tx = multiprocessing.Pipe()

    csv_raw_rx, csv_raw_tx = multiprocessing.Pipe()

    with ProcessPoolExecutor(max_workers=None) as executor:

        graph_signals = executor.submit(draw_signals,CH1_volts_filt_rx,CH1_v_rx,fs,60)

        fs_fft = fs
        power_calc = executor.submit(calculate_powers,CH1_filt_for_fft_rx,CH1_fft_out_tx,1,fs_fft)
        # graph_bar_power = executor.submit(draw_power_bars,power_filt_rx)
        graph_fft = executor.submit(draw_power,CH1_power_filt_rx)

        
        time.sleep(2)
        

        filter_data = executor.submit(filt_data,a_lowpass,b_lowpass,CH1_volts_rx,CH1_volts_filt_tx,CH1_filt_for_fft_tx)

        # send = executor.submit(send_data,volts,CH1_volts_tx,CH1_v_tx,csv_raw_tx)
        send = executor.submit(read_UART,CH1_volts_tx,CH1_v_tx,csv_raw_tx)
        # uart = executor.submit(read_UART)
        save_in_csv = executor.submit(csv_save,csv_raw_rx)
        filt_power = executor.submit(filt_and_interpolate,a_smooth,b_smooth,1,b_int,CH1_fft_out_rx,CH1_power_filt_tx)