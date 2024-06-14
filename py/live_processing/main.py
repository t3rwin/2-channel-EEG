import numpy as np
from numpy import load
import scipy
import csv
import pandas
import time
from live_display import *
import multiprocessing
from scipy.fft import fft,fftfreq
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime
import os
import sys
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams['toolbar'] = 'None'

sys.path.append('../')
import python_rx as prx

def current_timestamp():
    """
    returns curent date and time
    """
    now = datetime.now()
    timestamp = now.strftime("%Y-%m-%d %H_%M_%S")
    return timestamp

def send_data(volts, volts2, CH1_out1, CH1_out2, CH2_out1, CH2_out2, CH1_volts_raw_csv,CH2_volts_raw_csv):
    """
    Saves data sample by sample onto the output queue. Includes a delay to simulate sampling rate. Used for testing
    Run in a parallel process. 
    Inputs:
        volts   - pre-recorded data list
    Outputs:
        q_out1 - Queue for filter function
        q_out2     - Queue for raw volts graphing function
    """
    # time.sleep(2)
    for i, v in enumerate(volts):
        # CH1_out1.send(v)
        CH1_out2.send(v)
        # CH2_out1.send(v)
        CH2_out2.send(volts2[i])
        CH1_volts_raw_csv.send(v)
        CH2_volts_raw_csv.send(volts2[i])
        time.sleep(.0017)
    print("STOP")

def read_UART(CH1_out1, CH1_out2, CH2_out1, CH2_out2, CH1_volts_raw_csv,CH2_volts_raw_csv):
    """
    Reads data coming in from UART connection and outputs the unfiltered 
    data into q_out1, q_out2
    """
    serialInst = prx.serial.Serial()
    prx.Port_Init(serialInst)
    packet = []
    while True: # Should change to while the usb port is open
        if (serialInst.inWaiting() > 0): 
            #add to data read
            packet.append(serialInst.read())
        if (len(packet) == 4):
            val = prx.byte2int(packet[1::])
            channel = int.from_bytes(packet[0], byteorder='big') #16 = ch 1
            volts = ((val*2.4)/8388608)-1.65
            if channel == 0:
                # CH1_out1.send(volts) # for raw
                CH1_out2.send(volts)      # for filtered
                CH1_volts_raw_csv.send(volts)
                # CH1_out2.send(volts)
                # CH1_out_csv.send(volts)
            else:
                # CH2_out1.send(volts)
                CH2_out2.send(volts)
                CH2_volts_raw_csv.send(volts)
            #reset read value memory
            packet = []
    
def csv_save_power(timestamp,window_time,q_powers):
    file_path = f'./outputs/{timestamp}/powers_{timestamp}.csv'
    lines = 0
    header = ['time(s)','theta power','alpha power','beta power', 'total power']
    while True:
        wait_while_empty(q_powers)
        powers = q_powers.recv()
        with open(file_path,'a') as csv_file:
            csv_writer = csv.writer(csv_file)
            if lines == 0:
                csv_writer.writerow(header)
            csv_writer.writerow([str(round(lines*(window_time/2),3)),str(powers[0]),str(powers[1]),str(powers[2]),str(powers[3])])
            lines = lines + 1

def csv_save_volts(timestamp,fs,CH1_volts_raw,CH2_volts_raw,CH1_filt,CH2_filt):
    file_path = f'./outputs/{timestamp}/volts_{timestamp}.csv'
    lines = 0
    header = ['time(s)','CH1 volts','CH2 volts','CH1 filtered','CH2 filtered']
    while True:
        wait_while_empty(CH1_volts_raw)
        v1_r = CH1_volts_raw.recv()
        wait_while_empty(CH2_volts_raw)
        v2_r = CH2_volts_raw.recv()

        wait_while_empty(CH1_filt)
        v1_f = CH1_filt.recv()
        wait_while_empty(CH2_filt)
        v2_f = CH2_filt.recv()

        with open(file_path,'a') as csv_file:
            csv_writer = csv.writer(csv_file)
            if lines == 0:
                csv_writer.writerow(header)
            csv_writer.writerow([str(round(lines*(1/fs),5)),str(v1_r),str(v2_r),str(v1_f),str(v2_f)])
            lines = lines + 1

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

def filt_data(a,b,CH1_in,CH1_out,CH1_out2,CH2_in,CH2_out,CH2_out2,CH1_out_csv, CH2_out_csv):
    """
    Filters incoming data according to a and b coefficients.
    q_out - filtered volts for graphing
    q_out2 - filtered volts for power calculations
    """
    z_1 = [0]*(len(b)-1)
    x_1 = 0

    z_2 = [0]*(len(b)-1)
    x_2 = 0
    while x_1 != STOP:
        val_1,z_1 = filter([x_1],a,b,z_1)
        CH1_out.send(val_1[0])
        CH1_out2.send(val_1[0])
        CH1_out_csv.send(val_1[0])

        val_2,z_2 = filter([x_2],a,b,z_2)
        CH2_out.send(val_2[0])
        CH2_out2.send(val_2[0])
        CH2_out_csv.send(val_2[0])

        wait_while_empty(CH1_in)
        x_1 = CH1_in.recv()

        wait_while_empty(CH2_in)
        x_2 = CH2_in.recv()
    CH1_out.send(STOP)
    # CH1_out2.send(STOP)


def filt_and_interpolate(a_smooth,b_smooth,a_int,b_int,
                         q_in,q_in2,q_out,q_out2):
    """
    Used to smooth incoming power values, upsample, and interpolate
    """
    wait_while_empty(q_in)
    in1 = q_in.recv()
    in2 = q_in2.recv()
    x = in1 + in2
    smooth_z = np.zeros(shape=(len(x),len(b_smooth)-1))
    int_z= np.zeros(shape=(len(x),len(b_int)-1))
    # results = np.zeros(shape=(len(x),3))
    results = [0]*len(x)#[0,0,0] # for no interpolation

    while x != STOP:
        # q_out.send(x)
        # x = q_in.recv()
        for i in range(len(x)):
            smooth_val,smooth_z[i] = filter([x[i]],a_smooth,b_smooth,smooth_z[i])
            # smooth_result = np.append(smooth_val,0)
            # smooth_result = np.append(smooth_result,0)
            # smooth_result = smooth_val

        # -------- no interpolation -------- #
            results[i] = smooth_val[0]
            # q_out.send(results)
        q_out.send(results[:len(in1)])
        q_out2.send(results[len(in1):])

            # print(f'smooth_val={smooth_val}')
            # print('')
        # q_out.send(results[:,0])
        # q_out.send(results[:,0])

        # -------- interpolate -------- #
            # for count, value in enumerate(smooth_result):
            #     int_val,int_z[i] = filter([value],a_int,b_int,int_z[i])
            #     results[i][count] = int_val[0]
            #     print(int_val)
        
        # for i in range(len(results[0])):
        #     q_out.send(results[:,i])
            # print(results)

        # x = q_in.recv()
        in1 = q_in.recv()
        in2 = q_in2.recv()
        x = in1 + in2
    q_out.send([STOP]*3)
    

def wait_while_empty(q):
    while q.poll() is False:
        pass
    return

def calculate_powers(CH1_in, CH2_in,out,out2,t_window,fs,powers_csv):
    """
    Performs Short Time Fourier Transform and power calculations for frequency bands
    """
    def fft_of_block(data,size):
        return np.abs(np.array(fft(data,size))) 
    
    # Frequency Bands
    theta_range = [4,8]  # 4hz
    alpha_range = [8,12] # 4hz
    beta_low = [12,15]   # 3hz
    beta_mid = [15,20]   # 5hz
    beta_high = [20,29]  # 9hz
    beta_range = [12,29] # 17hz

    

    if (n_window:=int(t_window*fs))%2:
        # ensures window size is even
        n_window = n_window + 1

    n_overlap = int(n_window/2)

    data_buffer = []
    read_val = 0

    data_buffer2 = []
    read_val2 = 0

    size = n_window#2 ** round(math.log(NSAMP,2))
    window = np.hanning(size)
    F = fs*np.array(fftfreq(size))

    theta_index = [(np.abs(F - theta_range[0])).argmin(),(np.abs(F - theta_range[1])).argmin()]
    alpha_index = [(np.abs(F - alpha_range[0])).argmin(),(np.abs(F - alpha_range[1])).argmin()]
    beta_index = [(np.abs(F - beta_range[0])).argmin(),(np.abs(F - beta_range[1])).argmin()]
    betalow_index = [(np.abs(F - beta_low[0])).argmin(),(np.abs(F - beta_low[1])).argmin()]
    betamid_index = [(np.abs(F - beta_mid[0])).argmin(),(np.abs(F - beta_mid[1])).argmin()]
    betahigh_index = [(np.abs(F - beta_high[0])).argmin(),(np.abs(F - beta_high[1])).argmin()]


    eeg_index = [(np.abs(F - theta_range[0])).argmin(),(np.abs(F - beta_range[1])).argmin()]

    i_begin = 0
    i_end = i_begin + n_window
    ffts_ch1 = []
    ffts_offset_ch1 = [0]*n_window
    ffts_ch2 = []
    ffts_offset_ch2 = [0]*n_window
    
    count = 0
    
    # fill up buffer for one window length
    while (len(data_buffer)<n_window-1) and (len(data_buffer2)<n_window-1):
        if (len(data_buffer)<n_window-1):
            if CH1_in.poll() is True:
                read_val = CH1_in.recv()
                data_buffer.append(read_val)
        if (len(data_buffer2)<n_window-1):
            if CH2_in.poll() is True:
                read_val2 = CH2_in.recv()
                data_buffer2.append(read_val2)
    # while len(data_buffer2)<n_window-1:
    #     if CH2_in.poll() is True:
    #         read_val2 = CH2_in.recv()
    #         data_buffer2.append(read_val2)

    wait_while_empty(CH1_in)
    wait_while_empty(CH2_in)
    while (data_buffer[-1]!=STOP):
        # wait_while_empty(CH1_in)
        # wait_while_empty(CH2_in)
        data_buffer.append(CH1_in.recv())
        data_buffer2.append(CH2_in.recv())
        if count%n_overlap == 0: #ready for next block
            windowed_data_ch1 = np.multiply(window,data_buffer[i_begin:i_end])
            windowed_data_ch2 = np.multiply(window,data_buffer2[i_begin:i_end])
            if not count%2: #ffts_block - count is even
                ffts_ch1 = fft_of_block(windowed_data_ch1,size)
                ffts_ch2 = fft_of_block(windowed_data_ch2,size)
                power_block_ch1 = np.square(np.add(ffts_ch1[0:n_overlap],ffts_offset_ch1[n_overlap::]))
                power_block_ch2 = np.square(np.add(ffts_ch2[0:n_overlap],ffts_offset_ch2[n_overlap::]))
            else: #offset_block - count is odd
                ffts_offset_ch1 = fft_of_block(windowed_data_ch1,size)
                ffts_offset_ch2 = fft_of_block(windowed_data_ch2,size)
                power_block_ch1 = np.square(np.add(ffts_offset_ch1[0:n_overlap],ffts_ch1[n_overlap::]))
                power_block_ch2 = np.square(np.add(ffts_offset_ch2[0:n_overlap],ffts_ch2[n_overlap::]))
                
                data_buffer = data_buffer[i_begin:i_end]
                data_buffer2 = data_buffer2[i_begin:i_end]
                count = 0
                i_begin = 0
                i_end = i_begin + n_window
            
            alpha_power = np.sum(power_block_ch1[alpha_index[0]:alpha_index[1]+1]) + np.sum(power_block_ch2[alpha_index[0]:alpha_index[1]+1])

            beta_power = np.sum(power_block_ch1[beta_index[0]:beta_index[1]+1]) + np.sum(power_block_ch2[beta_index[0]:beta_index[1]+1])

            beta_power_low = np.sum(power_block_ch1[betalow_index[0]:betalow_index[1]+1]) + np.sum(power_block_ch2[betalow_index[0]:betalow_index[1]+1])

            beta_power_mid = np.sum(power_block_ch1[betamid_index[0]:betamid_index[1]+1]) + np.sum(power_block_ch2[betamid_index[0]:betamid_index[1]+1])

            beta_power_high = np.sum(power_block_ch1[betahigh_index[0]:betahigh_index[1]+1]) + np.sum(power_block_ch2[betahigh_index[0]:betahigh_index[1]+1])

            theta_power = np.sum(power_block_ch1[theta_index[0]:theta_index[1]+1]) + np.sum(power_block_ch2[theta_index[0]:theta_index[1]+1])

            tot_power = np.sum(power_block_ch1[eeg_index[0]:eeg_index[1]+1]) + np.sum(power_block_ch2[eeg_index[0]:eeg_index[1]+1])

            denominator = tot_power
            out.send([theta_power/denominator,alpha_power/denominator,beta_power_low/denominator,beta_power_mid/denominator,beta_power_high/denominator])
            relaxation = ((alpha_power + beta_power_low)/(beta_power_mid+beta_power_high))/10
            concentration = ((beta_power_mid + beta_power_high)/(alpha_power + beta_power_low)/2)
            out2.send([relaxation,concentration])
            # out.send([theta_power/denominator,alpha_power/denominator,beta_power_low/denominator,beta_power_mid/denominator,beta_power_high/denominator])
            # powers_csv.send([theta_power,alpha_power,beta_power,tot_power])
        i_begin +=1
        i_end +=1
        count += 1
        if data_buffer[-1]!=STOP:
            wait_while_empty(CH1_in)
    out.send(STOP)

count = 0
t0 = 0
def draw_signals(CH1_in,CH2_in,fs,f_refresh): 
    # CH2_in = 0
    samp_per_update = int(fs/f_refresh)
    fig, (ax1,ax2) = plt.subplots(2,1, sharex=True,sharey=True)
    # fig, (ax1,ax2) = plt.subplots(2,1)
    fig.canvas.manager.set_window_title('Channel Data')
    # fig, ax1 = plt.subplots()
    fig.set_figheight(5)
    fig.set_figwidth(10)
    fig.text(0.5, 0.02, 'Seconds', ha='center')
    fig.text(0.08, 0.5, 'Volts', va='center', rotation='vertical')
    # plt.grid()
    ax1.set_title('Channel 1',)
    ax2.set_title('Channel 2')
    scope = Scope_Signal(fig,ax1,ax2,CH1_in,CH2_in,samp_per_update)
    ani = animation.FuncAnimation(fig, scope.update,
                                blit=True, cache_frame_data=False, save_count=0,interval=int(1000/(f_refresh*1.5)))
    # ani = animation.FuncAnimation(fig, scope.update,
                            # blit=False, cache_frame_data=False, save_count=0,interval=0)

    plt.show()

def draw_power(q_in):
    fig, ax1 = plt.subplots()
    fig.set_figheight(5)
    fig.set_figwidth(8)
    ax1.set_title('Alpha Relative Power')
    ax1.set_xlabel('Time (s)')
    # ax1.set_ylim(0, 1)
    # ax1.set_xlim(0, maxt)
    plt.grid()
    # fft_display = Scope_Power(ax1,q_in,maxt=30,dt=.166)
    fft_display = Scope_Power(ax1,q_in,maxt=30,dt=.5)

    ani = animation.FuncAnimation(fig, fft_display.update, blit=True,cache_frame_data=False, save_count=0,interval=160)#160
    plt.show()

def draw_feedback_bars(q_in):
    fig, ax1 = plt.subplots()
    fig.set_figheight(4)
    fig.set_figwidth(4)
    fig.canvas.manager.set_window_title('Feedback')
    ax1.set_ylim(0,1)
    ax1.set_title('Feedback')
    labels = ['relaxation','concentration']
    bar_positions = [i for i in range(len(labels))]
    colors = ['lightsteelblue','#8f99fb','salmon']
    bars = ax1.bar(bar_positions,[0]*len(bar_positions),color=colors)
    ax1.set_xticks(bar_positions)
    ax1.set_xticklabels(labels)

    def update(x):
        y = q_in.recv()
        for count, bar in enumerate(bars):
            bar.set_height(y[count])
        return bars

    ani = animation.FuncAnimation(fig, update,
                                blit=True,cache_frame_data=False, save_count=0,interval=160)#160 
    plt.show()

def draw_power_bars(q_in): # bar graph
    fig, ax1 = plt.subplots()
    fig.set_figheight(4)
    fig.set_figwidth(4)
    fig.canvas.manager.set_window_title('Relative Power of EEG Bands')
    ax1.set_ylim(0,1)
    ax1.set_xlabel('EEG Frequency Band')
    ax1.set_ylabel('Relative Power')
    # bar_positions = [0,1,2,3,4]
    # bar_positions = [0,1,2]
    # labels = ['theta', 'alpha','beta']
    labels = ['theta','alpha','low beta','mid beta','high beta']#,'relax','conc']
    # labels=['theta','alpha','beta low','beta mid']
    # labels = ['theta','alpha','beta high']#,'low beta','mid beta','high beta']
    bar_positions = [i for i in range(len(labels))]
    colors = ['lightsteelblue','#8f99fb','salmon']
    # bar_positions = [0]
    bars = ax1.bar(bar_positions,[0]*len(bar_positions),color=colors)

    ax1.set_xticks(bar_positions)
    ax1.set_xticklabels(labels)
    # ax1.set_facecolor('tab:gray')
    def update(x):
        y = q_in.recv()
        # y2 = q_in2.recv()

        for count, bar in enumerate(bars):
            # if count < len(y):
            bar.set_height(y[count])
            # else:
                # bar.set_height(y2[count-len(y)])
        return bars
    
    ani = animation.FuncAnimation(fig, update,
                                blit=True,cache_frame_data=False, save_count=0,interval=160)#160 
    plt.show()

def eat_pipe(*qin):
    """
    Flushes any unused pipes so that the program doesn't back up due to clogged pipes
    """
    while True:
        for q in qin:
            if q.poll() is True:
                a = q.recv()

STOP = 400
# fs = 333
fs = 566
# fs = 324
window_time = 1
NSAMP = int(fs * window_time)
if __name__ == '__main__':
    # with np.load('40hz_lowpass_fs1k.npz') as data:
    # with np.load('40-55-333hz_lowpass.npz') as data:
    with np.load('566fs_lowpass.npz') as data:
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
    a_int = [0]*len(b_int)
    a_int[0] = 1

    # data = pandas.read_csv('data_inputs/1open-4reading.csv')
    # data = pandas.read_csv('data_inputs/1open-4reading-300.csv')
    data = pandas.read_csv('data_inputs/1open-1closed-1open-2closedmusicrelax.csv')
    volts = data['0.034611']

    data_demo = pandas.read_csv('data_inputs/demo.csv')
    demo_ch1 = data_demo['CH1 volts']
    demo_ch2 = data_demo['CH2 volts']
    print('starting')

    # CH1_power_filt_rx, CH1_power_filt_tx = multiprocessing.Pipe()
    # CH1_volts_rx, CH1_volts_tx = multiprocessing.Pipe()
    # CH1_v_rx, CH1_v_tx = multiprocessing.Pipe()
    # CH1_volts_filt_rx, CH1_volts_filt_tx = multiprocessing.Pipe()
    # CH1_filt_for_fft_rx, CH1_filt_for_fft_tx = multiprocessing.Pipe()
    # fft_out_rx, fft_out_tx = multiprocessing.Pipe()
    # csv_raw_rx, csv_raw_tx = multiprocessing.Pipe()

    CH1_rx, CH1_tx = [0,0]#multiprocessing.Pipe()
    CH2_rx, CH2_tx = [0,0]#multiprocessing.Pipe()

    CH1_to_filt_rx, CH1_to_filt_tx = multiprocessing.Pipe()
    CH1_filtered_rx, CH1_filtered_tx = multiprocessing.Pipe()

    CH2_to_filt_rx, CH2_to_filt_tx = multiprocessing.Pipe()
    CH2_filtered_rx, CH2_filtered_tx = multiprocessing.Pipe()

    CH1_for_power_rx, CH1_for_power_tx = multiprocessing.Pipe()
    CH2_for_power_rx, CH2_for_power_tx = multiprocessing.Pipe()

    power_calc_rx, power_calc_tx = multiprocessing.Pipe()
    power_calc2_rx, power_calc2_tx = multiprocessing.Pipe()

    power_filtered_rx, power_filtered_tx = multiprocessing.Pipe()
    power_filtered2_rx, power_filtered2_tx = multiprocessing.Pipe()

    CH1_volts_raw_csv_rx, CH1_volts_raw_csv_tx = multiprocessing.Pipe()
    CH2_volts_raw_csv_rx, CH2_volts_raw_csv_tx = multiprocessing.Pipe()
    CH1_filt_csv_rx, CH1_filt_csv_tx = multiprocessing.Pipe()
    CH2_filt_csv_rx, CH2_filt_csv_tx = multiprocessing.Pipe()

    powers_csv_rx, powers_csv_tx = [0,0]#multiprocessing.Pipe()

    timestamp = current_timestamp()
    save_path = f'./outputs/{timestamp}/'
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    with ProcessPoolExecutor(max_workers=None) as executor:

        save_volts = executor.submit(csv_save_volts,timestamp,fs,           CH1_volts_raw_csv_rx,CH2_volts_raw_csv_rx,CH1_filt_csv_rx,CH2_filt_csv_rx)
        print('.')
        # save_power = executor.submit(csv_save_power,timestamp,window_time,powers_csv_rx)

        graph_power = executor.submit(draw_power_bars,power_filtered_rx)
        graph_feedback = executor.submit(draw_feedback_bars,power_filtered2_rx)
        print('.')
        # graph_power = executor.submit(draw_power_bars,power_calc_rx)
        # graph_power = executor.submit(draw_power,power_filtered_rx)
        # graph_power = executor.submit(draw_power,power_calc_rx)
        time.sleep(4)
        print('.')
        graph_signals = executor.submit(draw_signals,CH1_filtered_rx,CH2_filtered_rx,fs,60)
        print('.')
        filter_data = executor.submit(filt_data,a_lowpass,b_lowpass,CH1_to_filt_rx,CH1_filtered_tx,CH1_for_power_tx,CH2_to_filt_rx,CH2_filtered_tx,CH2_for_power_tx,CH1_filt_csv_tx,CH2_filt_csv_tx)
        print('.')
        power_calc = executor.submit(calculate_powers,CH1_for_power_rx, CH2_for_power_rx, power_calc_tx, power_calc2_tx,1,fs,powers_csv_tx)
        print('.')
        filt_power = executor.submit(filt_and_interpolate,a_smooth,b_smooth,a_int,b_int,power_calc_rx,power_calc2_rx,power_filtered_tx,power_filtered2_tx)
        print('.')
        # eat = executor.submit(eat_pipe,CH1_rx,CH2_rx)#,powers_csv_rx,CH1_volts_raw_csv_rx,CH2_volts_raw_csv_rx,CH1_filt_csv_rx,CH2_filt_csv_rx)#,CH2_for_power_rx)#,power_calc_rx)


        time.sleep(5)
        print('.')
        # send_signals = executor.submit(read_UART,CH1_tx,CH1_to_filt_tx,CH2_tx,CH2_to_filt_tx,CH1_volts_raw_csv_tx,CH2_volts_raw_csv_tx)
        send_signals = executor.submit(send_data,demo_ch1,demo_ch2,CH1_tx,CH1_to_filt_tx,CH2_tx,CH2_to_filt_tx,CH1_volts_raw_csv_tx,CH2_volts_raw_csv_tx)
        


        # graphing_done = False
        # signals_done = False
        # power_done = False
        # while graphing_done == False:
        #     if graph_signals.done():
        #         eat2 = executor.submit(eat_pipe,CH1_filtered_rx,CH2_rx)
        #     if graph_power.done():
        #         eat3 = executor.submit(eat_pipe,power_calc_rx)
        #     if graph_signals.done() and graph_power.done():
        #         graphing_done = True
        # print('done graph')
        # time.sleep(10)
        

        # graph_signals = executor.submit(draw_signals,CH1_volts_filt_rx,CH1_v_rx,fs,60)

        fs_fft = fs
        # power_calc = executor.submit(calculate_powers,CH1_filt_for_fft_rx,fft_out_tx,1,fs_fft)
        # graph_bar_power = executor.submit(draw_power_bars,power_filt_rx)
        # graph_fft = executor.submit(draw_power,CH1_power_filt_rx)

        
        # time.sleep(2)
        
        # filter_data = executor.submit(filt_data,a_lowpass,b_lowpass,CH1_volts_rx,CH1_volts_filt_tx,CH1_filt_for_fft_tx)

        # send = executor.submit(send_data,volts,CH1_volts_tx,CH1_v_tx,csv_raw_tx)
        # send = executor.submit(read_UART,CH1_volts_tx,CH1_v_tx,csv_raw_tx)
        # uart = executor.submit(read_UART)
        # save_in_csv = executor.submit(csv_save,csv_raw_rx)
        # filt_power = executor.submit(filt_and_interpolate,a_smooth,b_smooth,1,b_int,fft_out_rx,CH1_power_filt_tx)