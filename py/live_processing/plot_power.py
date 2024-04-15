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

def send_data(volts,q_volts,q_v):
    for v in volts:
        q_volts.put(v)
        q_v.put(v)
        # print(f'(v:{v})')
        time.sleep(1e-3)
    q_volts.put(9)
    q_v.put(9)

def filter(data,a,b,z):
    return scipy.signal.lfilter(b,a,data,zi=z)

def filt_data(a,b,q_volts,q_volts_filt):
    result = []
    z = [0]*(len(b)-1)#np.zeros(len(b)-1)
    x = 0
    while x != 9:
    # while q_volts.empty() is False:
        # t0 = time.time()
        # if q_volts.empty() is False:
        val,z = filter([x],a,b,z)
        result.append(val)
        q_volts_filt.put(val)
            # print(f'(v_filt:{val})')
        # q_volts_filt.put(x)
        x = q_volts.get()
    q_volts_filt.put(9)

        # t1 = time.time()

# def filter_bysample(data,a,b):
#     result = []
#     z = np.zeros(len(b)-1)
#     for i, x in enumerate(data):
#         # t0 = time.time()
#         time.sleep(1e-3)
#         val,z = filter([x],a,b,z)
#         result.append(val)
#         # t1 = time.time()
#     # print(t1-t0)
#     return result
# i=0
def emitter():
    global i
    val = volts_filt[i]
    i += 1
    yield [val,.5*(val-.6)]

def emitter2():
    global q_volts
    global q_volts_filt
    # yield [q_volts.get(),q_volts_filt.get()]
    yield [q_volts.get(),q_volts.get()]
# filt = 0
# def emitter3():
#         val = q_volts.get()
#         # filt = q_volts_filt.get()
#         # print(filt)
#         yield [val,val]
# #         # yield [q_volts.get(),q_volts.get()]
def anim(q_volts,q_volts_filt,q_v):
    # global filt
    def emitter3():
        if q_volts_filt.empty() is False:
            filt_val = q_volts_filt.get()
            if filt_val != 9:
                r2 = filt_val[0]
            else:
                r2 = 0
        else:
            r2 = 0
        if q_v.empty() is False:
            val = q_v.get()#q_volts.get()
            if val != 9:
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
                                blit=True,cache_frame_data=False, save_count=0,interval=3.5)
    plt.show()
    # print(scope.tdata_export)
    # df = pandas.DataFrame(scope.tdata_export)
    # df.to_csv('time.csv',index=False)
    # t1 = time.time()
    # print(t1-t0)

    # print(q_volts.get())

if __name__ == '__main__':
    # fs = 1000
    fs = 333

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
    volts = volts[0:(33300)]
    volts_filt = 0#scipy.signal.lfilter(b_lowpass,a_lowpass,volts)

    q_volts = multiprocessing.Queue()
    q_volts_filt = multiprocessing.Queue()
    q_v = multiprocessing.Queue()

    print('starting')
    # with concurrent.futures.ProcessPoolExecutor() as executor:
    #     send = executor.submit(send_data,volts,q_volts)
    #     # filt = executor.submit(filter_bysample(a_lowpass,b_lowpass,q_volts,q_volts_filt))
    #     # plotting = executor.submit(anim,q_volts,q_volts_filt)
    send = multiprocessing.Process(target=send_data,args=(volts,q_volts,q_v))
    filt = multiprocessing.Process(target=filt_data,args=(a_lowpass,b_lowpass,q_volts,q_volts_filt))
    graph = multiprocessing.Process(target=anim,args=(q_volts,q_volts_filt,q_v))
    t0 = time.time()
    send.start()
    filt.start()
    graph.start()

        # pass a generator in "emitter" to produce data for the update func
    # fig, (ax1,ax2) = plt.subplots(2,1)
    # scope = Scope(ax1,ax2)
    # ani = animation.FuncAnimation(fig, scope.update, frames=emitter3,
    #                             blit=True, save_count=100,interval=1)
    # plt.draw()
    # x = q_volts.get()
    # while x != None:
    #     y = x
    #     x = q_volts.get()
    
    send.join()
    # print('done send')
    filt.join()
    # print('done filt')
    graph.join()
    print('done graph')
    # filt.join()
    

    
    # x = q_volts.get()
    # while True:
    #     y = 1
    #     x = q_volts.get()
    # graph.join()
    # print(y)

    # print(q_volts_filt.get())
    # graph.join()
    

    # while x != 6969:
    #     if q_volts.empty() is False:
    #         # print(x)
    #         # x = q_volts.get()
    #         a = 0


    # pass a generator in "emitter" to produce data for the update func
    # ani = animation.FuncAnimation(fig, scope.update, frames=emitter2,
    #                             blit=True, save_count=100,interval=100)
    # plt.show()



    # ax1 = plt.subplot(211)
    # # ax1.plot(volts[1:fs*2])
    # plt.plot(volts_filt[0:fs*2])

    # ax2 = plt.subplot(212, sharex=ax1)
    # plt.plot(filt_live[0:fs*2])
    # # plt.plot(volts_filt[1:fs*2])

    # plt.show()