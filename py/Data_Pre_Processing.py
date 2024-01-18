import csv
import numpy as np
import matplotlib.pyplot as plt
processed_data = [] #processed data = [id, sample num, [raw data]]

def extract_raw_data(array):
    result = array[8].split(', ')
    result[0] = 0
    result[len(result) - 1] = 0
    r1 = []
    for item in result:
        r1.append(float(item))
    return r1

def extract_id(array): #array should be each "line" with all 13 fields
    return int(array[1])

def extract_sample_num(array): #array should be each "line" with all 13 fields
    return int(array[0])

def extract_signal_quality(array):
    return int(array[9])

def print_FFT(values):
    n = len(values)
    fhat = np.fft.fft(values,n)
    cPSD = fhat*np.conj(fhat) / n
    freq = (1/((1/512)*n)) * np.arange(n)
    L = np.arange(1,np.floor(n/2), dtype = 'int')

    plt.plot(freq[L], cPSD[L], color = 'c', linewidth = 0.5)
    plt.xlim(0,30)
    plt.show()

with open('C:/Users/sebtu/Documents/Senior Project/test-file/EEG DATA/eeg-data.csv','r') as csv_file:
    csv_reader = csv.reader(csv_file)
    rows = list(csv_reader)
    for row in rows:
        temp = []
        if extract_signal_quality(row) < 150:
            temp.append(extract_id(row))
            temp.append(extract_sample_num(row))
            temp.append(extract_raw_data(row))
            processed_data.append(temp)

    print(len(processed_data[0][2]))
    #b = extract_raw_data(rows[612])

    #t = np.linspace(0,1,512)

    #plt.plot(t,b, color = 'c', linewidth = 0.5, label = 'data')
    #plt.show()
    #print_FFT(extract_raw_data(rows[612]))
