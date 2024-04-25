import matplotlib.pyplot as plt
import numpy as np

import matplotlib.animation as animation
from matplotlib.lines import Line2D
from scipy import fft

class Bar:
    def __init__(self, ax1, maxt=1, dt=.5):
        self.ax1 = ax1
        self.dt = dt
        self.maxt = maxt
        self.tdata = [0]
        self.tdata_export = [0]
        self.ydata = [0]
        self.line = Line2D(self.tdata, self.ydata,marker='.')
        self.ax1.add_line(self.line)
        self.ax1.set_ylim(0, 1)
        self.ax1.set_xlim(0, self.maxt)
        self.b = self.ax1.bar(0,0)

    def update(self, y):
        self.ax1.set_xlim(self.tdata[0], self.tdata[0] + self.maxt)
        self.ax1.figure.canvas.draw()
        self.b.set_height(y)

        # This slightly more complex calculation avoids floating-point issues
        # from just repeatedly adding `self.dt` to the previous value.
        # t = self.tdata[0] + len(self.tdata) * self.dt
        
        # self.line.set_data(self.tdata, self.ydata)
        # self.line.set_data(self.tdata, self.ydata)
        return self.b,

class Power:
    def __init__(self, ax1, maxt=30, dt=.25): #.125
        self.ax1 = ax1
        self.dt = dt
        self.maxt = maxt
        self.tdata = [0]
        self.tdata_export = [0]
        self.ydata = [0]
        self.line = Line2D(self.tdata, self.ydata,marker='.',linewidth=.5)
        self.ax1.add_line(self.line)
        self.ax1.set_ylim(0, 1)
        self.ax1.set_xlim(0, self.maxt)

        self.ydata2 = [0]
        self.line2 = Line2D(self.tdata,self.ydata2,marker='+',color='black')
        self.ax1.add_line(self.line2)

    def update(self, y):
        # y2=.6
        lastt = self.tdata[-1]
        if lastt >= self.tdata[0] + self.maxt:  # reset the arrays
            self.tdata_export.append(self.tdata)
            self.tdata = [self.tdata[-1]]
            self.ydata = [self.ydata[-1]]
            self.ax1.set_xlim(self.tdata[0], self.tdata[0] + self.maxt)

            self.ydata2 = [self.ydata2[-1]]

            self.ax1.figure.canvas.draw()

        # This slightly more complex calculation avoids floating-point issues
        # from just repeatedly adding `self.dt` to the previous value.
        t = self.tdata[0] + len(self.tdata) * self.dt
            
        self.tdata.append(t)
        self.ydata.append(y[0])
        self.ydata2.append(y[1])
        
        # self.line.set_data(self.tdata, self.ydata)
        self.line.set_data(self.tdata, self.ydata)
        self.line2.set_data(self.tdata, self.ydata2)
        return self.line,self.line2

class Scope:
    def __init__(self, ax1, ax2=0, maxt=2, dt=.003):
        self.ax1 = ax1
        # self.ax2 = ax2
        self.dt = dt
        self.maxt = maxt
        self.tdata = [0]
        self.tdata_export = [0]
        self.ydata = [0]
        self.ydata2 = [0]
        self.line = Line2D(self.tdata, self.ydata)
        # self.line.set_antialiased(True)
        self.line2 = Line2D(self.tdata, self.ydata2)
        self.ax1.add_line(self.line2)
        # self.ax2.add_line(self.line2)
        self.ax1.set_ylim(-1.65, 1.65)
        # self.ax1.set_ylim(0, 3.5)
        # self.ax2.set_ylim(-1.65, 1.65)
        self.ax1.set_xlim(0, self.maxt)
        # self.ax2.set_xlim(0, self.maxt)

    def update(self, y):
        # y2=.6
        lastt = self.tdata[-1]
        if lastt >= self.tdata[0] + self.maxt:  # reset the arrays
            self.tdata_export.append(self.tdata)
            self.tdata = [self.tdata[-1]]
            self.ydata = [self.ydata[-1]]
            self.ydata2 = [self.ydata2[-1]]
            self.ax1.set_xlim(self.tdata[0], self.tdata[0] + self.maxt)
            self.ax1.figure.canvas.draw()
            # self.ax2.set_xlim(self.tdata[0], self.tdata[0] + self.maxt)
            # self.ax2.figure.canvas.draw()

        # This slightly more complex calculation avoids floating-point issues
        # from just repeatedly adding `self.dt` to the previous value.
        for i in range(len(y[0])):
            t = self.tdata[0] + len(self.tdata) * self.dt
            self.tdata.append(t)

        # self.ydata.append(y[0])
        # self.ydata2.append(y[1])

        # self.ydata = self.ydata + y[0]
        self.ydata2 = self.ydata2 + y[1]
        
        # self.line.set_data(self.tdata, self.ydata)
        self.line2.set_data(self.tdata, self.ydata2)

        # self.line.set_label("CH1")
        # self.ax.figure.legend()
        # self.ax1.set_title("Unfiltered")
        # self.ax2.set_title("Filtered")

        # return self.line2,self.line
        return self.line2,

class FFT_Display:
    def __init__(self, ax1, nsamp, fs):
        self.ax1 = ax1
        self.fs = fs
        self.nsamp = nsamp
        self.tdata = [0]
        self.tdata_export = [0]
        self.ydata = [0]
        self.line = Line2D(self.tdata, self.ydata, marker='.',linestyle='-.')
        # self.line.set_antialiased(True)
        self.ax1.add_line(self.line)
        self.ax1.set_ylim(0, 1)
        self.ax1.set_xlim(0, self.nsamp)

    def update(self, y):
        # y2=.6
        # lastt = self.tdata[-1]
        # if lastt >= self.tdata[0] + self.nsamp:  # reset the arrays
            # self.tdata_export.append(self.tdata)
            # self.tdata = [self.tdata[-1]]
            # self.ydata = [self.ydata[-1]]
        F = np.array(fft.fftfreq(self.nsamp))
        F = self.fs*F[0:int((len(F)+1)/2)]
        index = (np.abs(F - 70)).argmin()
        self.ax1.set_xlim(0, index)
        self.ax1.set_xticks(F[0:index:10])
        self.ax1.tick_params(axis='x', labelrotation=90)
        self.ax1.set_xlabel('f(Hz)')
        # print(f'index:{index}')
        # print(f'freq:{F[index-3:index]}')
        # print(f'F:{F}')

        # self.ax1.figure.canvas.draw()

        # This slightly more complex calculation avoids floating-point issues
        # from just repeatedly adding `self.dt` to the previous value.
        # t = self.tdata[0] + len(self.tdata) * self.dt
        # t = list(range(0,self.nsamp))

        # self.tdata.append(t)
        # self.ydata.append(y)
        # data = y[0:index]
        data = np.array(y)
        # print(len(data))
        # print(index)
        # self.line.set_data(list(range(0,len(data))),data)
        self.line.set_data(F,data/max(y))


        # self.line.set_label("CH1")
        # self.ax.figure.legend()
        # self.ax1.set_title("Unfiltered")
        # self.ax2.set_title("Filtered")

        return self.line,


# def emitter(p=0.1):
#     """Return a random value in [0, 1) with probability p, else 0."""
#     while True:
#         v = np.random.rand()
#         if v > p:
#             yield 0.
#         else:
#             yield np.random.rand()


# Fixing random state for reproducibility
# np.random.seed(19680801 // 10)


# fig, ax = plt.subplots()
# scope = Scope(ax)

# # pass a generator in "emitter" to produce data for the update func
# ani = animation.FuncAnimation(fig, scope.update, emitter,
#                               blit=True, save_count=100)

# plt.show()