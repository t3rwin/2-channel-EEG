from tkinter import *
import serial.tools.list_ports
import functools


#need to clear line graph not just write over
class LineGrpah:
    def __init__(self, initialNode):
        self.node = initialNode
        self.points = [[initialNode, 0]]

    def insertNode(self, number):
        for item in self.points:
            item[1] = item[1] + 1
        self.points.insert(0, [number, 0])
        if len(self.points) > 100:
            self.points.pop()

    def printNodes(self):
        for item in self.points:
            print(item)
        
lineGraph = LineGrpah(50)


ports = serial.tools.list_ports.comports()
serialObj = serial.Serial()

root = Tk()
root.config(bg='grey')

def initComPort(index):
    currentPort = str(ports[index])
    comPortVar = str(currentPort.split(' ')[0])
    print(comPortVar)
    serialObj.port = comPortVar
    serialObj.baudrate = 11600
    serialObj.open()

for onePort in ports:
    comButton = Button(root, text=onePort, font=('Calibri', '13'), height=1, width=45, command = functools.partial(initComPort, index = ports.index(onePort)))
    comButton.grid(row=ports.index(onePort), column=0)

dataCanvas = Canvas(root, width=600, height=400, bg='white')
dataCanvas.grid(row=0, column=1, rowspan=100)

vsb = Scrollbar(root, orient='vertical', command=dataCanvas.yview)
vsb.grid(row=0, column=2, rowspan=100, sticky='ns')

dataCanvas.config(yscrollcommand = vsb.set)

dataFrame = Frame(dataCanvas, bg="white")
dataCanvas.create_window((10,0),window=dataFrame,anchor='nw')

def drawLineGraph():
    #dataCanvas.create_rectangle(0,0,600,400,fill="white")
    dataCanvas.delete("all")
    for i in range(len(lineGraph.points) - 1):
        dataCanvas.create_line(lineGraph.points[i][1] * 5,lineGraph.points[i][0], lineGraph.points[i + 1][1] * 5,lineGraph.points[i + 1][0])

def checkSerialPort():
    if serialObj.isOpen() and serialObj.in_waiting:
        recentPacket = serialObj.readline()
        recentPacketString = recentPacket.decode('utf').rstrip('\n')
        lineGraph.insertNode(int(recentPacketString))
        Label(dataFrame, text=recentPacketString)
        drawLineGraph()

while True:
    root.update()
    checkSerialPort()
    dataCanvas.config(scrollregion=dataCanvas.bbox("all"))
