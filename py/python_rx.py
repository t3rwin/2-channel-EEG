import serial.tools.list_ports


#port intialization and defenition
def Port_Init(serialInst):
    ports = serial.tools.list_ports.comports()
    portsList = []
    for onePort in ports:
        portsList.append(str(onePort))
        print(str(onePort))

    # val = input("Select Port: COM")

    # portVar = "/dev/cu.usbmodem1203"
    portVar = "/dev/cu.usbmodem1403"
    # for x in range(0,len(portsList)):
    #     if portsList[x].startswith("COM" + str(val)):
    #         portVar = "COM" + str(val)
    #         print(portVar)

    serialInst.baudrate = 209700
    serialInst.port = portVar
    serialInst.open()
    print('opened')

#input: array of 3 bytes [0xFF, 0xAA, 0x6C] -> output: integer 16,755,308
def byte2int(byte_list):
    concat_hex = 0
    for byte in byte_list:
        concat_hex = (concat_hex << 8) | int.from_bytes(byte, byteorder='big')
    return (concat_hex)

if __name__ == '__main__':
    serialInst = serial.Serial()
    Port_Init(serialInst)
    packet=[]
    #constant read loop
    while True:
        #check if data in UART read register
        if (serialInst.inWaiting() > 0): 
            #add to data read
            packet.append(serialInst.read())

            #once full number is recieved print the value in: original 3 byte array, 
            #concatinated 6 digit hex, and decimal values
        if (len(packet) == 4):
            # print(packet)
            # print("{0:x}".format(byte2int(packet[:3])), byte2int(packet[:3]))
            # print(byte2int(packet[:3]))
            # print(int.from_bytes(packet[3], byteorder='big'))
            val = byte2int(packet[1::])
            channel = int.from_bytes(packet[0], byteorder='big')
            # channel = 0
            volts = (val*2.4)/8388608
            # print(channel, volts)

            #reset read value memory
            packet = []