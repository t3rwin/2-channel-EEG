import serial.tools.list_ports

serialInst = serial.Serial()

#port intialization and defenition
def Port_Init():
    ports = serial.tools.list_ports.comports()
    portsList = []
    for onePort in ports:
        portsList.append(str(onePort))
        print(str(onePort))

    val = input("Select Port: COM")

    for x in range(0,len(portsList)):
        if portsList[x].startswith("COM" + str(val)):
            portVar = "COM" + str(val)
            print(portVar)

    serialInst.baudrate = 209700
    serialInst.port = portVar
    serialInst.open()

#input: array of 3 bytes [0xFF, 0xAA, 0x6C] -> output: integer 16,755,308
def byte2int(byte_list):
    concat_hex = 0
    for byte in byte_list:
        concat_hex = (concat_hex << 8) | int.from_bytes(byte, byteorder='big')
    return (concat_hex)

Port_Init()
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
        print(packet)
        print("{0:x}".format(byte2int(packet[:3])), byte2int(packet[:3]))

        #reset read value memory
        packet = []