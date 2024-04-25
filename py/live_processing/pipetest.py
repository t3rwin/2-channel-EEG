from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, wait
import multiprocessing
import time

def send(p):
    a = list(range(20))
    for val in a:
        p.send(val)
    p.send('STOP')
    p.close()
    print('DONE!')

def rx(p):
    val = 0
    while val != 'STOP':
        time.sleep(1)
        while p.poll() is False:
            pass
        print(val)
        val = p.recv()
    return
if __name__ == '__main__':
    p_rx, p_tx = multiprocessing.Pipe()

    with ProcessPoolExecutor(max_workers=None) as executor:
        a = executor.submit(send,p_tx)
        b = executor.submit(rx,p_rx)
        # wait(b)
