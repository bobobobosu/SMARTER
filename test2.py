import multiprocessing
from time import sleep


my_global = None


def test():
    global my_global
    read_only_secs = 3
    while read_only_secs > 0:
        sleep(1)
        print(f'child proc global: {my_global} at {hex(id(my_global))}')
        read_only_secs -= 1
    print('child proc writing to copy-on-write...')
    my_global = 'something else'
    while True:
        sleep(1)
        print(f'child proc global: {my_global} at {hex(id(my_global))}')


def set_func():
    global my_global
    my_global = [{'hi': 1, 'bye': 'foo'}]

if __name__ == "__main__":
    print(f'main proc global: {my_global} at {hex(id(my_global))}')
    set_func()
    print(f'main proc global: {my_global} at {hex(id(my_global))}')
    p1 = multiprocessing.Process(target=test)
    p1.start()

    while True:
        sleep(1)
        print(f'main proc global: {my_global} at {hex(id(my_global))}')