import time

def func(args): 
    x = args[0]  
    y = args[1] 
    time.sleep(1) 
    return x + y


def run_pool():  # main process
    from multiprocessing import Pool

    cpu_worker_num = 3
    process_args = [(1, 1), (9, 9), (4, 4), (3, 3), ]

    print(f'| inputs:  {process_args}')
    start_time = time.time()
    with Pool(cpu_worker_num) as p:
        outputs = p.map(func2, process_args)
    print(f'| outputs: {outputs}    TimeUsed: {time.time() - start_time:.1f}    \n')

if __name__ =='__main__':
    run_pool()
