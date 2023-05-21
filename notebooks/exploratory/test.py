import multiprocessing as mp

def my_function(arg):
    return arg

if __name__ == '__main__':
    # Create a pool of worker processes
    pool = mp.Pool()

    # Prepare your input data (e.g., a list)
    data = list(range(1000))

    # Submit tasks to the pool in a for loop
    results = []
    for item in data:
        result = pool.apply_async(my_function, args=(item,))
        results.append(result)

    # Get the results from the async tasks
    output = [result.get() for result in results]
    print(output)

    # Close the pool
    pool.close()
    pool.join()





