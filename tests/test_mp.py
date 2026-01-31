import concurrent.futures
import os
import sys

def worker(x):
    return x * x

def main():
    print(f"Running in {sys.executable}")
    print(f"CPU count: {os.cpu_count()}")
    try:
        with concurrent.futures.ProcessPoolExecutor(max_workers=2) as executor:
            results = list(executor.map(worker, [1, 2, 3]))
        print(f"Results: {results}")
    except Exception as e:
        print(f"MP Failed: {e}")

if __name__ == '__main__':
    main()
