import concurrent.futures
import os
import sys

def worker(x):
    # Simulate some work
    sum = 0
    for i in range(1000000):
        sum += i
    return x * x

def main():
    print(f"Running in {sys.executable}")
    workers = 20
    print(f"Testing with {workers} workers...")
    try:
        with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as executor:
            results = list(executor.map(worker, range(workers * 2)))
        print(f"Results length: {len(results)}")
        print("Success!")
    except Exception as e:
        print(f"Stress MP Failed: {e}")

if __name__ == '__main__':
    main()
