'''
A bandage script to kill python.exe files that got hung up by cuda issue to prevent OOM on the CPU
'''

import psutil
import time

def kill_small_python_processes():
    for proc in psutil.process_iter(['pid', 'name', 'memory_info']):
        try:
            if proc.info['name'] == "python.exe":
                working_set_size_kb = proc.info['memory_info'].rss / 1024  # Convert bytes to KB
                if working_set_size_kb < 100 and working_set_size_kb > 47:
                    print(f"Killing PID {proc.pid} with Working Set Size: {working_set_size_kb} KB")
                    proc.terminate()
                    # proc.wait(5)
                    print(f"Process {proc.pid} has been terminated.")
        except psutil.NoSuchProcess:
            print(f"No process found with PID {proc.pid}")
        except psutil.AccessDenied:
            print("Access denied to terminate the process")
        except Exception as e:
            print(f"An error occurred with PID {proc.pid}: {str(e)}")

if __name__ == "__main__":
    while True:
        kill_small_python_processes()
        print("Checking...")
        time.sleep(2)
        continue
