import numpy as np
import matplotlib.pyplot as plt

N=np.randint(192,1024)

def generate_data(N,data_sync = 0):
    if data_sync == 0:
      data_sync_osc = np.ones(176, dtype=int)
      data_sync_symb = [0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1]
      data_sync = np.concatenate((data_sync_osc, data_sync_symb), axis=None)
    assert N >= len(data_sync), "N must be at least as large as the length of data_sync"
    data_r = np.random.randint(2, size=N - len(data_sync))
    data = np.concatenate((data_sync, data_r))
    return data

def plot_data(data, n_samples=300):
    n = min(len(data), n_samples)
    plt.figure(figsize=(10, 3))
    plt.step(np.arange(n), data[:n], where='post')
    plt.ylim(-0.2, 1.2)
    plt.xlabel('Sample index')
    plt.ylabel('Bit value')
    plt.title(f'Generated data (first {n} samples, total N={len(data)})')
    plt.grid(True)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    data = generate_data(N)
    plot_data(data, n_samples=300)
