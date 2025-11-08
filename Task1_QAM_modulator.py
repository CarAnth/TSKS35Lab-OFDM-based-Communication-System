import numpy as np
import matplotlib.pyplot as plt

N=np.randint(192,1024)
########BACKUP CODE FOR CHANGES###############
#my_sync = [1,0,1,0] * 48  #sample pattern (lenght: 192)
#L = len(my_sync)
#N = int(np.random.randint(L, L + 1000 + 1))  #between [L, L+1000] 
#x = data_gen(N, data_sync=my_sync)
##############################################
N = int(np.random.randint(192, 1200+1))  # [192, 1200] random

def gen_data(N, data_sync=0):
   if data_sync == 0:
      data_sync_osc = np.ones(176, dtype=int)
      data_sync_symb = [0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1]
      data_sync = np.concatenate((data_sync_osc, data_sync_symb), axis=None)
   data_r = np.random.randint(2, size=N - len(data_sync))
   data = np.concatenate((data_sync, data_r))
   return data


####Additive White Gaussian Noise#####
def chnl_AWGN(sig, SNR, K):#channel
   sig_abs2 = [abs(s)**2 for s in sig]
   P = (K * sum(sig_abs2)) / len(sig_abs2)
   gamma = 10**(SNR/10)#gamma
   N0 = P / gamma
   n = np.sqrt(N0 / 2) * np.random.standard_normal(len(sig))
   sig_n = sig + n
   return sig_n


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

