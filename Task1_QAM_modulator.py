import numpy as np
import matplotlib.pyplot as plt

def gen_data(N, data_sync=0):
   if data_sync == 0:
      data_sync_osc = np.ones(176, dtype=int) #array
      data_sync_symb = [0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1]#16-bit pattern
      data_sync = np.concatenate((data_sync_osc, data_sync_symb), axis=None)
   data_r = np.random.randint(2, size=N - len(data_sync))
   data = np.concatenate((data_sync, data_r))
   return data

def slicer(data):
   dataI = data[slice(0, len(data), 2)]
   dataQ = data[slice(1, len(data), 2)]
   return dataI, dataQ

def mapper_16QAM(QAM16, data):
   map_indices = 2 * data[:-1:2] + data[1::2]
   dataMapped = np.take(QAM16, map_indices)
   return dataMapped

def chnl_AWGN(sig, SNR, K):#channel
   sig_abs2 = [abs(s)**2 for s in sig]
   P = (K * sum(sig_abs2)) / len(sig_abs2)
   gamma = 10**(SNR/10)#gamma
   N0 = P / gamma
   n = np.sqrt(N0 / 2) * np.random.standard_normal(len(sig))
   sig_n = sig + n
   return sig_n


def demapper(symbols_I, symbols_Q, packetSize, threshold = 3.0):
   Ns = packetSize // 4
   bits_I = []
   bits_Q = []
   for i in range(Ns):
      if symbols_I[i] >= 0 and symbols_I[i] <= threshold:
         bits_I.append(1)
         bits_I.append(0)

      if symbols_I[i] > threshold:
         bits_I.append(1)
         bits_I.append(1)

      if symbols_I[i] < 0 and symbols_I[i] >= -threshold:
         bits_I.append(0)
         bits_I.append(1)

      if symbols_I[i] < -threshold:
         bits_I.append(0)
         bits_I.append(0)

      if symbols_Q[i] >= 0 and symbols_Q[i] <= threshold:
         bits_Q.append(1)
         bits_Q.append(0)

      if symbols_Q[i] > threshold:
         bits_Q.append(1)
         bits_Q.append(1)

      if symbols_Q[i] < 0 and symbols_Q[i] >= -threshold:
         bits_Q.append(0)
         bits_Q.append(1)

      if symbols_Q[i] < -threshold:
         bits_Q.append(0)
         bits_Q.append(0)

   bits_I = np.array(bits_I, dtype=int)
   bits_Q = np.array(bits_Q, dtype=int)

   bitStream = np.zeros(packetSize, dtype=int)

   bitStream[::2] = bits_I
   bitStream[1::2] = bits_Q

   return bitStream

path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(path)
import example as qam

def SNR_BER_analysis():
   """Analyze the BER for each SNR value."""
   plt.close('all')

   n_tests_per_snr = 50
   max_snr = 20

   SNR_values = np.arange(1, max_snr + 1)
   BER_mean_acc = []

   for SNR in SNR_values:
      print("SNR:", SNR)
      BER = np.zeros(n_tests_per_snr)

      for j in range(len(BER)):
            BER[j] = qam.QAMsys(SNR, 0)

      BER_mean = np.mean(BER)
      BER_mean_acc.append(BER_mean)

   plt.figure(1)
   plt.scatter(SNR_values, BER_mean_acc)
   plt.xscale('linear')
   plt.yscale('log')
   plt.xlabel('SNR (dB)')
   plt.ylabel('BER')
   plt.grid()

   plt.show()
