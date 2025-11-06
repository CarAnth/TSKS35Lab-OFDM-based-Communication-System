import numpy as np

def generate_data(N,data_sync = 0):
     if data_sync == 0:
      data_sync_osc = np.ones(176, dtype=int)
      data_sync_symb = [0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1]
      data_sync = np.concatenate((data_sync_osc, data_sync_symb), axis=None)
   data_r = np.random.randint(2, size=N - len(data_sync))
   data = np.concatenate((data_sync, data_r))
   return data
