# task3_ofdm.py
import numpy as np
import matplotlib.pyplot as plt

from Task1_QAM_modulator import mapper_16QAM, demapper_16QAM


class CPOFDM:
    """
    Simple CP-OFDM transmitter and receiver.
    """
    def __init__(self,
                 Nfft: int = 128,
                 cp_len: int = 32,
                 fs: float = 25.6e6,
                 data_subcarriers: int | None = None):
        self.Nfft = int(Nfft)
        self.cp_len = int(cp_len)
        self.fs = float(fs)

        if data_subcarriers is None:
            data_subcarriers = int(0.75 * self.Nfft)

        self.num_data_subcarriers = int(data_subcarriers)

        guard_total = self.Nfft - self.num_data_subcarriers
        if guard_total < 0:
            raise ValueError("data_subcarriers cannot exceed Nfft")

        guard_left = guard_total // 2
        start_idx = guard_left
        end_idx = start_idx + self.num_data_subcarriers
        self.data_idx = np.arange(start_idx, end_idx, dtype=int)

    def subcarrier_spacing(self) -> float:
        return self.fs / self.Nfft

    def ofdm_symbol_duration(self) -> float:
        return (self.Nfft + self.cp_len) / self.fs

    def bit_rate(self, bits_per_symbol: int) -> float:
        T_ofdm = self.ofdm_symbol_duration()
        Rb = bits_per_symbol * self.num_data_subcarriers / T_ofdm
        return Rb

    def modulate(self, qam_symbols: np.ndarray) -> tuple[np.ndarray, int]:
        s = np.asarray(qam_symbols, dtype=complex).ravel()
        L = len(s)
        if L == 0:
            return np.array([], dtype=complex), 0

        symbols_per_ofdm = self.num_data_subcarriers
        num_ofdm_symbols = (L + symbols_per_ofdm - 1) // symbols_per_ofdm

        L_pad = num_ofdm_symbols * symbols_per_ofdm
        s_padded = np.zeros(L_pad, dtype=complex)
        s_padded[:L] = s

        Smat = s_padded.reshape(num_ofdm_symbols, symbols_per_ofdm)

        tx_blocks = []
        for k in range(num_ofdm_symbols):
            Xk = np.zeros(self.Nfft, dtype=complex)
            Xk[self.data_idx] = Smat[k, :]

            x = np.fft.ifft(Xk)
            cp = x[-self.cp_len:]
            x_cp = np.concatenate([cp, x])
            tx_blocks.append(x_cp)

        tx_signal = np.concatenate(tx_blocks)
        return tx_signal, num_ofdm_symbols

    def demodulate(self, rx_signal: np.ndarray) -> np.ndarray:
        r = np.asarray(rx_signal, dtype=complex).ravel()
        if len(r) == 0:
            return np.array([], dtype=complex)

        ofdm_block_len = self.Nfft + self.cp_len
        num_ofdm_symbols = len(r) // ofdm_block_len
        if num_ofdm_symbols == 0:
            return np.array([], dtype=complex)

        r = r[:num_ofdm_symbols * ofdm_block_len]
        Rmat = r.reshape(num_ofdm_symbols, ofdm_block_len)

        qam_out = []
        for k in range(num_ofdm_symbols):
            x_no_cp = Rmat[k, self.cp_len:]
            X_hat = np.fft.fft(x_no_cp)
            qam_k = X_hat[self.data_idx]
            qam_out.append(qam_k)

        qam_hat = np.concatenate(qam_out)
        return qam_hat

    def compute_spectrum(self, tx_signal: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        x = np.asarray(tx_signal, dtype=complex).ravel()
        if len(x) == 0:
            return np.array([]), np.array([])

        N = len(x)
        X = np.fft.fft(x)
        half = N // 2
        X_half = X[:half]
        mag = np.abs(X_half)
        mag /= np.max(mag)

        f_axis = np.linspace(0.0, self.fs / 2.0, half, endpoint=False)
        return f_axis, mag

    def compute_envelope(self, tx_signal: np.ndarray) -> np.ndarray:
        x = np.asarray(tx_signal, dtype=complex).ravel()
        return np.abs(x)


if __name__ == "__main__":
    ofdm = CPOFDM(Nfft=128, cp_len=32, fs=25.6e6)
    bits_per_symbol = 4  # 16-QAM

    print("Subcarrier spacing [Hz]:", ofdm.subcarrier_spacing())
    print("OFDM symbol duration [s]:", ofdm.ofdm_symbol_duration())
    print("Bit rate [bit/s] (approx):", ofdm.bit_rate(bits_per_symbol))

    # 1) Info bits
    num_symbols = 256
    num_bits = num_symbols * bits_per_symbol
    bits_tx = np.random.randint(0, 2, num_bits)

    # 2) Task-1 mapper (imported)
    qam_tx = mapper_16QAM(bits_tx)

    # 3) OFDM TX
    tx_signal, num_ofdm = ofdm.modulate(qam_tx)
    print("Number of OFDM symbols:", num_ofdm)
    print("TX signal length:", len(tx_signal))

    # Ideal channel
    rx_signal = tx_signal.copy()

    # 4) OFDM RX
    qam_rx = ofdm.demodulate(rx_signal)

    # 5) Task-1 demapper (imported)
    qam_rx_trunc = qam_rx[:num_symbols]
    bits_rx = demapper_16QAM(qam_rx_trunc)

    ber = np.mean(bits_rx != bits_tx)
    print("BER (ideal channel, Task-1 + Task-3):", ber)

    # Plots for Task-3
    env = ofdm.compute_envelope(tx_signal)
    plt.figure()
    plt.plot(env)
    plt.xlabel("Sample index")
    plt.ylabel("|x[n]|")
    plt.title("OFDM time-domain envelope")
    plt.grid(True)

    f_axis, mag = ofdm.compute_spectrum(tx_signal)
    plt.figure()
    plt.plot(f_axis / 1e6, 20 * np.log10(mag + 1e-12))
    plt.xlabel("Frequency [MHz]")
    plt.ylabel("Magnitude [dB] (normalized)")
    plt.title("OFDM magnitude spectrum")
    plt.grid(True)

    plt.show()
