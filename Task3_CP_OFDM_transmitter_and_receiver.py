# task3_ofdm.py
import numpy as np
import matplotlib.pyplot as plt

from Task1_QAM_modulator import (
    gen_data, mapper_16QAM, demapper_16QAM,
    mapper_4QAM,  demapper_4QAM
)


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

        # Data subcarriers as a contiguous block in the middle (baseband ordering)
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

    def build_ifft_input(self, qam_block: np.ndarray) -> np.ndarray:
        s = np.asarray(qam_block, dtype=complex).ravel()
        if len(s) != self.num_data_subcarriers:
            raise ValueError(
                f"Expected {self.num_data_subcarriers} QAM symbols, got {len(s)}"
            )

        X = np.zeros(self.Nfft, dtype=complex)
        X[self.data_idx] = s
        return X

    def plot_ifft_input_magnitude(self, qam_block: np.ndarray) -> None:
        """
        Plot |X[k]| for a single OFDM symbol with fftshift,
        so that DC (and the unused/guard region) is in the middle.
        """
        X = self.build_ifft_input(qam_block)
        Xs = np.fft.fftshift(X)
        mag = np.abs(Xs)

        k = np.arange(-self.Nfft // 2, self.Nfft // 2)
        plt.figure()
        plt.stem(k, mag, basefmt=" ")
        plt.xlabel("Subcarrier index (shifted)")
        plt.ylabel("|X[k]|")
        plt.title("Magnitude of IFFT input X[k] (guards centered)")
        plt.grid(True)

    def compute_spectrum(self, tx_signal: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Compute baseband spectrum of the transmitted OFDM signal.

        Uses fftshift so the OFDM band is centered at 0 Hz and
        the non-active subcarriers around DC appear in the middle.
        """
        x = np.asarray(tx_signal, dtype=complex).ravel()
        if len(x) == 0:
            return np.array([]), np.array([])

        N = len(x)
        X = np.fft.fft(x)
        Xs = np.fft.fftshift(X)

        mag = np.abs(Xs)
        mag /= np.max(mag)  # normalize

        f_axis = np.linspace(-self.fs / 2.0, self.fs / 2.0, N, endpoint=False)
        return f_axis, mag

    def compute_envelope(self, tx_signal: np.ndarray) -> np.ndarray:
        x = np.asarray(tx_signal, dtype=complex).ravel()
        return np.abs(x)
    def remove_cp(self, tx_signal: np.ndarray) -> np.ndarray:
        """
        Remove cyclic prefix from a serialized OFDM signal.

        Returns only the useful Nfft samples for each OFDM symbol.
        """
        x = np.asarray(tx_signal, dtype=complex).ravel()
        ofdm_block_len = self.Nfft + self.cp_len
        num_ofdm = len(x) // ofdm_block_len
        if num_ofdm == 0:
            return np.array([], dtype=complex)

        x = x[:num_ofdm * ofdm_block_len]
        Xmat = x.reshape(num_ofdm, ofdm_block_len)

        # Drop CP: keep only last Nfft samples of each block
        x_no_cp = Xmat[:, self.cp_len:]      # shape: (num_ofdm, Nfft)
        return x_no_cp.ravel()
    def lowpass_filter(self, x: np.ndarray, L: int = 8) -> np.ndarray:
        """
        Very simple low-pass FIR filter: moving average of length L.
        This is enough to demonstrate 'filtering' effect on the OFDM spectrum.
        """
        x = np.asarray(x, dtype=complex).ravel()
        if L <= 1:
            return x

        h = np.ones(L) / L     # boxcar FIR
        y = np.convolve(x, h, mode="same")
        return y
    def compute_spectrum_no_cp(
        self,
        tx_signal: np.ndarray,
        apply_filter: bool = False,
        L_filter: int = 8
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Compute baseband spectrum of the OFDM signal AFTER CP removal.
        Optionally applies a simple low-pass filter to the CP-free signal.
        """
        x_nc = self.remove_cp(tx_signal)   # CP removed
        if len(x_nc) == 0:
            return np.array([]), np.array([])

        if apply_filter:
            x_nc = self.lowpass_filter(x_nc, L=L_filter)

        N = len(x_nc)
        X = np.fft.fft(x_nc)
        Xs = np.fft.fftshift(X)

        mag = np.abs(Xs)
        mag /= np.max(mag)

        f_axis = np.linspace(-self.fs / 2.0, self.fs / 2.0, N, endpoint=False)
        return f_axis, mag
    
    def compute_ofdm_psd(
        self,
        tx_signal: np.ndarray,
        nfft_psd: int = 8192,
        remove_cp_first: bool = True,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Estimate Power Spectrum Density (PSD) of OFDM signal.

        - Optionally removes CP before calculating PSD.
        - Uses one big FFT with zero-padding to nfft_psd.
        - Returns frequency axis (Hz) and PSD in dB, normalized to 0 dB max.
        """
        x = np.asarray(tx_signal, dtype=complex).ravel()
        if remove_cp_first:
            x = self.remove_cp(x)
        if len(x) == 0:
            return np.array([]), np.array([])

        # Zero-pad or truncate to nfft_psd
        if len(x) < nfft_psd:
            x_pad = np.zeros(nfft_psd, dtype=complex)
            x_pad[:len(x)] = x
        else:
            x_pad = x[:nfft_psd]

        # FFT → shift → power
        X = np.fft.fft(x_pad)
        Xs = np.fft.fftshift(X)
        P = np.abs(Xs)**2           # power spectrum

        # Normalize to 0 dB
        P /= np.max(P)
        P_dB = 10 * np.log10(P + 1e-15)

        # Frequency axis [-fs/2, fs/2)
        f_axis = np.linspace(-self.fs/2.0, self.fs/2.0, nfft_psd, endpoint=False)
        return f_axis, P_dB




if __name__ == "__main__":
    ofdm = CPOFDM(Nfft=128, cp_len=32, fs=25.6e6)

    print("Subcarrier spacing [Hz]:", ofdm.subcarrier_spacing())
    print("OFDM symbol duration [s]:", ofdm.ofdm_symbol_duration())

    # ------------------------------
    # 16-QAM TEST (Task-1 + Task-3)
    # ------------------------------
    bits_per_symbol = 4
    print("Bit rate (16-QAM):", ofdm.bit_rate(bits_per_symbol))

    num_symbols = 256
    bits_tx = gen_data(num_symbols * bits_per_symbol, seed=0)

    qam_tx = mapper_16QAM(bits_tx)
    tx_signal, num_ofdm = ofdm.modulate(qam_tx)

    rx_signal = tx_signal.copy()
    qam_rx = ofdm.demodulate(rx_signal)

    bits_rx = demapper_16QAM(qam_rx[:num_symbols])
    ber16 = np.mean(bits_rx != bits_tx)
    print("BER (ideal channel, 16-QAM):", ber16)

    # ------------------------------
    # 4-QAM TEST (Task-1 + Task-3)
    # ------------------------------
    bits_per_symbol_4 = 2
    print("Bit rate (4-QAM):", ofdm.bit_rate(bits_per_symbol_4))

    num_symbols_4 = 256
    bits_tx4 = gen_data(num_symbols_4 * bits_per_symbol_4, seed=1)

    qam_tx4 = mapper_4QAM(bits_tx4)
    tx_signal_4, num_ofdm4 = ofdm.modulate(qam_tx4)

    rx_signal_4 = tx_signal_4.copy()
    qam_rx4 = ofdm.demodulate(rx_signal_4)

    bits_rx4 = demapper_4QAM(qam_rx4[:num_symbols_4])
    ber4 = np.mean(bits_rx4 != bits_tx4)
    print("BER (ideal channel, 4-QAM):", ber4)

    # ------------------------------
    # Time-domain envelope (16-QAM)
    # ------------------------------
    env = ofdm.compute_envelope(tx_signal)
    plt.figure()
    plt.plot(env)
    plt.title("OFDM time-domain envelope (16-QAM)")
    plt.xlabel("Sample index")
    plt.ylabel("|x[n]|")
    plt.grid(True)
    # ----------------------------------------
    # Spectrum WITHOUT CP (unfiltered)
    # ----------------------------------------
    f_axis_nc, mag_nc = ofdm.compute_spectrum_no_cp(tx_signal, apply_filter=False)
    plt.figure()
    plt.plot(f_axis_nc / 1e6, 20 * np.log10(mag_nc + 1e-12))
    plt.title("OFDM magnitude spectrum (16-QAM) - CP removed")
    plt.xlabel("Frequency [MHz]")
    plt.ylabel("Magnitude [dB] (normalized)")
    plt.grid(True)

    # ----------------------------------------
    # Spectrum WITHOUT CP + LOW-PASS FILTER
    # ----------------------------------------
    f_axis_filt, mag_filt = ofdm.compute_spectrum_no_cp(
        tx_signal, apply_filter=True, L_filter=8
    )
    plt.figure()
    plt.plot(f_axis_filt / 1e6, 20 * np.log10(mag_filt + 1e-12))
    plt.title("OFDM magnitude spectrum (16-QAM) - CP removed + LPF")
    plt.xlabel("Frequency [MHz]")
    plt.ylabel("Magnitude [dB] (normalized)")
    plt.grid(True)

    # ------------------------------
    # OFDM magnitude spectrum (16-QAM) – shifted
    # ------------------------------
    f_axis, mag = ofdm.compute_spectrum(tx_signal)
    plt.figure()
    plt.plot(f_axis / 1e6, 20 * np.log10(mag + 1e-12))
    plt.title("OFDM magnitude spectrum (16-QAM)")
    plt.xlabel("Frequency [MHz]")
    plt.ylabel("Magnitude [dB] (normalized)")
    plt.grid(True)

    # ------------------------------
    # Discrete-time OFDM symbol spectrum (|X[k]|) – 16-QAM
    # ------------------------------
    qam_block = qam_tx[:ofdm.num_data_subcarriers]
    X = ofdm.build_ifft_input(qam_block)
    Xs = np.fft.fftshift(X)
    k = np.arange(-ofdm.Nfft // 2, ofdm.Nfft // 2)
    plt.figure()
    plt.stem(k, np.abs(Xs), basefmt=" ")
    plt.xlabel("Subcarrier index (shifted)")
    plt.ylabel("|X[k]|")
    plt.title("Magnitude of IFFT input X[k] (16-QAM, guards centered)")
    plt.grid(True)
    # ----------------------------------------
    # Power Spectrum Density of OFDM (16-QAM)
    # ----------------------------------------
    f_psd, P_psd_dB = ofdm.compute_ofdm_psd(
        tx_signal,
        nfft_psd=8192,
        remove_cp_first=True
    )

    plt.figure()
    plt.plot(f_psd / 1e4, P_psd_dB)
    plt.title("Power spectrum density of OFDM")
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Normalized Power (dB)")
    plt.grid(True)


    # ------------------------------
    # 4-QAM plots
    # ------------------------------
    env4 = ofdm.compute_envelope(tx_signal_4)
    plt.figure()
    plt.plot(env4)
    plt.xlabel("Sample index")
    plt.ylabel("|x[n]|")
    plt.title("OFDM time-domain envelope (4-QAM)")
    plt.grid(True)

    f_axis4, mag4 = ofdm.compute_spectrum(tx_signal_4)
    plt.figure()
    plt.plot(f_axis4 / 1e6, 20 * np.log10(mag4 + 1e-12))
    plt.xlabel("Frequency [MHz]")
    plt.ylabel("Magnitude [dB] (normalized)")
    plt.title("OFDM magnitude spectrum (4-QAM)")
    plt.grid(True)

    # ------------------------------
    # Discrete-time OFDM symbol spectrum (|X[k]|) – 4-QAM
    # ------------------------------
    qam_block_4 = qam_tx4[:ofdm.num_data_subcarriers]
    X4 = ofdm.build_ifft_input(qam_block_4)
    X4s = np.fft.fftshift(X4)
    k4 = np.arange(-ofdm.Nfft // 2, ofdm.Nfft // 2)
    plt.figure()
    plt.stem(k4, np.abs(X4s), basefmt=" ")
    plt.xlabel("Subcarrier index (shifted)")
    plt.ylabel("|X[k]|")
    plt.title("Magnitude of IFFT input X[k] (4-QAM, guards centered)")
    plt.grid(True)

    plt.show()
