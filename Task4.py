import numpy as np
import matplotlib.pyplot as plt

from Task1_QAM_modulator import (
    mapper_16QAM, demapper_16QAM,
    mapper_4QAM,  demapper_4QAM
)
from Task3_CP_OFDM_transmitter_and_receiver import CPOFDM
from Task2_Hamming_Code import hamming74_encode_bits, hamming74_decode_bits

ofdm = CPOFDM(Nfft=128, cp_len=32, fs=25.6e6)


def rayleigh_channel(tx_signal, SNR_dB, Lh=14):
    """
    Pass tx_signal through an Lh-tap Rayleigh channel + AWGN.
    Returns:
        y : received time-domain signal
        h : channel impulse response (length Lh)
    """
    h = (np.random.randn(Lh) + 1j*np.random.randn(Lh)) / np.sqrt(2)
    y_chan = np.convolve(tx_signal, h)

    Ps = np.mean(np.abs(y_chan)**2)
    gamma = 10**(SNR_dB/10)
    N0 = Ps / gamma
    w = np.sqrt(N0/2) * (np.random.randn(len(y_chan)) + 1j*np.random.randn(len(y_chan)))
    y = y_chan + w
    return y, h

def plot_received_spectrum(rx_signal, fs, title="Received signal spectrum"):
    """
    Plots the spectrum of the OFDM receiver time-domain signal (rx_signal).
    """
    N = len(rx_signal)
    # FFT + shift
    RX = np.fft.fftshift(np.fft.fft(rx_signal))
    freqs = np.fft.fftshift(np.fft.fftfreq(N, d=1.0/fs))

    plt.figure()
    plt.plot(freqs / 1e6, 20 * np.log10(np.abs(RX) + 1e-12))
    plt.xlabel("Frequency [MHz]")
    plt.ylabel("Magnitude [dB]")
    plt.title(title)
    plt.grid(True)
    plt.tight_layout()


def plot_constellations(qam_noeq, qam_eq, mod_label="16-QAM", SNR_dB=None):
    """
    Plots the constellation diagrams of QAM symbols before and after ZF equalization side-by-side.
    """
    # Reduce number of samples (for clearer visualization)
    max_points = 5000
    if len(qam_noeq) > max_points:
        qam_noeq = qam_noeq[:max_points]
    if len(qam_eq) > max_points:
        qam_eq = qam_eq[:max_points]

    title_suffix = ""
    if SNR_dB is not None:
        title_suffix = f", SNR = {SNR_dB} dB"

    plt.figure(figsize=(10, 4))

    # Before ZF
    plt.subplot(1, 2, 1)
    plt.plot(qam_noeq.real, qam_noeq.imag, ".", markersize=2)
    plt.axhline(0, color="gray", linewidth=0.5)
    plt.axvline(0, color="gray", linewidth=0.5)
    plt.title(f"{mod_label} before ZF{title_suffix}")
    plt.xlabel("In-phase")
    plt.ylabel("Quadrature")
    plt.grid(True)

    # After ZF
    plt.subplot(1, 2, 2)
    plt.plot(qam_eq.real, qam_eq.imag, ".", markersize=2)
    plt.axhline(0, color="gray", linewidth=0.5)
    plt.axvline(0, color="gray", linewidth=0.5)
    plt.title(f"{mod_label} after ZF{title_suffix}")
    plt.xlabel("In-phase")
    plt.ylabel("Quadrature")
    plt.grid(True)

    plt.tight_layout()



def channel_frequency_response(h, ofdm):
    """
    Compute the frequency response of h on the OFDM grid.
    Returns H_data for data subcarriers.
    """
    Nfft = ofdm.Nfft
    h_pad = np.zeros(Nfft, dtype=complex)
    Lh = len(h)
    h_pad[:Lh] = h
    H = np.fft.fft(h_pad)
    H_data = H[ofdm.data_idx]
    return H_data


def simulate_one_snr(SNR_dB, num_bits=100_000, use_hamming=False, mod='16QAM', return_extras=False):
    """
    Full chain at a single SNR:
      bits -> (optional Hamming) -> QAM(mod) -> OFDM -> Rayleigh+AWGN
            -> OFDM RX -> (before/after ZF EQ) -> demap -> (optional Hamming)
            -> BER_before, BER_after
    mod: '16QAM' or '4QAM'
    """
    # 1) Bit source
    bits_tx = np.random.randint(0, 2, num_bits)

    # 2) Optional Hamming(7,4) coding
    if use_hamming:
        c_bits, orig_len = hamming74_encode_bits(bits_tx)
        bits_for_mapper = c_bits
    else:
        bits_for_mapper = bits_tx

    # 3) Choose mapper/demapper based on modulation
    if mod == '16QAM':
        map_fun = mapper_16QAM
        demap_fun = demapper_16QAM
    elif mod == '4QAM':
        map_fun = mapper_4QAM
        demap_fun = demapper_4QAM
    else:
        raise ValueError("mod must be '16QAM' or '4QAM'")

    # 4) QAM mapping
    qam_symbols = map_fun(bits_for_mapper)

    # 5) OFDM TX
    tx_signal, _ = ofdm.modulate(qam_symbols)

    # 6) Scale to 10 dBw (10 W)
    P_tx = np.mean(np.abs(tx_signal)**2)
    P_target = 10.0
    alpha = np.sqrt(P_target / P_tx)
    tx_scaled = alpha * tx_signal

    # 7) Rayleigh channel + AWGN
    rx_signal, h = rayleigh_channel(tx_scaled, SNR_dB)

    # 8) OFDM RX: block alignment
    block_len = ofdm.Nfft + ofdm.cp_len
    L = (len(rx_signal) // block_len) * block_len
    if L == 0:
        return np.nan, np.nan

    r_use = rx_signal[:L]
    Rmat = r_use.reshape(-1, block_len)

    # 9) Frequency response on data subcarriers
    H_data = channel_frequency_response(h, ofdm)

    qam_noeq_list = []
    qam_eq_list = []

    for row in Rmat:
        x_no_cp = row[ofdm.cp_len:]
        Y_block = np.fft.fft(x_no_cp)
        Y_data = Y_block[ofdm.data_idx]

        # Before EQ
        qam_noeq_list.append(Y_data)

        # After ZF EQ
        Y_eq = Y_data / H_data
        qam_eq_list.append(Y_eq)

    qam_noeq = np.concatenate(qam_noeq_list)
    qam_eq = np.concatenate(qam_eq_list)

    # Truncation due to padding in the OFDM modulator
    qam_noeq = qam_noeq[:len(qam_symbols)]
    qam_eq = qam_eq[:len(qam_symbols)]

    # 10) Demap to bits
    bits_noeq = demap_fun(qam_noeq)
    bits_eq = demap_fun(qam_eq)

    # 11) Optional Hamming decoding + BER calc on INFO bits
    if use_hamming:
        uhat_noeq = hamming74_decode_bits(bits_noeq, orig_len)
        uhat_eq = hamming74_decode_bits(bits_eq, orig_len)

        Lcmp_noeq = min(len(uhat_noeq), len(bits_tx))
        Lcmp_eq = min(len(uhat_eq), len(bits_tx))

        ber_before = np.mean(uhat_noeq[:Lcmp_noeq] != bits_tx[:Lcmp_noeq])
        ber_after = np.mean(uhat_eq[:Lcmp_eq] != bits_tx[:Lcmp_eq])
    else:
        Lcmp_noeq = min(len(bits_noeq), len(bits_tx))
        Lcmp_eq = min(len(bits_eq), len(bits_tx))

        ber_before = np.mean(bits_noeq[:Lcmp_noeq] != bits_tx[:Lcmp_noeq])
        ber_after = np.mean(bits_eq[:Lcmp_eq] != bits_tx[:Lcmp_eq])

    if return_extras:
        return ber_before, ber_after, rx_signal, qam_noeq, qam_eq
    else:
        return ber_before, ber_after


if __name__ == "__main__":
    SNRs = list(range(0, 21, 4))
    Nbits = 100_000

    # ===========================
    # 16-QAM RESULTS
    # ===========================
    ber_before_u_16 = []
    ber_after_u_16 = []
    ber_before_c_16 = []
    ber_after_c_16 = []

    for S in SNRs:
        b_u, a_u = simulate_one_snr(S, num_bits=Nbits, use_hamming=False, mod='16QAM')
        b_c, a_c = simulate_one_snr(S, num_bits=Nbits, use_hamming=True,  mod='16QAM')

        ber_before_u_16.append(b_u)
        ber_after_u_16.append(a_u)
        ber_before_c_16.append(b_c)
        ber_after_c_16.append(a_c)

        print(f"[16-QAM] SNR={S} dB | uncoded (before, after) = ({b_u:.3e}, {a_u:.3e}) "
              f"| coded = ({b_c:.3e}, {a_c:.3e})")

    ber_before_u_16 = np.array(ber_before_u_16)
    ber_after_u_16 = np.array(ber_after_u_16)
    ber_before_c_16 = np.array(ber_before_c_16)
    ber_after_c_16 = np.array(ber_after_c_16)

    plt.figure()
    plt.semilogy(SNRs, ber_before_u_16, 'o-',  label='16-QAM uncoded, before EQ')
    plt.semilogy(SNRs, ber_after_u_16,  'o--', label='16-QAM uncoded, after ZF')
    plt.semilogy(SNRs, ber_before_c_16, 's-',  label='16-QAM coded, before EQ')
    plt.semilogy(SNRs, ber_after_c_16,  's--', label='16-QAM coded, after ZF')
    plt.grid(True, which='both')
    plt.xlabel('SNR (dB)')
    plt.ylabel('BER (info bits)')
    plt.title('16-QAM OFDM over Rayleigh channel (ZF equalizer)')
    plt.legend()
    plt.tight_layout()

    # ===========================
    # 4-QAM RESULTS
    # ===========================
    ber_before_u_4 = []
    ber_after_u_4 = []
    ber_before_c_4 = []
    ber_after_c_4 = []

    for S in SNRs:
        b_u, a_u = simulate_one_snr(S, num_bits=Nbits, use_hamming=False, mod='4QAM')
        b_c, a_c = simulate_one_snr(S, num_bits=Nbits, use_hamming=True,  mod='4QAM')

        ber_before_u_4.append(b_u)
        ber_after_u_4.append(a_u)
        ber_before_c_4.append(b_c)
        ber_after_c_4.append(a_c)

        print(f"[4-QAM]  SNR={S} dB | uncoded (before, after) = ({b_u:.3e}, {a_u:.3e}) "
              f"| coded = ({b_c:.3e}, {a_c:.3e})")

    ber_before_u_4 = np.array(ber_before_u_4)
    ber_after_u_4 = np.array(ber_after_u_4)
    ber_before_c_4 = np.array(ber_before_c_4)
    ber_after_c_4 = np.array(ber_after_c_4)

    plt.figure()
    plt.semilogy(SNRs, ber_before_u_4, 'o-',  label='4-QAM uncoded, before EQ')
    plt.semilogy(SNRs, ber_after_u_4,  'o--', label='4-QAM uncoded, after ZF')
    plt.semilogy(SNRs, ber_before_c_4, 's-',  label='4-QAM coded, before EQ')
    plt.semilogy(SNRs, ber_after_c_4,  's--', label='4-QAM coded, after ZF')
    plt.grid(True, which='both')
    plt.xlabel('SNR (dB)')
    plt.ylabel('BER (info bits)')
    plt.title('4-QAM OFDM over Rayleigh channel (ZF equalizer)')
    plt.legend()
    plt.tight_layout()

    plt.show()
