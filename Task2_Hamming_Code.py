import numpy as np
import matplotlib.pyplot as plt

from Task1_QAM_modulator import (
    mapper_16QAM, demapper_16QAM,
    mapper_4QAM,  demapper_4QAM,
    chnl_AWGN,    SNR_BER_analysis
)


class H74Codec:
    G_enc_T = np.array([
        [1, 1, 0, 1],
        [1, 0, 1, 1],
        [1, 0, 0, 0],
        [0, 1, 1, 1],
        [0, 1, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1]
    ], dtype=int)

    H_par = np.array([
        [1, 0, 1, 0, 1, 0, 1],
        [0, 1, 1, 0, 0, 1, 1],
        [0, 0, 0, 1, 1, 1, 1]
    ], dtype=int)

    R_sel = np.array([
        [0, 0, 1, 0, 0, 0, 0],
        [0, 0, 0, 0, 1, 0, 0],
        [0, 0, 0, 0, 0, 1, 0],
        [0, 0, 0, 0, 0, 0, 1]
    ], dtype=int)

    @staticmethod
    def _parse_4bits(bit_str: str) -> np.ndarray:
        return np.fromiter(
            (1 if ch == '1' else 0 for ch in bit_str.strip()),
            dtype=int,
            count=4
        )

    def encode_block(self, info4) -> np.ndarray:
        if isinstance(info4, str):
            u_vec = self._parse_4bits(info4).reshape(4, 1)
        else:
            u_vec = np.asarray(info4, dtype=int).reshape(4, 1)
        c_vec = (self.G_enc_T @ u_vec) % 2
        return c_vec

    def syndrome(self, c7) -> np.ndarray:
        c_vec = np.asarray(c7, dtype=int).reshape(7, 1)
        s_vec = (self.H_par @ c_vec) % 2
        return s_vec.ravel()

    def _syndrome_to_position(self, syn_bits):
        b1, b2, b3 = syn_bits.tolist()
        return b1 + 2 * b2 + 4 * b3

    def _toggle_bit(self, r7, pos):
        r_vec = np.asarray(r7, dtype=int).copy().ravel()
        if 1 <= pos <= 7:
            r_vec[pos - 1] ^= 1
        return r_vec.reshape(7, 1)

    def decode_block(self, c7):
        r_vec = np.asarray(c7, dtype=int).reshape(7, 1)
        s_vec = self.syndrome(r_vec)
        err_pos = self._syndrome_to_position(s_vec)
        corrected = (err_pos != 0)
        if corrected:
            r_vec = self._toggle_bit(r_vec, err_pos)
        u_hat = (self.R_sel @ r_vec) % 2
        return u_hat.ravel(), corrected, err_pos

    def encode_sequence(self, info_bits: np.ndarray) -> tuple[np.ndarray, int]:
        info_bits = np.asarray(info_bits, dtype=int).ravel()
        L4 = (len(info_bits) + 3) // 4 * 4
        padded = np.zeros(L4, dtype=int)
        padded[:len(info_bits)] = info_bits

        code_list = []
        for idx in range(0, L4, 4):
            c_block = self.encode_block(padded[idx:idx + 4])
            code_list.append(c_block.ravel())

        return np.concatenate(code_list), len(info_bits)

    def decode_sequence(self, recv_bits: np.ndarray, orig_len: int) -> tuple[np.ndarray, list]:
        recv_bits = np.asarray(recv_bits, dtype=int).ravel()
        L7 = (len(recv_bits) // 7) * 7

        info_est_list = []
        block_stats = []

        for idx in range(0, L7, 7):
            u_hat, corrected, pos = self.decode_block(recv_bits[idx:idx + 7])
            info_est_list.append(u_hat)
            block_stats.append((corrected, pos))

        u_all = np.concatenate(info_est_list)[:orig_len]
        return u_all, block_stats


def h74_encode_bits(info_bits: np.ndarray) -> tuple[np.ndarray, int]:
    codec = H74Codec()
    coded_bits, orig_len = codec.encode_sequence(info_bits)
    return coded_bits, orig_len


def h74_decode_bits(recv_bits: np.ndarray, orig_len: int) -> np.ndarray:
    codec = H74Codec()
    info_hat, stats = codec.decode_sequence(recv_bits, orig_len)
    return info_hat


def coded_SNR_BER(mod='16QAM', SNRs=range(-2, 11), Nbits=100_000, seed=0):
    np.random.seed(seed)
    ber_vals = []

    for snr_val in SNRs:
        info_bits = np.random.randint(0, 2, Nbits)

        coded_bits, L_orig = h74_encode_bits(info_bits)

        if mod == '16QAM':
            tx_sym = mapper_16QAM(coded_bits)
            rx_sym = chnl_AWGN(tx_sym, snr_val, 1)
            hard_bits = demapper_16QAM(rx_sym)
        else:
            tx_sym = mapper_4QAM(coded_bits)
            rx_sym = chnl_AWGN(tx_sym, snr_val, 1)
            hard_bits = demapper_4QAM(rx_sym)

        info_hat = h74_decode_bits(hard_bits, L_orig)
        ber_vals.append(np.mean(info_hat != info_bits[:len(info_hat)]))

    return np.array(SNRs), np.array(ber_vals)


def illustrate_single_double_error():
    codec = H74Codec()

    u = np.array([1, 0, 1, 0])
    c = codec.encode_block(u).ravel()
    print("Original information bits:", u)
    print("Codeword c              :", c)

    r_single = c.copy()
    r_single[2] ^= 1
    uhat1, corrected1, pos1 = codec.decode_block(r_single)
    print("\n--- Single-bit error example ---")
    print("Received (1-bit error)  :", r_single)
    print("Decoded u_hat           :", uhat1)
    print("Correction applied?     :", corrected1, "(position =", pos1, ")")
    print("Any info bit error?     :", np.any(uhat1 != u))

    r_double = c.copy()
    r_double[1] ^= 1
    r_double[4] ^= 1
    uhat2, corrected2, pos2 = codec.decode_block(r_double)
    print("\n--- Double-bit error example ---")
    print("Received (2-bit errors) :", r_double)
    print("Decoded u_hat           :", uhat2)
    print("Correction applied?     :", corrected2, "(position =", pos2, ")")
    print("Any info bit error?     :", np.any(uhat2 != u))


if __name__ == "__main__":
    codec = H74Codec()

    u = np.array([1, 0, 1, 0])
    c = codec.encode_block(u)
    uhat, corrected, pos = codec.decode_block(c)
    print("No-error test:", uhat, corrected, pos)

    c_err = c.copy()
    c_err[2, 0] ^= 1
    uhat2, corrected2, pos2 = codec.decode_block(c_err)
    print("1-bit error test:", uhat2, corrected2, pos2)

    test_bits = np.random.randint(0, 2, 20)
    c_stream, Lorig = codec.encode_sequence(test_bits)
    p = 0.01
    flips = (np.random.rand(len(c_stream)) < p).astype(int)
    r_stream = (c_stream ^ flips)
    uhat_stream, stats = codec.decode_sequence(r_stream, Lorig)
    print("BER(info):", np.mean(uhat_stream != test_bits[:len(uhat_stream)]))
    print("1st 5 stats:", stats[:5])

    illustrate_single_double_error()

    SNRs = list(range(-2, 11))
    Nbits = 200_000

    SNRs_u, BER16_u = SNR_BER_analysis('16QAM', SNRs, Nbits=Nbits, seed=1)
    _,      BER4_u  = SNR_BER_analysis('4QAM',  SNRs, Nbits=Nbits, seed=1)

    SNRs_c, BER16_c = coded_SNR_BER('16QAM', SNRs, Nbits=Nbits, seed=2)
    _,      BER4_c  = coded_SNR_BER('4QAM',  SNRs, Nbits=Nbits, seed=2)

    plt.figure()
    plt.semilogy(SNRs_u, BER4_u,  'o-',  label='4-QAM uncoded')
    plt.semilogy(SNRs_c, BER4_c,  'o--', label='4-QAM Hamming(7,4)')
    plt.semilogy(SNRs_u, BER16_u, 's-',  label='16-QAM uncoded')
    plt.semilogy(SNRs_c, BER16_c, 's--', label='16-QAM Hamming(7,4)')
    plt.grid(True, which='both')
    plt.xlabel('SNR (dB)')
    plt.ylabel('BER (info bits)')
    plt.title('BER vs SNR with Hamming(7,4)')
    plt.legend()
    plt.tight_layout()
    plt.show()
