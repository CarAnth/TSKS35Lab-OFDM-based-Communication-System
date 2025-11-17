import numpy as np

class Hamming74:
    # Codeword structure: [p1 p2 d1 p3 d2 d3 d4]
    Gt = np.array([[1,1,0,1],
                   [1,0,1,1],
                   [1,0,0,0],
                   [0,1,1,1],
                   [0,1,0,0],
                   [0,0,1,0],
                   [0,0,0,1]], dtype=int)      # 7x4 (G^T)

    H  = np.array([[1,0,1,0,1,0,1],
                   [0,1,1,0,0,1,1],
                   [0,0,0,1,1,1,1]], dtype=int) # 3x7

    R  = np.array([[0,0,1,0,0,0,0],
                   [0,0,0,0,1,0,0],
                   [0,0,0,0,0,1,0],
                   [0,0,0,0,0,0,1]], dtype=int) # 4x7 (information selector)

    @staticmethod
    def _str_to_bits4(s: str) -> np.ndarray:
        return np.fromiter((1 if ch == '1' else 0 for ch in s.strip()),
                           dtype=int, count=4)

    def encode(self, msg4) -> np.ndarray:
        if isinstance(msg4, str):
            u = self._str_to_bits4(msg4).reshape(4, 1)
        else:
            u = np.asarray(msg4, dtype=int).reshape(4, 1)
        c = (self.Gt @ u) % 2
        return c

    def parityCheck(self, codeword7) -> np.ndarray:
        r = np.asarray(codeword7, dtype=int).reshape(7, 1)
        s = (self.H @ r) % 2
        return s.ravel()

    def _pos_from_syndrome(self, s3):
        s1, s2, s3b = s3.tolist()
        return s1 + 2 * s2 + 4 * s3b

    def _flipbit(self, r7, pos):
        r = np.asarray(r7, dtype=int).copy().ravel()
        if 1 <= pos <= 7:
            r[pos - 1] ^= 1
        return r.reshape(7, 1)

    def getOriginalMessage(self, codeword7):
        r = np.asarray(codeword7, dtype=int).reshape(7, 1)
        s = self.parityCheck(r)
        pos = self._pos_from_syndrome(s)
        corrected = (pos != 0)
        if corrected:
            r = self._flipbit(r, pos)
        uhat = (self.R @ r) % 2
        return uhat.ravel(), corrected, pos

    def h74_encode_stream(self, u_bits: np.ndarray) -> tuple[np.ndarray, int]:
        u_bits = np.asarray(u_bits, dtype=int).ravel()
        L4 = (len(u_bits) + 3) // 4 * 4
        u = np.zeros(L4, dtype=int)
        u[:len(u_bits)] = u_bits

        out = []
        for k in range(0, L4, 4):
            c = self.encode(u[k:k+4])
            out.append(c.ravel())

        return np.concatenate(out), len(u_bits)

    def h74_decode_stream(self, r_bits: np.ndarray,
                          orig_len: int) -> tuple[np.ndarray, list]:
        r_bits = np.asarray(r_bits, dtype=int).ravel()
        L7 = (len(r_bits) // 7) * 7
        outs = []
        stats = []

        for k in range(0, L7, 7):
            uhat, corrected, pos = self.getOriginalMessage(r_bits[k:k+7])
            outs.append(uhat)
            stats.append((corrected, pos))

        uhat_all = np.concatenate(outs)[:orig_len]
        return uhat_all, stats


# ----------------------------------------------------------------------
#  demonstrate single-bit vs double-bit error behavior
# ----------------------------------------------------------------------
def demo_single_vs_double_error():
    ham = Hamming74()

    u = np.array([1, 0, 1, 0])
    c = ham.encode(u).ravel()
    print("Original information bits:", u)
    print("Codeword c              :", c)

    r_single = c.copy()
    r_single[2] ^= 1
    uhat1, corrected1, pos1 = ham.getOriginalMessage(r_single)
    print("\n--- Single-bit error example ---")
    print("Received (1-bit error)  :", r_single)
    print("Decoded u_hat           :", uhat1)
    print("Correction applied?     :", corrected1, "(position =", pos1, ")")
    print("Any info bit error?     :", np.any(uhat1 != u))

    r_double = c.copy()
    r_double[1] ^= 1
    r_double[4] ^= 1
    uhat2, corrected2, pos2 = ham.getOriginalMessage(r_double)
    print("\n--- Double-bit error example ---")
    print("Received (2-bit errors) :", r_double)
    print("Decoded u_hat           :", uhat2)
    print("Correction applied?     :", corrected2, "(position =", pos2, ")")
    print("Any info bit error?     :", np.any(uhat2 != u))


# ----------------------------------------------------------------------
#  clean interface for coded simulations with QAM
# ----------------------------------------------------------------------
def hamming74_encode_bits(u_bits: np.ndarray) -> tuple[np.ndarray, int]:
    ham = Hamming74()
    c_bits, orig_len = ham.h74_encode_stream(u_bits)
    return c_bits, orig_len


def hamming74_decode_bits(r_bits: np.ndarray, orig_len: int) -> np.ndarray:
    ham = Hamming74()
    uhat, stats = ham.h74_decode_stream(r_bits, orig_len)
    return uhat


# ----------------------------------------------------------------------
# Quick self-test (optional)
# ----------------------------------------------------------------------
if __name__ == "__main__":
    ham = Hamming74()

    u = np.array([1, 0, 1, 0])
    c = ham.encode(u)
    uhat, corrected, pos = ham.getOriginalMessage(c)
    print("No-error test:", uhat, corrected, pos)

    c_err = c.copy()
    c_err[2, 0] ^= 1
    uhat2, corrected2, pos2 = ham.getOriginalMessage(c_err)
    print("1-bit error test:", uhat2, corrected2, pos2)

    bits = np.random.randint(0, 2, 20)
    c_stream, Lorig = ham.h74_encode_stream(bits)
    p = 0.01
    flips = (np.random.rand(len(c_stream)) < p).astype(int)
    r_stream = (c_stream ^ flips)
    uhat_stream, stats = ham.h74_decode_stream(r_stream, Lorig)
    print("BER(info):", np.mean(uhat_stream != bits[:len(uhat_stream)]))
    print("1st 5 stats:", stats[:5])

    demo_single_vs_double_error()
