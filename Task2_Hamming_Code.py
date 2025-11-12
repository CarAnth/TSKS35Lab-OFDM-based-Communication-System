import numpy as np
class Hamming74:
    # [p1 p2 d1 p3 d2 d3 d4]
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
                   [0,0,0,0,0,0,1]], dtype=int) # 4x7 (bilgi seçici)

    @staticmethod
    def _str_to_bits4(s: str) -> np.ndarray:
        # "1010" -> [1,0,1,0]
        return np.fromiter((1 if ch=='1' else 0 for ch in s.strip()), dtype=int, count=4)

    def encode(self, msg4) -> np.ndarray:
        """ msg4: '1010' (str) veya shape (4,1)/(4,) bit """
        if isinstance(msg4, str):
            u = self._str_to_bits4(msg4).reshape(4,1)
        else:
            u = np.asarray(msg4, dtype=int).reshape(4,1)
        c = (self.Gt @ u) % 2                # 7x1
        return c                             

    def parityCheck(self, codeword7) -> np.ndarray:
        r = np.asarray(codeword7, dtype=int).reshape(7,1)
        s = (self.H @ r) % 2                 # 3x1, [s1 s2 s3]^T
        return s.ravel()                     # [s1,s2,s3]

    def _pos_from_syndrome(self, s3):
        # sütun indeksleme: pos = s1 + 2*s2 + 4*s3  (1..7; 0=hatasız)
        s1, s2, s3b = s3.tolist()
        return s1 + 2*s2 + 4*s3b

    def _flipbit(self, r7, pos):
        r = np.asarray(r7, dtype=int).copy().ravel()
        if 1 <= pos <= 7:
            r[pos-1] ^= 1
        return r.reshape(7,1)

    def getOriginalMessage(self, codeword7):
        r = np.asarray(codeword7, dtype=int).reshape(7,1)
        s = self.parityCheck(r)              # [s1,s2,s3]
        pos = self._pos_from_syndrome(s)     # 0..7 
        corrected = (pos != 0)
        if corrected:
            r = self._flipbit(r, pos)
        uhat = (self.R @ r) % 2              # 4x1
        return uhat.ravel(), corrected, pos
    def h74_encode_stream(self, u_bits: np.ndarray) -> tuple[np.ndarray, int]:
        
        L4 = (len(u_bits)+3)//4*4
        u   = np.zeros(L4, dtype=int); u[:len(u_bits)] = u_bits
        out = []
        for k in range(0, L4, 4):
            c = self.encode(u[k:k+4])             # 7x1
            out.append(c.ravel())
        return np.concatenate(out), len(u_bits)

    def h74_decode_stream(self ,r_bits: np.ndarray, orig_len: int) -> tuple[np.ndarray, list]:
        
        L7 = (len(r_bits)//7)*7
        outs = []; stats=[]
        for k in range(0, L7, 7):
            uhat, corrected, pos = self.getOriginalMessage(r_bits[k:k+7])
            outs.append(uhat)
            stats.append((corrected, pos))
        uhat_all = np.concatenate(outs)[:orig_len]
        return uhat_all, stats
    
    
ham = Hamming74()
# 1) Encode / Decode (no error)
u = np.array([1,0,1,0])
c = ham.encode(u)               # shape (7,1)
uhat, corrected, pos = ham.getOriginalMessage(c)
print("OK:", uhat, corrected, pos)   # corrected False, pos 0

# 2) One-bit error correction
c_err = c.copy(); c_err[2,0] ^= 1    # flip bit-3
uhat2, corrected2, pos2 = ham.getOriginalMessage(c_err)
print("1-bit:", uhat2, corrected2, pos2)  # corrected True, pos 3

# 3) Stream test
bits = np.random.randint(0,2, 20)    # 20 info bits
c_stream, Lorig = ham.h74_encode_stream(bits)
p = 0.01
flips = (np.random.rand(len(c_stream)) < p).astype(int)
r_stream = (c_stream ^ flips)
uhat_stream, stats = ham.h74_decode_stream(r_stream, Lorig)
print("BER(info):", np.mean(uhat_stream != bits[:len(uhat_stream)]))
print("1st 5 stats:", stats[:5])     # (corrected?, pos)
