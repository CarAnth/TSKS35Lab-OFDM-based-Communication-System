import numpy as np
import matplotlib.pyplot as plt
SQRT10 = np.sqrt(10)
SQRT2 = np.sqrt(2)

# Gray 2-bit → level: 00→-3, 01→-1, 10→+3, 11→+1
_QAM16_LEVELS = np.array([-3, -1, +3, +1])  # index = 2*bh + bl
_I_4 = np.array([-1, -1, +1, +1])          # index = 2*b1 + b0
_Q_4 = np.array([-1, +1, -1, +1])


def gen_data(N, seed=None):
    if seed is not None:
        np.random.seed(seed)              
    return np.random.randint(0, 2, N, dtype=int)
 
def mapper_16QAM(bits):
    L = (len(bits)//4)*4
    b = bits[:L].reshape(-1, 4)             # [b3 b2 b1 b0]
    idxI = 2*b[:,0] + b[:,1]                # (b3,b2)
    idxQ = 2*b[:,2] + b[:,3]                # (b1,b0)
    I = _QAM16_LEVELS[idxI]
    Q = _QAM16_LEVELS[idxQ]
    return (I + 1j*Q) / SQRT10              # Es ≈ 1
 
def mapper_4QAM(bits):
    L = (len(bits)//2)*2
    b = bits[:L].reshape(-1, 2)            # [b1 b0]
    idx = 2*b[:,0] + b[:,1]
    I = _I_4[idx]
    Q = _Q_4[idx]
    return (I + 1j*Q) / SQRT2              # Es ≈ 1

def chnl_AWGN(sig, SNR, K):#channel
    Ps = np.mean(np.abs(sig)**2) * K
    gamma = 10**(SNR/10)
    N0 = Ps / gamma
    n = np.sqrt(N0/2) * (np.random.randn(len(sig)) + 1j*np.random.randn(len(sig)))
    return sig + n


def demapper_16QAM(sym):
    out = []
    s = sym * SQRT10
    def sl(v): return -3 if v<-2 else -1 if v<0 else +1 if v<2 else +3
    def inv(l): return (0,0) if l==-3 else (0,1) if l==-1 else (1,1) if l==+1 else (1,0)
    for z in s:
        bI = inv(sl(z.real)); bQ = inv(sl(z.imag))
        out.extend([bI[0], bI[1], bQ[0], bQ[1]])
    return np.array(out, int)

def demapper_4QAM(sym):
    out = []
    s = sym * SQRT2
    for z in s:
        # Gray: 00:-1-j, 01:-1+j, 11:+1+j, 10:+1-j
        if   z.real<0 and z.imag<0: out += [0,0]
        elif z.real<0 and z.imag>=0: out += [0,1]
        elif z.real>=0 and z.imag>=0: out += [1,1]
        else:                        out += [1,0]
    return np.array(out, int)

def SNR_BER_analysis(mod='16QAM', SNRs=range(-10,11), Nbits=100_000, seed=0):
    np.random.seed(seed)
    BER = []
    for S in SNRs:
        bits = np.random.randint(0,2, Nbits)
        if mod=='16QAM':
            bits = bits[: (len(bits)//4)*4]
            tx   = mapper_16QAM(bits)
            rx   = chnl_AWGN(tx, S, 1)
            bhat = demapper_16QAM(rx)
        else:
            bits = bits[: (len(bits)//2)*2]
            tx   = mapper_4QAM(bits)
            rx   = chnl_AWGN(tx, S, 1)
            bhat = demapper_4QAM(rx)
        BER.append(np.mean(bhat != bits[:len(bhat)]))
    return np.array(SNRs), np.array(BER)
 
def plot_constellation(x, title, n=100):
    s = x[:min(n, len(x))]
    plt.figure(); plt.plot(s.real, s.imag, '.', ms=5)
    plt.axhline(0, lw=0.6); plt.axvline(0, lw=0.6)
    plt.gca().set_aspect('equal'); plt.grid(True); plt.title(title)
    plt.xlabel('I'); plt.ylabel('Q'); plt.tight_layout(); plt.show()
      
bits = gen_data(100_000, seed=0) 
 
##print("Hello world")### RESPECT TO ALFRED
b16 = bits[: (len(bits)//4)*4]
tx16 = mapper_16QAM(b16)
plot_constellation(tx16, "16-QAM — TX (clean)", n=200)

# ===================== 16-QAM: RX konstelasyonları (1,5,10 dB; 100 nokta) =====================
for S in [1, 5, 10]:
    rx16 = chnl_AWGN(tx16, S, 1)
    plot_constellation(rx16, f"16-QAM — RX (SNR={S} dB)", n=100)

# ===================== 4-QAM: TX & RX konstelasyonları (isteğe bağlı) =====================
b4 = bits[: (len(bits)//2)*2]
tx4 = mapper_4QAM(b4)
plot_constellation(tx4, "4-QAM — TX (clean)", n=200)
for S in [1, 5, 10]:
    rx4 = chnl_AWGN(tx4, S, 1)
    plot_constellation(rx4, f"4-QAM — RX (SNR={S} dB)", n=100)

# ===================== BER eğrileri (−10..10 dB, ≥1e5 bit/SNR) =====================
SNRs, BER16 = SNR_BER_analysis('16QAM', range(-10,11), Nbits=100_000, seed=1)
_,    BER4  = SNR_BER_analysis('4QAM',  range(-10,11), Nbits=100_000, seed=1)

plt.figure()
plt.semilogy(SNRs, BER4,  'o-', label='4-QAM')
plt.semilogy(SNRs, BER16, 's-', label='16-QAM')
plt.grid(True, which='both')
plt.xlabel('SNR (dB)'); plt.ylabel('BER')
plt.title('BER vs SNR — 4-QAM vs 16-QAM')
plt.legend() 
plt.tight_layout()
plt.show()