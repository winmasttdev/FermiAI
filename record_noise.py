import sys, numpy as np, sounddevice as sd

DEV = int(sys.argv[1]) if len(sys.argv) > 1 else None
SECS = int(sys.argv[2]) if len(sys.argv) > 2 else 6
SR = int(sd.query_devices(DEV)['default_samplerate']) if DEV is not None else 44100

print(f"recording {SECS}s from device {DEV} (hold mic ~15cm from the PC)...")
x = sd.rec(int(SECS * SR), samplerate=SR, channels=1, dtype='float32', device=DEV)
sd.wait()
x = x[:, 0]
x -= x.mean()

rms = float(np.sqrt(np.mean(x**2)))
peak = float(np.max(np.abs(x)))
dbfs = 20 * np.log10(rms) if rms > 0 else -99
peak_dbfs = 20 * np.log10(peak) if peak > 0 else -99

# crude ambient estimate: assumes a typical headset mic ~ -36 dBFS == ~35 dB(A) at this distance
REF_dBFS, REF_dBA = -36.0, 35.0
est_dba = REF_dBA + (dbfs - REF_dBFS) if rms > 0 else None

print(f"RMS   : {rms:.5f}")
print(f"level : {dbfs:6.1f} dBFS   peak {peak_dbfs:6.1f} dBFS")
if est_dba is not None:
    print(f"~approx: {est_dba:5.1f} dB(A)  (rough, needs mic calibration)")
print("scale: < -50 quiet | -50..-35 normal room | -35..-20 loud | > -10 near clipping")
