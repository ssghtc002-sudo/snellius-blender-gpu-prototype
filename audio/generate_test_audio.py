"""Generate a crisp 10-second stereo cinematic test audio track for the Snellius GPU prototype."""
import math
import os
import struct
import wave

SR = 48000
DUR = 10.0
N = int(SR * DUR)
OUT_WAV = os.path.join(os.path.dirname(__file__), "snellius_test_audio.wav")

samples_l = [0.0] * N
samples_r = [0.0] * N

# Lush ambient chord (D minor -> F major -> A major) + crystal refraction ping
chords = [
    (0.0, 3.5, [146.83, 220.00, 293.66, 349.23]),
    (3.5, 7.0, [174.61, 261.63, 349.23, 440.00]),
    (7.0, 10.0, [220.00, 277.18, 329.63, 440.00]),
]

for i in range(N):
    t = i / SR
    fade = min(1.0, t / 0.4) * min(1.0, (DUR - t) / 0.6)
    val_l = 0.0
    val_r = 0.0

    # Warm pad chords
    for t0, t1, freqs in chords:
        if t0 <= t < t1:
            env = math.sin(math.pi * (t - t0) / (t1 - t0)) ** 0.5
            for idx, f in enumerate(freqs):
                w = math.sin(2.0 * math.pi * f * t) * 0.08 * env
                pan = 0.5 + (idx - 1.5) * 0.14
                val_l += w * (1.0 - pan)
                val_r += w * pan

    # Crystal shimmer arpeggio synced with beam bending
    arp_notes = [587.33, 698.46, 880.00, 1046.50, 1174.66]
    step = int(t * 2.0)
    local_t = (t * 2.0) - step
    arp_env = math.exp(-local_t * 5.5)
    arp_f = arp_notes[step % len(arp_notes)]
    ping = math.sin(2.0 * math.pi * arp_f * t) * 0.12 * arp_env
    val_l += ping * 0.55
    val_r += ping * 0.45

    # Soft sub-bass pulse
    sub = math.sin(2.0 * math.pi * 73.42 * t) * 0.10 * (0.6 + 0.4 * math.sin(2.0 * math.pi * 0.5 * t))
    samples_l[i] = max(-0.92, min(0.92, (val_l + sub) * fade))
    samples_r[i] = max(-0.92, min(0.92, (val_r + sub) * fade))

with wave.open(OUT_WAV, "wb") as wf:
    wf.setnchannels(2)
    wf.setsampwidth(2)
    wf.setframerate(SR)
    frames = bytearray()
    for l, r in zip(samples_l, samples_r):
        frames += struct.pack("<hh", int(l * 32767), int(r * 32767))
    wf.writeframes(frames)

print(f"Generated {OUT_WAV} ({DUR:.1f}s stereo 48kHz)")
