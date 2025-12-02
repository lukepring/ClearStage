import os
import argparse
import logging
import soundfile as sf
import numpy as np
import torch
import torchaudio
import sys
import time
import random
import threading
import shutil
from scipy import signal

# --- CONFIGURATION ---
logging.getLogger().setLevel(logging.CRITICAL)

# --- VISUALIZER ---
class FullScreenVisualizer:
    def __init__(self, total_steps=6):
        self.total_steps = total_steps
        self.current_step = 0
        self.step_desc = "Initializing"
        self.logs = []
        self.stop_event = threading.Event()
        self.thread = threading.Thread(target=self._animate)
        
        # Retro Terminal Colors
        self.C_CYAN = "\033[96m"
        self.C_MAGENTA = "\033[95m"
        self.C_GREEN = "\033[92m"
        self.C_ORANGE = "\033[93m"
        self.C_GREY = "\033[90m"
        self.C_RESET = "\033[0m"
        self.C_BOLD = "\033[1m"

    def __enter__(self):
        sys.stdout.write("\033[2J\033[H")
        self.thread.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop_event.set()
        self.thread.join()
        sys.stdout.write(f"\033[{shutil.get_terminal_size().lines}H")
        sys.stdout.flush()

    def log(self, message):
        timestamp = time.strftime("%H:%M:%S")
        self.logs.append(f"{self.C_GREY}[{timestamp}]{self.C_RESET} {message}")
        if len(self.logs) > 10:
            self.logs.pop(0)

    def update_step(self, step, desc):
        self.current_step = step
        self.step_desc = desc

    def _animate(self):
        while not self.stop_event.is_set():
            term_w, term_h = shutil.get_terminal_size()
            sys.stdout.write("\033[H") # Go Home

            # HEADER
            print(f"{self.C_BOLD}{self.C_ORANGE}  LUPRI CLEARSTAGE v3.1 // DYNAMIC RECONSTRUCTION ENGINE  {self.C_RESET}".center(term_w))
            print(f"{self.C_GREY}{'-'*term_w}{self.C_RESET}")

            # DASHBOARD LAYOUT
            print(f"\n {self.C_BOLD}PROCESS TELEMETRY:{self.C_RESET}")
            
            # Simulated VU Meters
            vu_l = random.randint(0, 20)
            vu_r = random.randint(0, 20)
            print(f" L [{self.C_GREEN}{'|'*vu_l}{self.C_GREY}{'.'*(20-vu_l)}{self.C_RESET}] -12dB")
            print(f" R [{self.C_GREEN}{'|'*vu_r}{self.C_GREY}{'.'*(20-vu_r)}{self.C_RESET}] -12dB")

            print("")
            
            # Main Progress
            pct = int((self.current_step / self.total_steps) * 100)
            bar_w = term_w - 20
            filled = int((pct / 100) * bar_w)
            bar = f"{self.C_CYAN}{'━'*filled}{self.C_GREY}{'┄'*(bar_w - filled)}{self.C_RESET}"
            print(f" {self.C_BOLD}STATUS:{self.C_RESET} {pct}% COMPLETE")
            print(f" [{bar}]")
            print(f" {self.C_MAGENTA}>> {self.step_desc}{self.C_RESET}")

            print(f"\n {self.C_BOLD}ENGINE LOGS:{self.C_RESET}")
            print(f" {self.C_GREY}┌{'─' * (term_w-4)}┐{self.C_RESET}")
            for log in self.logs:
                clean_log = (log + " " * term_w)[:term_w-6]
                print(f" {self.C_GREY}│{self.C_RESET} {clean_log}")
            for _ in range(10 - len(self.logs)):
                print(f" {self.C_GREY}│{self.C_RESET}")
            print(f" {self.C_GREY}└{'─' * (term_w-4)}┘{self.C_RESET}")

            sys.stdout.flush()
            time.sleep(0.08)

# --- AUDIO ENGINE ---
class ClearStageEngine:
    def __init__(self, use_gpu=True, visualizer=None):
        self.use_gpu = use_gpu
        self.viz = visualizer
        
        # GPU Check
        if self.use_gpu:
            if torch.cuda.is_available():
                self.device = "cuda"
                self.viz.log("Core: CUDA Acceleration Enabled")
            elif torch.backends.mps.is_available():
                self.device = "mps"
                self.viz.log("Core: Apple Metal (MPS) Enabled")
            else:
                self.device = "cpu"
                self.viz.log("Core: CPU Fallback Mode")
        else:
            self.device = "cpu"

        # Initialize Demucs
        try:
            import demucs.pretrained
            import demucs.apply
            self.demucs_pretrained = demucs.pretrained
            self.demucs_apply = demucs.apply
        except ImportError:
            print("Critical Error: Demucs library missing.")
            exit(1)

    def separate_sources(self, input_file, output_dir="temp_stems"):
        self.viz.log(f"Separating Stems with HTDemucs_FT on {self.device.upper()}...")
        
        model = self.demucs_pretrained.get_model('htdemucs_ft')
        model.to(self.device)
        
        try:
            wav, sr = sf.read(input_file)
        except Exception as e:
            self.viz.log(f"File Error: {e}")
            return None

        # Prep Tensor
        wav = torch.tensor(wav).float()
        if wav.dim() == 1: wav = wav[None, :].repeat(2, 1)
        else: wav = wav.t() 
            
        if sr != model.samplerate:
            wav = torchaudio.functional.resample(wav, sr, model.samplerate)
            sr = model.samplerate

        # Standardize
        ref = wav.mean(0)
        wav_norm = (wav - ref.mean()) / (ref.std() + 1e-8)
        wav_input = wav_norm[None, :, :].to(self.device)

        # Infer
        sources = self.demucs_apply.apply_model(model, wav_input, shifts=1, split=True, overlap=0.25, progress=False)[0]
        
        # Save
        sources = sources * ref.std() + ref.mean()
        stem_paths = {}
        song_name = os.path.splitext(os.path.basename(input_file))[0]
        save_dir = os.path.join(output_dir, "clearstage_v3", song_name)
        os.makedirs(save_dir, exist_ok=True)

        for source, name in zip(sources, model.sources):
            source = source.cpu().numpy().T 
            out_path = os.path.join(save_dir, f"{name}.wav")
            sf.write(out_path, source, sr)
            stem_paths[name] = out_path
            
        return stem_paths

    def hard_center_extraction(self, audio_path):
        """
        VOCAL PROCESSING:
        Aggressively removes stereo content from the Vocal stem.
        Since vocals are mono, anything 'wide' is bleed/crowd.
        """
        self.viz.log("Vocal: Removing Stereo Bleed (Hard Center)...")
        data, sr = sf.read(audio_path)
        
        if data.ndim == 1: return audio_path

        # Mid-Side
        mid = (data[:, 0] + data[:, 1]) * 0.5
        
        # Discard Side channel completely for Vocals
        # Reconstruct as Dual Mono
        new_l = mid
        new_r = mid
        stereo_clean = np.column_stack((new_l, new_r))
        
        output_path = audio_path.replace(".wav", "_center.wav")
        sf.write(output_path, stereo_clean, sr)
        return output_path

    def tame_crowd_stereo(self, audio_path):
        """
        CROWD PROCESSING:
        Keeps the crowd width but removes harsh transients (clapping)
        and high frequencies from the sides to make it sound 'behind' the singer.
        """
        self.viz.log("Crowd: Smoothing Stereo Width...")
        data, sr = sf.read(audio_path)
        if data.ndim == 1: return audio_path

        mid = (data[:, 0] + data[:, 1]) * 0.5
        side = (data[:, 0] - data[:, 1]) * 0.5

        # 1. Low Pass the Side Channel (6kHz) to kill whistling/clapping sizzle
        sos_lp = signal.butter(4, 6000, 'low', fs=sr, output='sos')
        side = signal.sosfilt(sos_lp, side, axis=0)

        # 2. Transient Smear (Limiter) on Side
        # Crushes spikes in the stereo field
        side = np.tanh(side * 2.0) * 0.5
        
        # Reconstruct
        new_l = mid + side
        new_r = mid - side
        stereo = np.column_stack((new_l, new_r))
        
        output_path = audio_path.replace(".wav", "_tamed.wav")
        sf.write(output_path, stereo, sr)
        return output_path

    def smart_vocal_ducking(self, vocal_path, backing_path, depth=0.4):
        """
        DYNAMIC MIXING:
        Lowers the Backing Track volume automatically when the Singer is active.
        depth=0.4 means ~40% reduction (-4dB) during vocals.
        """
        self.viz.log("Mixing: Applying Smart Vocal Ducking...")
        v_data, sr = sf.read(vocal_path)
        b_data, _ = sf.read(backing_path)
        
        # 1. Get Vocal Envelope
        abs_v = np.abs(v_data)
        # Slow Lowpass (5Hz) for smooth fading
        sos = signal.butter(4, 5, 'low', fs=sr, output='sos')
        if v_data.ndim > 1: abs_v = np.mean(abs_v, axis=1) # Mono envelope
        env = signal.sosfilt(sos, abs_v, axis=0)
        
        # Normalize Envelope 0-1
        peak = np.max(env)
        if peak > 0: env = env / peak
        
        # 2. Create Gain Curve
        # Invert envelope: Loud Vocal = Low Gain
        gain_curve = 1.0 - (env * depth)
        
        # Apply to backing track (handle channels)
        if b_data.ndim > 1:
            gain_curve = gain_curve[:, np.newaxis]
            
        # Ensure lengths match
        min_len = min(len(b_data), len(gain_curve))
        b_data = b_data[:min_len] * gain_curve[:min_len]
        
        output_path = backing_path.replace(".wav", "_ducked.wav")
        sf.write(output_path, b_data, sr)
        return output_path

    def vocal_polishing(self, audio_path):
        """
        Restores body to the center-extracted vocals.
        """
        self.viz.log("Vocal: Parallel Compression & EQ...")
        data, sr = sf.read(audio_path)
        
        # 1. Dynamic Expansion (Gate noise floor)
        sos_hp = signal.butter(4, 100, 'hp', fs=sr, output='sos')
        data_f = signal.sosfilt(sos_hp, data, axis=0)
        abs_sig = np.abs(data_f)
        b, a = signal.butter(4, 10/(sr/2), 'low') 
        env = signal.filtfilt(b, a, abs_sig, axis=0)
        peak = np.max(env)
        thresh = peak * 0.05
        gain_map = np.ones_like(env)
        mask = env < thresh
        gain_map[mask] = (env[mask] / thresh) ** 0.5
        gain_map = np.maximum(gain_map, 0.1)
        data = data * gain_map

        # 2. Parallel Compression (Body)
        crushed = np.tanh(data * 4.0)
        mixed = (data * 0.75) + (crushed * 0.15)
        
        # 3. Presence Boost (3kHz)
        sos_pres = signal.butter(2, [2500, 4000], 'bandpass', fs=sr, output='sos')
        pres = signal.sosfilt(sos_pres, mixed, axis=0)
        mixed = mixed + (pres * 0.2)

        output_path = audio_path.replace(".wav", "_polished.wav")
        sf.write(output_path, mixed, sr)
        return output_path

    def mix_master(self, stems, output_file):
        """
        Final Analog Summing.
        """
        def load(p): 
            d, s = sf.read(p)
            if d.ndim == 1: d = d[:, np.newaxis]
            return d, s

        v, sr = load(stems['vocals'])
        d, _ = load(stems['drums'])
        b, _ = load(stems['bass'])
        o, _ = load(stems['other']) # Now ducked/tamed
        
        # Safe Trim
        ln = min(len(v), len(d), len(b), len(o))
        v=v[:ln]; d=d[:ln]; b=b[:ln]; o=o[:ln]
        
        # --- MIX PROFILE ---
        # Vocals: Up front
        # Crowd: Pushed back (-4dB)
        master = (v * 1.05) + (d * 1.0) + (b * 1.0) + (o * 0.65)
        
        # Master Limiter (Soft Knee)
        master = np.tanh(master * 1.1)
        
        sf.write(output_file, master, sr)
        self.viz.log(f"Render Complete: {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Lupri ClearStage v3.1")
    parser.add_argument("input", help="Input file")
    parser.add_argument("--output", default="clearstage_master.wav", help="Output file")
    args = parser.parse_args()
    
    with FullScreenVisualizer(total_steps=6) as viz:
        engine = ClearStageEngine(use_gpu=True, visualizer=viz)
        
        # 1. Separation
        viz.update_step(1, "Spectral Separation (Demucs)")
        stems = engine.separate_sources(args.input)
        
        if stems:
            # 2. Vocal Isolation
            viz.update_step(2, "Vocal: Hard Center Extraction")
            stems['vocals'] = engine.hard_center_extraction(stems['vocals'])
            
            # 3. Crowd Taming
            viz.update_step(3, "Crowd: Side-Channel Smoothing")
            stems['other'] = engine.tame_crowd_stereo(stems['other'])
            
            # 4. Vocal Polish
            viz.update_step(4, "Vocal: Dynamics & Presence")
            stems['vocals'] = engine.vocal_polishing(stems['vocals'])
            
            # 5. Smart Mixing
            viz.update_step(5, "Mix: Smart Vocal Ducking")
            # We duck the 'other' stem (crowd) based on the 'vocals' stem
            stems['other'] = engine.smart_vocal_ducking(stems['vocals'], stems['other'], depth=0.4)
            
            # 6. Master
            viz.update_step(6, "Mastering: Analog Summing")
            engine.mix_master(stems, args.output)
            
            viz.log("SESSION COMPLETE.")
            time.sleep(3)