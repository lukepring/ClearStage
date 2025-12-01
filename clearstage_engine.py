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
from scipy import signal

# --- CONFIGURATION ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class TerminalVisualizer:
    """
    Creates a cool 'Audio Visualizer' style loading bar in the terminal.
    """
    def __init__(self, description="Processing"):
        self.description = description
        self.stop_event = threading.Event()
        self.thread = threading.Thread(target=self._animate)
        # Unicode block elements for 'spectrum' bars
        self.chars = [" ", " ", "▂", "▃", "▄", "▅", "▆", "▇", "█"]
        # ANSI colors (Cyan, Blue, Purple, White) for that cyberpunk look
        self.colors = ["\033[96m", "\033[94m", "\033[95m", "\033[97m"]

    def __enter__(self):
        self.thread.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop_event.set()
        self.thread.join()
        # Clear the line
        sys.stdout.write("\r" + " " * 80 + "\r")
        sys.stdout.flush()

    def _animate(self):
        width = 30
        while not self.stop_event.is_set():
            # Generate a random 'spectrum'
            bar = ""
            for _ in range(width):
                color = random.choice(self.colors)
                char = random.choice(self.chars)
                bar += f"{color}{char}"
            
            # Reset color at end
            sys.stdout.write(f"\r\033[1m{self.description}\033[0m \033[90m[\033[0m{bar}\033[0m\033[90m]\033[0m \033[3mAI Active...\033[0m")
            sys.stdout.flush()
            time.sleep(0.08)

class ClearStageEngine:
    def __init__(self, use_gpu=True):
        self.use_gpu = use_gpu
        
        # Intelligent Device Detection
        if self.use_gpu:
            if torch.cuda.is_available():
                self.device = "cuda"
                logging.info("NVIDIA GPU (CUDA) detected.")
            elif torch.backends.mps.is_available():
                self.device = "mps"
                logging.info("Apple Silicon GPU (MPS) detected.")
            else:
                self.device = "cpu"
                logging.warning("GPU requested but not available. Falling back to CPU.")
        else:
            self.device = "cpu"

        logging.info(f"Initializing ClearStage Engine on {self.device.upper()}...")
        
        # Check for demucs
        try:
            import demucs.pretrained
            import demucs.apply
            self.demucs_pretrained = demucs.pretrained
            self.demucs_apply = demucs.apply
            logging.info("Demucs library loaded successfully.")
        except ImportError:
            logging.error("Demucs not found. Please run: pip install demucs")
            exit(1)

    def separate_sources(self, input_file, output_dir="temp_stems"):
        """
        Uses Demucs via direct library call to bypass Torchaudio backend issues.
        """
        logging.info(f"Loading Demucs model (htdemucs_ft) on {self.device}...")
        
        # Load the model directly
        model = self.demucs_pretrained.get_model('htdemucs_ft')
        model.to(self.device)
        
        logging.info(f"Separating sources for: {input_file}")
        
        # Load Audio using SoundFile (Safe from TorchCodec errors)
        try:
            wav, sr = sf.read(input_file)
        except Exception as e:
            logging.error(f"Could not read audio file: {e}")
            return None

        # Convert to Torch Tensor: (Time, Channels) -> (Channels, Time)
        # .float() ensures float32, preventing MPS errors
        wav = torch.tensor(wav).float()
        if wav.dim() == 1:
            wav = wav[None, :] # Add channel dim for mono
            wav = wav.repeat(2, 1) # Force stereo
        else:
            wav = wav.t() 
            
        # Resample if necessary (Demucs htdemucs runs at 44.1kHz)
        if sr != model.samplerate:
            logging.info(f"Resampling from {sr}Hz to {model.samplerate}Hz...")
            wav = torchaudio.functional.resample(wav, sr, model.samplerate)
            sr = model.samplerate

        # Normalize (Standardize) for the model
        ref = wav.mean(0)
        wav_norm = (wav - ref.mean()) / (ref.std() + 1e-8)
        
        # Add batch dimension: (1, Channels, Time)
        wav_input = wav_norm[None, :, :].to(self.device)

        # Apply Model
        # We wrap this in our visualizer because it takes the longest
        # We turn OFF the default progress bar (progress=False) so our visualizer owns the screen
        with TerminalVisualizer("Separating Stems  "):
            sources = self.demucs_apply.apply_model(model, wav_input, shifts=1, split=True, overlap=0.25, progress=False)[0]
        
        # Sources order: drums, bass, other, vocals
        source_names = model.sources
        stem_paths = {}
        
        song_name = os.path.splitext(os.path.basename(input_file))[0]
        save_dir = os.path.join(output_dir, "htdemucs_ft", song_name)
        os.makedirs(save_dir, exist_ok=True)
        
        sources = sources * ref.std() + ref.mean()

        for source, name in zip(sources, source_names):
            source = source.cpu().numpy().T # Back to (Time, Channels)
            out_path = os.path.join(save_dir, f"{name}.wav")
            sf.write(out_path, source, sr)
            stem_paths[name] = out_path
            
        logging.info("Source separation complete.")
        return stem_paths

    def apply_smart_masking(self, vocal_path, crowd_path, aggressiveness=1.0):
        """
        ADVANCED AI METHOD: Time-Frequency Ratio Masking.
        """
        logging.info(f"AI Auto-Masking initialized (Aggressiveness target: {aggressiveness})...")
        
        v_wav, sr = sf.read(vocal_path)
        c_wav, _ = sf.read(crowd_path)
        
        v_tensor = torch.tensor(v_wav).float().t().to(self.device)
        c_tensor = torch.tensor(c_wav).float().t().to(self.device)
        
        min_len = min(v_tensor.shape[1], c_tensor.shape[1])
        v_tensor = v_tensor[:, :min_len]
        c_tensor = c_tensor[:, :min_len]
        
        with TerminalVisualizer("AI Spectral Masking"):
            # STFT
            n_fft = 2048
            hop_length = 512
            window = torch.hann_window(n_fft).to(self.device)
            
            v_spec = torch.stft(v_tensor, n_fft=n_fft, hop_length=hop_length, window=window, return_complex=True)
            c_spec = torch.stft(c_tensor, n_fft=n_fft, hop_length=hop_length, window=window, return_complex=True)
            
            v_mag = torch.abs(v_spec)
            c_mag = torch.abs(c_spec)
            
            # Auto-Calibration
            rms_v = torch.sqrt(torch.mean(v_mag**2))
            rms_c = torch.sqrt(torch.mean(c_mag**2))
            snr_ratio = rms_v / (rms_c + 1e-6)
            
            # INCREASED AGGRESSIVENESS MULTIPLIERS
            auto_alpha = 1.0
            if snr_ratio < 1.5: 
                auto_alpha = 6.0 * aggressiveness 
            elif snr_ratio < 5.0:
                auto_alpha = 3.5 * aggressiveness 
            else:
                auto_alpha = 1.5 * aggressiveness 

            mask = v_mag / (v_mag + (c_mag * auto_alpha) + 1e-8)
            mask = mask ** 2 
            
            v_spec_clean = v_spec * mask
            v_clean_tensor = torch.istft(v_spec_clean, n_fft=n_fft, hop_length=hop_length, window=window, length=min_len)
            
            v_clean = v_clean_tensor.cpu().numpy().T
            
        # Log the decision AFTER the visualizer clears
        if snr_ratio < 1.5:
             logging.info(f"  -> Heavy Crowd/Noise (SNR: {snr_ratio:.2f}). MAX MASKING (Alpha: {auto_alpha:.1f})")
        elif snr_ratio < 5.0:
             logging.info(f"  -> Moderate Background Noise (SNR: {snr_ratio:.2f}). High masking (Alpha: {auto_alpha:.1f})")
        else:
             logging.info(f"  -> Clean Recording (SNR: {snr_ratio:.2f}). Standard masking (Alpha: {auto_alpha:.1f})")

        output_path = vocal_path.replace(".wav", "_automasked.wav")
        sf.write(output_path, v_clean, sr)
        return output_path

    def clean_backing_track(self, stem_path):
        """
        Cleans the 'Other' stem (Guitars/Keys) so we can boost it without boosting crowd.
        Uses Spectral Gating.
        """
        logging.info(f"Cleaning Instrument Track: {stem_path}")
        data, sr = sf.read(stem_path)
        
        with TerminalVisualizer("Filtering Stems   "):
            # 1. Gentle High Pass to remove mud
            sos = signal.butter(10, 80, 'hp', fs=sr, output='sos')
            data = signal.sosfilt(sos, data, axis=0)
            
            # 2. Spectral Gate (Moderate settings to keep guitar sustain)
            # We calculate an envelope and clamp down when volume drops (noise floor)
            abs_signal = np.abs(data)
            b, a = signal.butter(4, 5 / (sr / 2), 'low') # Slow envelope
            envelope = signal.filtfilt(b, a, abs_signal, axis=0)
            
            peak = np.max(envelope)
            threshold = peak * 0.15 # Gate everything below 15% volume
            
            gate_mask = np.ones_like(envelope)
            below_thresh = envelope < threshold
            
            # Don't silence completely, just reduce noise by 60%
            gate_mask[below_thresh] = 0.4 
            
            # Smooth the gate
            b_s, a_s = signal.butter(1, 10 / (sr / 2), 'low')
            gate_mask = signal.filtfilt(b_s, a_s, gate_mask, axis=0)
            
            data = data * gate_mask
        
        output_path = stem_path.replace(".wav", "_cleaned.wav")
        sf.write(output_path, data, sr)
        return output_path

    def apply_studio_reverb(self, audio_path, amount=0.15):
        """
        Adds a synthetic 'Plate' reverb using convolution to make vocals natural.
        """
        logging.info(f"Applying Studio Reverb (Wet: {amount})...")
        data, sr = sf.read(audio_path)
        
        with TerminalVisualizer("Convolving Reverb "):
            # Generate Impulse Response (Exponential Decay White Noise = Simple Plate)
            duration = 1.5 # seconds
            t = np.linspace(0, duration, int(sr * duration))
            decay = np.exp(-5 * t)
            noise = np.random.normal(0, 1, len(t))
            ir = noise * decay
            
            # Normalize IR
            ir = ir / np.max(np.abs(ir))
            
            # Convolve (Process per channel)
            wet_signal = np.zeros_like(data)
            for i in range(data.shape[1]):
                # mode='same' keeps length equal to input
                wet_signal[:, i] = signal.convolve(data[:, i], ir, mode='full')[:len(data)]
            
            # Blend Dry/Wet
            mixed = (data * 0.9) + (wet_signal * amount * 0.1) # amount is scaled down
        
        output_path = audio_path.replace(".wav", "_reverb.wav")
        sf.write(output_path, mixed, sr)
        return output_path

    def enhance_presence(self, audio_path, amount=0.5):
        """Simple EQ polish for the final vocal."""
        data, sr = sf.read(audio_path)
        # 3kHz boost for vocal clarity
        b, a = signal.iirpeak(3000, 2.0, fs=sr)
        enhanced = signal.lfilter(b, a, data, axis=0)
        mixed = (data * (1-amount*0.3)) + (enhanced * (amount*0.3))
        sf.write(audio_path, mixed, sr)
        return audio_path

    def mix_master(self, stems, output_file):
        """
        Recombines the processed stems into a final stereo master.
        """
        logging.info("Remixing stems into final master...")
        
        v_data, sr = sf.read(stems['vocals'])
        d_data, _ = sf.read(stems['drums'])
        b_data, _ = sf.read(stems['bass'])
        o_data, _ = sf.read(stems['other'])
        
        # Ensure lengths match
        min_len = min(len(v_data), len(d_data), len(b_data), len(o_data))
        v_data = v_data[:min_len]
        d_data = d_data[:min_len]
        b_data = b_data[:min_len]
        o_data = o_data[:min_len]
        
        with TerminalVisualizer("Final Mixdown     "):
            # "ClearStage" Studio Profile
            # 1. Vocals are now cleaner, so we can push them.
            # 2. Instruments (Other) are cleaned, so we can BOOST them (1.3x)
            # 3. Drums/Bass get a solid foundation.
            master = (v_data * 1.0) + (d_data * 1.3) + (b_data * 1.2) + (o_data * 1.3)
            
            # Soft Limiter / Saturation
            master = np.tanh(master * 1.1) # Slight drive
        
        sf.write(output_file, master, sr)
        logging.info(f"Done! Saved to {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Lupri ClearStage AI Engine")
    parser.add_argument("input", help="Path to the input audio file")
    parser.add_argument("--crowd", type=float, default=1.0, help="Target aggressiveness bias (default: 1.0)")
    parser.add_argument("--output", default="clearstage_output.wav", help="Output file path")
    
    args = parser.parse_args()
    
    engine = ClearStageEngine(use_gpu=True)
    
    # 1. Separate
    stems = engine.separate_sources(args.input)
    
    if stems:
        logging.info("--- Starting AI Auto-Analysis ---")
        
        # 2. Clean Vocals (Heavy Masking)
        stems['vocals'] = engine.apply_smart_masking(stems['vocals'], stems['other'], args.crowd)
        
        # 3. Clean Instruments (Gating) - New Step
        # This removes crowd form the 'guitar' stem so we can boost it later
        stems['other'] = engine.clean_backing_track(stems['other'])
        
        # 4. Polish Vocals (EQ + Reverb)
        stems['vocals'] = engine.enhance_presence(stems['vocals'], amount=0.6)
        stems['vocals'] = engine.apply_studio_reverb(stems['vocals'], amount=0.25)
        
        # 5. Master
        engine.mix_master(stems, args.output)