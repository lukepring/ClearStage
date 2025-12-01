import os
import argparse
import logging
import soundfile as sf
import numpy as np
import torch
import torchaudio
from scipy import signal

# --- CONFIGURATION ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

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
        # shifts=1, split=True for memory efficiency on GPU/MPS
        sources = self.demucs_apply.apply_model(model, wav_input, shifts=1, split=True, overlap=0.25, progress=True)[0]
        
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
        Instead of gating volume, this compares the spectrograms of the Vocal and Crowd stems.
        It calculates a 'probability mask' for every frequency bin.
        If a frequency is dominated by the crowd, it is suppressed.
        """
        logging.info(f"AI Auto-Masking initialized (Aggressiveness target: {aggressiveness})...")
        
        # Load audio
        v_wav, sr = sf.read(vocal_path)
        c_wav, _ = sf.read(crowd_path)
        
        # Ensure tensor format (Channels, Time)
        v_tensor = torch.tensor(v_wav).t().to(self.device)
        c_tensor = torch.tensor(c_wav).t().to(self.device)
        
        # Handle length mismatch
        min_len = min(v_tensor.shape[1], c_tensor.shape[1])
        v_tensor = v_tensor[:, :min_len]
        c_tensor = c_tensor[:, :min_len]
        
        # 1. Compute Spectrograms (STFT)
        # Using a window size of 2048 samples (approx 46ms) for good frequency resolution
        n_fft = 2048
        hop_length = 512
        window = torch.hann_window(n_fft).to(self.device)
        
        # STFT results in complex numbers (Magnitude + Phase)
        v_spec = torch.stft(v_tensor, n_fft=n_fft, hop_length=hop_length, window=window, return_complex=True)
        c_spec = torch.stft(c_tensor, n_fft=n_fft, hop_length=hop_length, window=window, return_complex=True)
        
        # 2. Compute Magnitudes (Energy)
        v_mag = torch.abs(v_spec)
        c_mag = torch.abs(c_spec)
        
        # 3. AUTO-CALIBRATION (The "Listening" part)
        # We calculate the average energy ratio to see how loud the crowd is overall
        # If crowd is super loud, we automatically bump up the suppression
        avg_v = torch.mean(v_mag)
        avg_c = torch.mean(c_mag)
        snr_ratio = avg_v / (avg_c + 1e-6)
        
        # Dynamic Auto-Tuning of Alpha (Sensitivity)
        # If SNR is low (crowd is loud), alpha goes UP to separate harder.
        # If SNR is high (vocals are loud), alpha goes DOWN to preserve quality.
        auto_alpha = 1.0
        if snr_ratio < 0.5: # Crowd is 2x louder than vocals
            auto_alpha = 4.0 * aggressiveness
            logging.info(f"  -> High Crowd Noise Detected (SNR: {snr_ratio:.2f}). engaging heavy masking (Alpha: {auto_alpha:.1f})")
        elif snr_ratio < 1.0: # Crowd is slightly louder
            auto_alpha = 2.0 * aggressiveness
            logging.info(f"  -> Moderate Crowd Noise Detected. Balancing mask (Alpha: {auto_alpha:.1f})")
        else: # Vocals are louder
            auto_alpha = 1.2 * aggressiveness
            logging.info(f"  -> Clean Recording Detected. Light masking only (Alpha: {auto_alpha:.1f})")

        # 4. Construct Soft Mask (Wiener Filter approximation)
        # Mask = Vocal_Energy / (Vocal_Energy + (Crowd_Energy * Alpha))
        # This creates a value between 0.0 and 1.0 for EVERY frequency bin
        mask = v_mag / (v_mag + (c_mag * auto_alpha) + 1e-8)
        
        # 5. Apply Soft Thresholding (Sigmoid-like behavior)
        # This pushes uncertain frequencies (0.5) down to silence, cleaning the sound
        mask = mask ** 2 
        
        # 6. Apply Mask to Original Vocal Phase
        v_spec_clean = v_spec * mask
        
        # 7. Inverse STFT (Rebuild Audio)
        v_clean_tensor = torch.istft(v_spec_clean, n_fft=n_fft, hop_length=hop_length, window=window, length=min_len)
        
        # Back to numpy for saving
        v_clean = v_clean_tensor.cpu().numpy().T
        
        output_path = vocal_path.replace(".wav", "_automasked.wav")
        sf.write(output_path, v_clean, sr)
        return output_path

    def enhance_presence(self, audio_path, amount=0.5):
        """Simple EQ polish for the final vocal."""
        data, sr = sf.read(audio_path)
        # Axis=0 fixed for scipy signal processing
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
        
        # For the backing, we use the separated drums/bass, but we keep the 'other' stem
        # volume VERY low because that's where the crowd lives.
        o_data, _ = sf.read(stems['other'])
        
        # Ensure lengths match
        min_len = min(len(v_data), len(d_data), len(b_data), len(o_data))
        v_data = v_data[:min_len]
        d_data = d_data[:min_len]
        b_data = b_data[:min_len]
        o_data = o_data[:min_len]
        
        # "ClearStage" Mix Profile
        # We assume the vocals are now super clean but maybe thin, so we boost them.
        master = (v_data * 1.2) + (d_data * 1.0) + (b_data * 1.0) + (o_data * 0.4)
        
        # Soft Limiter
        master = np.tanh(master) 
        
        sf.write(output_file, master, sr)
        logging.info(f"Done! Saved to {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Lupri ClearStage AI Engine")
    parser.add_argument("input", help="Path to the input audio file")
    # We kept the arguments for backward compatibility, but they act as 'biases' now
    parser.add_argument("--crowd", type=float, default=1.0, help="Target aggressiveness bias (default: 1.0)")
    parser.add_argument("--output", default="clearstage_output.wav", help="Output file path")
    
    args = parser.parse_args()
    
    engine = ClearStageEngine(use_gpu=True)
    
    # 1. Separate
    stems = engine.separate_sources(args.input)
    
    if stems:
        # 2. AI Auto-Masking
        # We pass the RAW crowd stem as the reference for what to remove
        logging.info("--- Starting AI Auto-Analysis ---")
        stems['vocals'] = engine.apply_smart_masking(stems['vocals'], stems['other'], args.crowd)
        
        # Polish
        stems['vocals'] = engine.enhance_presence(stems['vocals'], amount=0.5)
        
        # 3. Master
        engine.mix_master(stems, args.output)