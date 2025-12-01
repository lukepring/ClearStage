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
        
        # Denormalize? Demucs output is usually roughly correctly scaled, 
        # but we might need to restore amplitude if we strictly normalized. 
        # However, htdemucs usually outputs proper amplitude. We will save directly.
        
        sources = sources * ref.std() + ref.mean()

        for source, name in zip(sources, source_names):
            source = source.cpu().numpy().T # Back to (Time, Channels)
            out_path = os.path.join(save_dir, f"{name}.wav")
            sf.write(out_path, source, sr)
            stem_paths[name] = out_path
            
        logging.info("Source separation complete.")
        return stem_paths

    def reduce_crowd_noise(self, other_stem_path, intensity=0.5):
        """
        Crowd noise usually lives in the 'other' stem in Demucs output.
        We apply Spectral Gating here to reduce the 'wash' of the crowd.
        """
        logging.info(f"Applying Crowd Suppression (Intensity: {intensity}) to {other_stem_path}")
        
        data, samplerate = sf.read(other_stem_path)
        
        # Simple spectral gating simulation
        # 1. High Pass Filter (Remove rumble)
        sos = signal.butter(10, 150, 'hp', fs=samplerate, output='sos')
        # axis=0 ensures we filter along time, not across channels
        filtered = signal.sosfilt(sos, data, axis=0)
        
        # 2. Hard clamp on background noise (Simple Gate)
        # reducing volume of 'other' stem based on intensity
        processed_data = filtered * (1.0 - (intensity * 0.6))
        
        output_path = other_stem_path.replace(".wav", "_clean.wav")
        sf.write(output_path, processed_data, samplerate)
        return output_path

    def perform_spectral_subtraction(self, main_signal, noise_signal, intensity):
        """
        Subtracts the frequency content of the noise_signal (crowd) from the main_signal (vocals).
        This is far superior to gating when the noise is louder than the signal.
        """
        # Ensure correct shape (Samples, Channels)
        if main_signal.ndim == 1: main_signal = main_signal[:, np.newaxis]
        if noise_signal.ndim == 1: noise_signal = noise_signal[:, np.newaxis]
        
        # Parameters for STFT
        nperseg = 2048
        noverlap = 1536
        
        output_audio = np.zeros_like(main_signal)
        
        # Process per channel
        for ch in range(main_signal.shape[1]):
            # STFT: Convert time domain to frequency domain
            f, t, Zxx_main = signal.stft(main_signal[:, ch], nperseg=nperseg, noverlap=noverlap)
            f, t, Zxx_noise = signal.stft(noise_signal[:, ch], nperseg=nperseg, noverlap=noverlap)
            
            # Get magnitudes (volume at each frequency)
            mag_main = np.abs(Zxx_main)
            mag_noise = np.abs(Zxx_noise)
            phase_main = np.angle(Zxx_main)
            
            # SUBTRACTION: Remove the noise profile from the main signal
            # We multiply noise by intensity to control how aggressive we are
            mag_clean = np.maximum(0, mag_main - (mag_noise * intensity))
            
            # Reconstruct complex signal
            Zxx_clean = mag_clean * np.exp(1j * phase_main)
            
            # Inverse STFT: Convert back to time domain
            _, clean_time = signal.istft(Zxx_clean, nperseg=nperseg, noverlap=noverlap)
            
            # Handle length mismatch due to padding/windowing
            target_len = len(main_signal[:, ch])
            if len(clean_time) > target_len:
                clean_time = clean_time[:target_len]
            elif len(clean_time) < target_len:
                clean_time = np.pad(clean_time, (0, target_len - len(clean_time)))
                
            output_audio[:, ch] = clean_time
            
        return output_audio

    def process_vocals(self, vocal_stem_path, other_stem_path, clarity=0.5, crowd_removal=0.0):
        """
        Enhances vocals with EQ, Compression, and Spectral Subtraction.
        """
        logging.info(f"Enhancing Vocals (Clarity: {clarity}, Crowd Removal: {crowd_removal})...")
        data, samplerate = sf.read(vocal_stem_path)
        
        # Load the "Other" stem (Crowd) to use as a noise profile
        noise_profile, _ = sf.read(other_stem_path)

        # 1. SPECTRAL SUBTRACTION (New Feature)
        if crowd_removal > 0:
            logging.info("Applying Spectral Subtraction using 'Other' stem as noise profile...")
            # We use the raw 'other' stem to identify crowd frequencies in the vocal stem
            data = self.perform_spectral_subtraction(data, noise_profile, crowd_removal)

        # 2. High Pass Filter (Remove mud/rumble from crowd)
        # 100Hz cutoff is standard for vocals to remove stage noise
        sos = signal.butter(10, 100, 'hp', fs=samplerate, output='sos')
        data = signal.sosfilt(sos, data, axis=0)

        # 3. Vocal Gate (Cleanup: Remove residual noise between phrases)
        if crowd_removal > 0:
             # Calculate amplitude envelope
             abs_signal = np.abs(data)
             
             # Smooth envelope
             b, a = signal.butter(4, 10 / (samplerate / 2), 'low') 
             envelope = signal.filtfilt(b, a, abs_signal, axis=0)
             
             peak = np.max(envelope)
             # More gentle threshold now that we did spectral subtraction
             threshold = peak * (0.01 + (0.10 * crowd_removal))
             
             gate_mask = np.ones_like(envelope)
             below_thresh = envelope < threshold
             
             floor_gain = 1.0 - (0.8 * crowd_removal) 
             gate_mask[below_thresh] = floor_gain
             
             b_smooth, a_smooth = signal.butter(1, 20 / (samplerate / 2), 'low')
             gate_mask = signal.filtfilt(b_smooth, a_smooth, gate_mask, axis=0)
             
             data = data * gate_mask

        # 4. Presence Boost
        b, a = signal.iirpeak(3000, 2.0, fs=samplerate)
        enhanced = signal.lfilter(b, a, data, axis=0) * (1 + clarity)
        
        final_vocals = (data * 0.7) + (enhanced * 0.3)
        
        output_path = vocal_stem_path.replace(".wav", "_processed.wav")
        sf.write(output_path, final_vocals, samplerate)
        return output_path

    def mix_master(self, stems, output_file):
        """
        Recombines the processed stems into a final stereo master.
        """
        logging.info("Remixing stems into final master...")
        
        v_data, sr = sf.read(stems['vocals'])
        d_data, _ = sf.read(stems['drums'])
        b_data, _ = sf.read(stems['bass'])
        o_data, _ = sf.read(stems['other'])
        
        # Ensure lengths match (safe truncate)
        min_len = min(len(v_data), len(d_data), len(b_data), len(o_data))
        v_data = v_data[:min_len]
        d_data = d_data[:min_len]
        b_data = b_data[:min_len]
        o_data = o_data[:min_len]
        
        # Mixing Levels (Studio Profile)
        # Vocals: +10%
        # Drums: 0%
        # Bass: -10%
        # Other (Crowd/Synth): -20%
        master = (v_data * 1.1) + (d_data * 1.0) + (b_data * 0.9) + (o_data * 0.8)
        
        # Soft Limiter (Tanh)
        master = np.tanh(master) 
        
        sf.write(output_file, master, sr)
        logging.info(f"Done! Saved to {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Lupri ClearStage AI Engine")
    parser.add_argument("input", help="Path to the input audio file")
    parser.add_argument("--crowd", type=float, default=0.5, help="Crowd removal intensity (0.0 - 1.0)")
    parser.add_argument("--vocal", type=float, default=0.5, help="Vocal clarity boost (0.0 - 1.0)")
    parser.add_argument("--output", default="clearstage_output.wav", help="Output file path")
    
    args = parser.parse_args()
    
    engine = ClearStageEngine(use_gpu=True)
    
    # 1. Separate
    stems = engine.separate_sources(args.input)
    
    if stems:
        # Save original crowd path for spectral subtraction reference
        # The 'other' stem is the crowd noise we want to subtract from the vocals
        raw_crowd_path = stems['other']

        # 2. Process
        stems['other'] = engine.reduce_crowd_noise(stems['other'], args.crowd)
        
        # We pass the RAW crowd path to process_vocals so it knows what to subtract
        stems['vocals'] = engine.process_vocals(stems['vocals'], raw_crowd_path, args.vocal, args.crowd)
        
        # 3. Master
        engine.mix_master(stems, args.output)
