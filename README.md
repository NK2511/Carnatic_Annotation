# Carnatic Annotater (CAARVAI)

**CAARVAI** (Carnatic Automated Annotation and Raga-Weighted Vocal Analysis Interface) is a comprehensive pipeline for automated melodic analysis of Carnatic music. It integrates state-of-the-art source separation, neural pitch tracking, and hierarchical machine learning to bridge the gap between raw audio and musicological insights.

## 🚀 Key Features

- **High-Fidelity Vocal Extraction**: Utilizes **Demucs v4** to isolate lead vocals from complex polyphonic accompaniments (Violin, Mridangam, and Tanpura).
- **Neural Pitch Tracking**: Implements **CREPE** (tiny model) with Viterbi decoding for precise $f_0$ estimation, capturing the intricate oscillations of *gamakas*.
- **RWHTM Tonic Identification**: A novel **Raga-Weighted Harmonic Template Matching** algorithm that identifies the tonic by aligning the pitch distribution with raga-specific svara sets.
- **Unsupervised Motif Discovery**: Automated segmentation via iterative multi-scale windowing and taxonomic clustering (Species, Genus, and Family).
- **Hierarchical Raga Identification**: A two-stage LSTM architecture that models raga identity across micro-temporal (intra-motif) and macro-temporal (inter-motif) scales.
- **Zero-Shot Recognition**: A stability-based CNN classifier that identifies 72 Melakarta ragas without requiring raga-specific training data.

## 📁 Project Structure

```text
├── VocalAnnotator/          # Core analysis logic and research tools
│   ├── carnatic_functions.py # Shared library for pitch and motif analysis
│   ├── process_audio_folder.py # Main extraction and processing pipeline
│   ├── interactive_viewer.py  # GUI for motif exploration and labeling
│   └── VocalAnnotatorNew.ipynb # Main research and visualization workflow
├── YoutubeScraper/          # Tools for data collection
│   ├── youtube_mp3_downloader.py
│   └── batch_transcribe.py
├── Raagas/                  # Research corpus (CSVs only in repository)
└── demucs_env/              # Pre-configured environment for source separation
```

## 🛠️ Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/CarnaticAnnotater.git
   cd CarnaticAnnotater
   ```
2. Set up the environment:
   ```bash
   # It is recommended to use the provided demucs_env for source separation
   pip install -r requirements.txt
   ```

## 📖 Usage

### 1. Process a Raga Folder
To extract vocals and generate pitch CSVs for a new raga:
```bash
python process_audio_folder.py Raagas/Mayamalavagowlai
```

### 2. Consolidate and Normalize
To merge individual song CSVs into a master dataset with tonic normalization:
```bash
python VocalAnnotator/consolidate_crepe.py
```

### 3. Interactive Research
Open `VocalAnnotator/VocalAnnotatorNew.ipynb` in Jupyter to perform motif clustering, tonic alignment, and LSTM training.

## 📄 Documentation
For detailed technical information on the methodology, please refer to the research manuscript (available upon request).

## 🎓 Citation
If you use this tool in your research, please cite:
*Ranade et al. (2026), "CAARVAI: A Hierarchical Pipeline for Melodic Documentation and Raga Identification in Carnatic Music."*
