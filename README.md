# Radon Analysis DARTS

## Overview
Radon Analysis DARTS is a Python-based project focused on modeling underground soil radon gas emanation. This project leverages machine learning techniques and affinity clustering to analyze radon concentration levels efficiently and effectively.

## Features
- Implementation of machine learning models: `run-all-nbeats` and `run-all-dlinear`.
- Models used for comparison include those under the naming scheme `45_13`.
- Integration of affinity clustering models for advanced analysis.
- Comprehensive preprocessing, training, and evaluation pipeline.
- Visualization tools for results and model insights.

## Installation

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/abbaasalif/radon_analysis_darts.git
   cd radon_analysis_darts
   ```

2. **Set Up a Virtual Environment (Optional)**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Set Up the Dataset**:
   - Place the dataset in the `data/` directory (create this directory if it does not exist).
   - Ensure the dataset is properly formatted as specified in the `data/README.md` file (if available).

## Usage

1. **Run Preprocessing**:
   ```bash
   python preprocess_data.py
   ```

2. **Train N-BEATS Model**:
   ```bash
   python run-all-nbeats.py
   ```

3. **Train D-Linear Model**:
   ```bash
   python run-all-dlinear.py
   ```

4. **Train and Compare Models**:
   - Models named `45_13` are used for comparison. To run these models:
   ```bash
   python run_45_13_model.py
   ```

5. **Run Affinity Clustering**:
   - To train and evaluate affinity clustering models:
   ```bash
   python affinity_clustering.py
   ```

6. **Evaluate Results**:
   ```bash
   python evaluate.py --model checkpoints/best_model.pth
   ```

7. **Visualize Results**:
   ```bash
   python visualize.py
   ```

## File Structure
```
radon_analysis_darts/
├── data/                     # Directory for datasets
├── configs/                  # Configuration files
├── models/                   # Model architectures
├── preprocess_data.py        # Data preprocessing script
├── run-all-nbeats.py         # N-BEATS model training
├── run-all-dlinear.py        # D-Linear model training
├── run_45_13_model.py        # Models for comparison
├── affinity_clustering.py    # Affinity clustering model script
├── evaluate.py               # Evaluation script
├── visualize.py              # Visualization script
├── requirements.txt          # Python dependencies
└── README.md                 # Project documentation
```

## Configuration
- Configuration parameters for training, evaluation, and clustering are available in the YAML files in the `configs/` directory.

## Requirements
- Python 3.8 or higher
- See `requirements.txt` for a full list of dependencies.

## Contributions
Contributions are welcome! To contribute:
1. Fork the repository.
2. Create a branch for your feature or bug fix.
3. Submit a pull request.

## License
This project is licensed under the MIT License. See the `LICENSE` file for details.

## Acknowledgments
This project builds upon techniques discussed in the accompanying paper "Towards Modeling Underground Soil Radon Gas Emanation."

