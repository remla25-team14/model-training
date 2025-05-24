# Model Training Repository

This repository contains the machine learning model training pipeline for sentiment analysis of restaurant reviews. The data and models are managed using DVC (Data Version Control) with storage on DagsHub.

## Setup Instructions

### Prerequisites
- Python 3.11 or higher
- Git
- Make
- Conda (for environment management)

### 1. Environment Setup
First, create and activate a conda environment:

```bash
# Create conda environment
conda create -n model-training-2 python=3.11

# Activate the environment
conda activate model-training-2

# Install pip in the conda environment
conda install pip
```

### 2. Clone and Setup Repository
```bash
# Clone the repository
git clone https://github.com/remla25-team14/model-training.git
cd model-training

# Install dependencies using Make
make setup
```

### 3. DVC Setup
No manual DVC configuration is needed! The repository comes with pre-configured DVC settings that are version controlled. Simply run:

```bash
# Pull all data and model files
dvc pull
```

This will download:
- Raw data files in `data/raw/`
- Processed features in `data/processed/`
- Trained model in `models/`
- Model artifacts in `model_service_artifacts/`

The DVC remote storage (DagsHub) and all necessary credentials are already configured in the repository and will be available automatically after cloning.

### Repository Structure
```
model-training/
├── data/
│   ├── raw/              # Raw restaurant review datasets
│   └── processed/        # Processed features and vectorizer
├── models/               # Trained model files
├── model_service_artifacts/ # Model service deployment artifacts
└── src/                 # Source code for model training
```

### Data Version Control
- All data files are tracked using DVC
- Data is stored on DagsHub S3 storage
- Git is used only for code version control
- DVC configuration is version controlled and will be automatically available after cloning

### Notes
- The repository uses a Makefile for standardized setup and operations
- DVC credentials are pre-configured and committed in the repository
- For any issues with data access, ensure you have the correct permissions on DagsHub

## Development

To contribute to this repository:
1. Create a new branch for your feature
2. Make your changes
3. Use DVC to track any new data files: `dvc add path/to/file`
4. Commit both git and DVC changes
5. Push to GitHub and DVC storage:
   ```bash
   git push
   dvc push
   ```