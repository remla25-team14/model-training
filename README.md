# Model Training Repository

This repository contains the machine learning model training pipeline for sentiment analysis of restaurant reviews. The data and models are managed using DVC (Data Version Control) with storage on DagsHub.

## Setup Instructions

### Prerequisites
- Python 3.11 or higher
- Git
- Make
- Conda (for environment management)
- DagsHub account (create one at https://dagshub.com if you don't have it)

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

### 3. DVC and DagsHub Setup
The repository comes with pre-configured DVC settings. Follow these steps to set up DagsHub and DVC:

1. First, log in to DagsHub (you'll need to create an account if you don't have one):
```bash
# This will open a browser window for authentication
dagshub login
```

2. Configure DVC with DagsHub storage credentials:
```bash
# Configure DVC with DagsHub S3 storage credentials
dvc remote modify storage endpointurl https://dagshub.com/api/v1/repo-buckets/s3/s.hakimi
dvc remote modify storage access_key_id 04dc266bcc211e1d07d5fdfa4f9c999979cf7bb3
dvc remote modify storage secret_access_key 04dc266bcc211e1d07d5fdfa4f9c999979cf7bb3
dvc remote modify storage region us-east-1

# After configuring, you can pull the data
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
└── model_training/                 # Source code for model training
```

### Data Version Control
- All data files are tracked using DVC
- Data is stored on DagsHub S3 storage
- Git is used only for code version control
- DVC configuration is version controlled and will be automatically available after cloning

### Pipeline Reproduction
The model training process is defined as a DVC pipeline in `dvc.yaml`. To reproduce the entire pipeline:

```bash
# Reproduce the pipeline
dvc repro
```

This command will:
1. Check if any pipeline stages need to be rerun based on changes to their dependencies
2. Process the raw restaurant reviews data
3. Generate features using the Bag of Words vectorizer
4. Train the sentiment analysis model
5. Save the model and artifacts

If no files have changed, `dvc repro` will indicate that the pipeline is up to date. If any input files or code have changed, only the affected stages will be rerun.

### Experiment Management
DVC provides powerful experiment tracking capabilities. Here's how to use them:

```bash
# View all experiments and their metrics
dvc exp show

# This will display a table showing:
# - Experiment names and creation times
# - Metrics for each experiment (accuracy, precision, recall, F1)
# - Parameter values used
# - File hashes for tracking changes
```

To run new experiments with different parameters:

```bash
# Modify parameters in params.yaml, then run:
dvc exp run

# Or run with parameter modifications directly:
dvc exp run --set-param training.random_state=123
dvc exp run --set-param training.test_size=0.3
```

The experiments are tracked and can be:
- Compared using `dvc exp show`
- Applied using `dvc exp apply <experiment-name>`
- Shared with others using `dvc exp push` and `dvc exp pull`

This helps in:
- Tracking different model configurations
- Comparing performance metrics
- Reproducing successful experiments
- Collaborating on model improvements

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
