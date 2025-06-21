# Model Training Repository

This repository contains the machine learning model training pipeline for sentiment analysis of restaurant reviews. The data and models are managed using DVC (Data Version Control) with storage on DagsHub.

## Project Quality Metrics

Pylint: 0.00/10
Coverage: 75%

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
Note: This repository should be cloned into the `remla/` directory alongside other team repositories like `lib-ml`:

```bash
# Navigate to the remla directory (create it if it doesn't exist)
mkdir -p ~/remla && cd ~/remla

# Clone the repository
git clone https://github.com/remla25-team14/model-training.git
cd model-training

# Install dependencies using Make
make setup
```

### 3. DVC and DagsHub Setup
The repository comes with pre-configured DVC settings that point to our team's DagsHub storage. No account creation or login is needed - the credentials are already set up in the repository.

To get started with the data:

```bash
# If you have any existing DVC config, remove it first to avoid conflicts
rm -f .dvc/config*

# Configure DVC with our team's DagsHub storage credentials (all in one command)
dvc remote add -d storage s3://dvc_cloud_setup && dvc remote modify storage endpointurl https://dagshub.com/api/v1/repo-buckets/s3/s.hakimi && dvc remote modify storage access_key_id 04dc266bcc211e1d07d5fdfa4f9c999979cf7bb3 && dvc remote modify storage secret_access_key 04dc266bcc211e1d07d5fdfa4f9c999979cf7bb3 && dvc remote modify storage region us-east-1 && dvc pull
```

This will download:
- Raw data files in `data/raw/`
- Processed features in `data/processed/`
- Trained model in `models/`
- Model artifacts in `model_service_artifacts/`

Note: These credentials are read-only and specifically configured for this project. For contributing your own data or models, please contact the team for write access.

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

### Versioned Model Releases (GitHub Releases)

This repository supports versioned model releases using GitHub Releases. The process involves two steps:

#### Step 1: Prepare the Version
First, manually update the version in `version.txt`:
```bash
# Edit version.txt to the desired version (e.g., 0.1.3)
echo "0.1.3" > version.txt
git add version.txt
git commit -m "Bump version to 0.1.3"
git push origin your-branch
```

#### Step 2: Create the Release
When you push a version tag that matches the version in `version.txt`, the CI workflow will:

1. **Run the full DVC pipeline** to train the model and prepare artifacts
2. **Create a new GitHub Release** with the tag (e.g., `v0.1.3`)
3. **Attach the model artifacts** as release assets:
   - `model_service_artifacts/c1_BoW_Sentiment_Model.pkl`
   - `model_service_artifacts/c2_Classifier_v1.pkl`
4. **Automatically bump the version** in `version.txt` for the next release

#### How to trigger a release

```bash
# 1. Make sure your branch is up to date and version.txt is committed
git push origin your-branch

# 2. Create a version tag that matches version.txt
# If version.txt contains "0.1.3", create tag "v0.1.3"
git tag v0.1.3
git push origin v0.1.3
```

This will trigger the `train.yml` workflow and create a new release. You can find the released model files in the [GitHub Releases](../../releases) section of the repository.

#### Important Notes
- The tag must match the version in `version.txt` (e.g., if version.txt has `0.1.3`, use tag `v0.1.3`)
- The workflow will also run on branch pushes for CI/testing, but only tag pushes create a GitHub Release
- The version in `version.txt` is automatically incremented after each release
- You can always download the latest released model from the Releases page for deployment or inference

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

## Rolling Back to a Previous Pipeline Version

If you need to restore your code, parameters, and DVC-tracked data/models to a previous state (for example, to reproduce an old experiment or recover from a bad change), follow these steps:

1. **Find the Commit or Tag You Want to Roll Back To**
   ```bash
   git log --oneline
   # or, to see tags:
   git tag
   ```

2. **Checkout the Desired Commit or Tag**
   ```bash
   # Replace <commit-hash> or <tag> with your target
   git checkout <commit-hash>
   # or
   git checkout <tag>
   ```

3. **Restore DVC-Trained Artifacts for That Version**
   ```bash
   dvc checkout
   ```

   > This will update your workspace to match the exact data, models, and artifacts tracked at that commit.

4. **(Optional) Fetch Data/Models from Remote Storage**
   If you don’t have the required files locally, pull them from remote:
   ```bash
   dvc pull
   ```

5. **(Optional) Return to the Latest Version**
   To go back to your latest branch:
   ```bash
   git checkout <your-branch>
   dvc checkout
   ```

**Notes:**
- This process restores both code and all DVC-tracked files (data, models, metrics, etc.) to the exact state of the chosen commit.
- You can use this to reproduce any past experiment or recover from mistakes.
- If you use DVC experiments (`dvc exp`), you can also roll back/apply experiments with `dvc exp apply <exp-name>`.

### Model Metrics and Evaluation

The pipeline generates accuracy-related metrics in JSON format across multiple stages:

#### Training Metrics (`reports/metrics.json`)
Generated by the `train` stage, includes:
- **Accuracy**: Model accuracy on test set
- **Confusion Matrix**: Detailed classification results

#### Evaluation Metrics (`reports/evaluation.json`) 
Generated by the `evaluate` stage, includes:
- **Accuracy**: Overall model accuracy
- **Precision**: Precision score for sentiment classification
- **Recall**: Recall score for sentiment classification  
- **F1**: F1 score combining precision and recall

These metrics are automatically tracked by DVC and can be compared across different experiments using `dvc exp show`.

## ML Testing

| Category          | Location               | Status         | Notes |
|-------------------|------------------------|----------------|-------|
| Feature & Data    | `tests/test_feat_data.py` | Done           |
| Model Development | `tests/test_model_dev.py` | Done           |
| ML Infrastructure | `tests/test_ml_infra.py` | Done           |
| Monitoring        | `tests/test_monitoring.py` | Done           |
| Metamorphic       | `tests/test_metamorphic.py` | 🚧 placeholder |

### Run the tests
```bash
# 1) install dev dependencies
pip install -e .

# 2) run tests separately (switch between the 5 .py files names)
pytest tests/test_feat_data.py -q

# 3) run all tests with coverage + ML-Test-Score
pytest -q
```

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
