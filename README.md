# Model Training Repository

This repository contains the machine learning model training pipeline for sentiment analysis of restaurant reviews. The data and models are managed using DVC (Data Version Control) with storage on DagsHub.

## Setup Instructions

### Prerequisites
- Python 3.11 or higher
- Git
- Make

### 1. Clone and Setup Repository
```bash
# Clone the repository
git clone https://github.com/remla25-team14/model-training.git
cd model-training

# Install dependencies using Make
make setup
```

### 2. DVC Setup
The DVC remote storage is already configured in the repository. The configuration points to DagsHub S3 storage and includes all necessary credentials. To get the data:

```bash
# Pull all data and model files
dvc pull
```

This will download:
- Raw data files in `data/raw/`
- Processed features in `data/processed/`
- Trained model in `models/`
- Model artifacts in `model_service_artifacts/`

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