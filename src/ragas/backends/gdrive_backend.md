# Google Drive Backend for Ragas

The Google Drive backend allows you to store Ragas datasets and experiments in Google Sheets within your Google Drive. This provides a cloud-based, collaborative storage solution that's familiar to many users.

## Features

- **Cloud Storage**: Store your datasets and experiments in Google Drive
- **Collaborative**: Share and collaborate on datasets using Google Drive's sharing features
- **Google Sheets Format**: Data is stored in Google Sheets for easy viewing and editing
- **Automatic Structure**: Creates organized folder structure (datasets/ and experiments/)
- **Type Preservation**: Attempts to preserve basic data types (strings, numbers)
- **Multiple Authentication**: Supports both OAuth and Service Account authentication

## Installation

```bash
# Install with Google Drive dependencies
pip install "ragas[gdrive]"
```

## Setup

### 1. Google Cloud Project Setup

1. Go to the [Google Cloud Console](https://console.cloud.google.com/)
2. Create a new project or select an existing one
3. Enable the following APIs:
   - Google Drive API
   - Google Sheets API

### 2. Authentication Setup

Choose one of two authentication methods:

#### Option A: Service Account (Recommended)

1. In Google Cloud Console, go to "Credentials"
2. Click "Create Credentials" → "Service account"
3. Create the service account and download the JSON key file
4. Share your Google Drive folder with the service account email

*This is the preferred method as it works well for both scripts and production environments without requiring user interaction.*

#### Option B: OAuth 2.0 (Alternative for Interactive Use)

1. In Google Cloud Console, go to "Credentials"
2. Click "Create Credentials" → "OAuth client ID"
3. Choose "Desktop application"
4. Download the JSON file (save as `credentials.json`)

### 3. Google Drive Folder Setup

1. Create a folder in Google Drive for your Ragas data
2. Get the folder ID from the URL: `https://drive.google.com/drive/folders/FOLDER_ID_HERE`
3. If using Service Account, share the folder with the service account email

## Usage

### Basic Usage

```python
from ragas.dataset import Dataset
from pydantic import BaseModel


# Define your data model
class EvaluationRecord(BaseModel):
    question: str
    answer: str
    score: float


# Create dataset with Google Drive backend
dataset = Dataset(
    name="my_evaluation",
    backend="gdrive",
    config={
        "folder_id": "your_google_drive_folder_id",
        "service_account_file": "path/to/service-account.json",
    },
)

# Add data
record = EvaluationRecord(
    question="What is the capital of France?", answer="Paris", score=1.0
)
dataset.append(record.model_dump())

# The data is now stored in Google Sheets within your Drive folder
```

### Service Account Authentication

```python
dataset = Dataset(
    name="my_evaluation",
    backend="gdrive",
    config={
        "folder_id": "1ABC123def456GHI789jkl",
        "service_account_file": "/path/to/service-account.json",
    },
)
```

### OAuth Authentication

```python
dataset = Dataset(
    name="my_evaluation",
    backend="gdrive",
    config={
        "folder_id": "1ABC123def456GHI789jkl",
        "credentials_file": "/path/to/credentials.json",
    },
)
```

### Loading Existing Data

```python
# Load an existing dataset
dataset = Dataset.load(
    name="my_evaluation",
    backend="gdrive",
    config={
        "folder_id": "1ABC123def456GHI789jkl",
        "service_account_file": "/path/to/service-account.json",
    },
)

# Access the data
for record in dataset:
    print(f"Question: {record['question']}")
    print(f"Answer: {record['answer']}")
    print(f"Score: {record['score']}")
```

### Working with Experiments

```python
# After running experiments, results are stored automatically
from ragas import experiment


@experiment()
async def my_evaluation_experiment(row):
    # Your evaluation logic here
    response = await my_ai_system(row["question"])

    return {**row, "response": response, "experiment_name": "baseline_v1"}


# Run experiment - results will be saved to Google Drive
results = await my_evaluation_experiment.arun(dataset)
```

## Configuration Options

### Required Configuration

- `folder_id`: The Google Drive folder ID where data will be stored
- Authentication (one of):
  - `service_account_file`: Path to service account JSON file
  - `credentials_file`: Path to OAuth credentials JSON file

### Optional Configuration

```python
config = {
    "folder_id": "your_folder_id",
    "service_account_file": "service-account.json",
    # Optional settings
    "credentials_file": None,  # Alternative to service_account_file
    "token_file": "token.json",  # For OAuth token storage
    "scopes": [  # Google API scopes (defaults shown)
        "https://www.googleapis.com/auth/drive.file",
        "https://www.googleapis.com/auth/spreadsheets",
    ],
}
```

## File Organization

The backend automatically organizes your data in Google Drive:

```
Your Google Drive Folder/
├── datasets/
│   ├── my_evaluation.csv (as Google Sheets)
│   └── another_dataset.csv
└── experiments/
    ├── 20231201-143022-baseline_v1.csv
    ├── 20231201-144515-improved_model.csv
    └── comparison_results.csv
```

## Advanced Usage

### Appending vs Overwriting

```python
# Append to existing data (default)
dataset.append(new_record)

# Overwrite all data
dataset.clear()
dataset.append(new_record)
```

### Custom Sheet Names

```python
# Datasets are saved as: {name}.csv
# Experiments are saved as: {timestamp}-{experiment_name}.csv

dataset = Dataset(
    name="custom_name",  # Creates "custom_name.csv" in Google Sheets
    backend="gdrive",
    config=config,
)
```

### Batch Operations

```python
# Add multiple records at once
records = [
    {"question": "Q1", "answer": "A1", "score": 0.9},
    {"question": "Q2", "answer": "A2", "score": 0.8},
    {"question": "Q3", "answer": "A3", "score": 0.95},
]

for record in records:
    dataset.append(record)
```

## Troubleshooting

### Common Issues

1. **Folder access errors**
   - Verify the folder ID is correct
   - Check that the folder exists and is accessible

2. **Authentication errors**
   - Verify credential file paths are correct
   - Check that required APIs are enabled in Google Cloud Console
   - For OAuth: delete token file and re-authenticate
   - For Service Account: verify the JSON file is valid

3. **Permission errors**
   - Ensure your account has edit access to the folder
   - For service accounts: share the folder with the service account email
   - Check Google Drive sharing settings

4. **Import errors**
   - Install dependencies: `pip install "ragas[gdrive]"`
   - Verify all required packages are installed

### Getting Help

If you encounter issues:

1. Check error messages carefully for specific details
2. Verify your Google Cloud project setup
3. Test with a simple example first
4. Check the Google Drive API documentation for rate limits

## Limitations

- Google Sheets has a limit of 10 million cells per spreadsheet
- Complex nested objects are JSON-serialized as strings
- API rate limits may affect performance with large datasets
- Requires internet connection for all operations

## Examples

See `examples/gdrive_backend_example.py` for a complete working example.
