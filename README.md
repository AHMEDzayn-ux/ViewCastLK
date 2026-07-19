@'
# ViewCastLK — YouTube API Data Collection Prototype

**Contributor:** Asfi Ahamed  
**Branch:** `feature/asfi-youtube-api-sample`

## Branch Purpose

This branch preserves my individual YouTube Data API collection prototype created during the early stage of the ViewCastLK project.

The main data-collection responsibility was later reassigned to Sabith. Therefore, this branch is retained only as:

- Evidence of my individual contribution
- A record of the initial API investigation and testing
- Supporting material for the project logbook and demonstrations
- A reference prototype for the team

> **Important:** This branch is not intended to be merged into `main` or used as the final production data collector.

## Project Background

ViewCastLK is a data-driven system intended to collect historical information about videos published by Sri Lankan YouTube channels and use that information to forecast future viewership trends.

This prototype explored how the YouTube Data API v3 could be used to retrieve channel details, uploaded-video information, video metadata, and engagement statistics.

## Repository Contents

```text
ViewCastLK/
├── .env.example
├── .gitignore
├── README.md
└── asfi_data_collection_sample/
    ├── collect_youtube_data_sample.py
    └── youtube_data_sample.csv
```

### `collect_youtube_data_sample.py`

The Python prototype used to:

- Connect to the YouTube Data API v3
- Retrieve YouTube channel information
- Obtain upload-playlist information
- Retrieve a small sample of recently uploaded videos
- Collect video metadata and engagement statistics
- Save the collected results to a CSV file

### `youtube_data_sample.csv`

A sample CSV output generated while testing the collector with real YouTube API data.

### `.env.example`

Shows the environment variable required by the script without exposing a real API key.

### `.gitignore`

Prevents secrets, virtual environments, cache files, and temporary files from being committed.

## Requirements

- Python 3.11 or later
- A valid YouTube Data API v3 key
- Internet access
- YouTube Data API v3 enabled in a Google Cloud project

## Installation and Setup

### 1. Open the project folder

```powershell
cd D:\projects\viewcastlk\ViewCastLK
```

### 2. Confirm the correct branch

```powershell
git branch --show-current
```

Expected output:

```text
feature/asfi-youtube-api-sample
```

### 3. Create a Python virtual environment

```powershell
python -m venv .venv
```

### 4. Activate the virtual environment

```powershell
.\.venv\Scripts\Activate.ps1
```

After activation, the terminal should show `(.venv)` at the beginning of the command line.

### 5. Upgrade pip

```powershell
python -m pip install --upgrade pip
```

### 6. Install the required Python packages

```powershell
python -m pip install -r requirements.txt
```

## API-Key Configuration

### 1. Create a local `.env` file

Copy the example file:

```powershell
Copy-Item .env.example .env
```

### 2. Add the YouTube API key

Open `.env` and replace the placeholder:

```text
YOUTUBE_API_KEY=your_actual_youtube_api_key
```

Do not add quotation marks unless required, and do not add extra PowerShell commands to the file.

The real `.env` file is ignored by Git and must never be committed or shared publicly.

## Running the Prototype

From the repository root, run:

```powershell
python .\asfi_data_collection_sample\collect_youtube_data_sample.py
```

The script connects to the YouTube Data API, retrieves the configured channel and video information, and writes the collected records to a CSV output.

The committed sample output is available at:

```text
asfi_data_collection_sample/youtube_data_sample.csv
```

## Work Completed

The following tasks were completed as part of this prototype:

- Created and configured a Google Cloud project for ViewCastLK
- Enabled the YouTube Data API v3
- Created an API key and applied API restrictions
- Set up Python 3.11 and a virtual environment
- Installed the required YouTube API and environment-variable packages
- Configured secure API-key loading using `.env`
- Tested YouTube channel-information retrieval
- Retrieved upload-playlist and recent-video information
- Collected video metadata and engagement statistics
- Generated a CSV file containing sample YouTube data
- Investigated repeated snapshot collection for viewership tracking
- Fixed an API-key loading problem
- Corrected an invalid YouTube channel handle during testing
- Verified the generated data using VS Code and CSV output

## Prototype Limitations

This code represents an initial API experiment rather than the final ViewCastLK collection system.

Current limitations include:

- Uses a limited test set of channels and videos
- Does not represent every Sri Lankan YouTube channel
- Is not the final automated three-hour collection pipeline
- Is not connected to the final Supabase/PostgreSQL database
- Does not include the team member’s later production collector
- May require changes before use in a deployed environment

The YouTube API does not directly identify the physical country in which every video was created. Sri Lankan association must instead be estimated using channel information, known channel lists, channel-country metadata where available, and other project rules.

## Task Reassignment

The initial YouTube API setup, prototype development, testing, debugging, and sample-data generation were completed by Asfi Ahamed.

After the team reviewed the project responsibilities, the main data-collection implementation was reassigned to Sabith. Asfi’s responsibility then shifted toward investigating and planning periodic backups for the project’s Supabase/PostgreSQL database.

For this reason:

- This branch documents Asfi’s completed early data-collection work
- Sabith’s later collector is not included here
- This branch should not be merged into the production `main` branch
- Any production changes should follow the team’s agreed branch and review process

## Security Notes

The following must not be committed:

```text
.env
.venv/
__pycache__/
*.pyc
API keys
credentials
database passwords
```

Only `.env.example`, containing a safe placeholder, should be included in Git.

## Status

**Status:** Prototype completed and retained for individual contribution evidence  
**Production use:** No  
**Merge into `main`:** No  
**Current main collector owner:** Sabith
'@ | Set-Content .\README.md -Encoding utf8