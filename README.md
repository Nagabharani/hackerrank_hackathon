# Multi-Modal Evidence Review Pipeline

This project is an autonomous "Multi-Modal Evidence Review" agent designed to evaluate damage claims for cars, laptops, and packages. It utilizes the powerful Gemini Vision-Language Model (VLM) to analyze both text claims and photographic evidence simultaneously, ensuring that visual evidence is the primary source of truth.

## Project Architecture

The codebase is organized as follows:
- `code/main.py`: The entry point script that orchestrates the data loading, API calls, and saving.
- `code/vlm_client.py`: Handles interaction with the Google Gemini API, encompassing the system prompt engineering and structured JSON enforcement.
- `code/data_processor.py`: Manages the reading and writing of CSV data and paths resolution for local images.
- `requirements.txt`: Project dependencies.

## Prompt Engineering Strategy

The prompt provided to the VLM is engineered with several strict constraints:
1. **Visual Primacy**: The model is instructed to treat the images as the absolute ground truth. Text provides context but never overrides the visual evidence.
2. **Schema Enforcement**: Utilizing Pydantic and Gemini's `response_schema` along with `response_mime_type="application/json"`, the VLM is strictly forced to return the expected 10-column output as valid JSON.
3. **Structured Decisions**: Output fields such as `claim_status` are restricted to specific enumerations (`supported`, `contradicted`, `lacks enough information`).

## Setup and Installation

1. Create a virtual environment (optional but recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Set up the Google Gemini API key:
   Set the `GEMINI_API_KEY` environment variable.
   ```bash
   # On Windows PowerShell
   $env:GEMINI_API_KEY="your_api_key_here"
   
   # On macOS/Linux
   export GEMINI_API_KEY="your_api_key_here"
   ```

## Running the Pipeline

Ensure that `claims.csv` is present in the project root directory alongside any referenced images (relative paths should match).

Run the pipeline:
```bash
python code/main.py
```

The script will process the claims, outputting progress to the console. It will automatically create/update `output.csv` with the required fields evaluated by the VLM.
