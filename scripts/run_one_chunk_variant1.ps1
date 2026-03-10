Set-Location 'C:\Users\paree\Documents\Analyzing Rationale of LLMs'
$env:CHUNK_SIZE = '10'
$env:ONE_CHUNK_ONLY = '1'
python scripts\run_full_then_nullfix.py
