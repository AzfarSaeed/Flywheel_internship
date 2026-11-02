TRAIN 3 PERCENTAGE DOWNSAMPLING VIEWER

Files:
- train3_downsampling_prompt.py: updated GUI script
- db_config.json: create this from db_config_template_train3.json and enter your PostgreSQL credentials
- requirements_train3_windows.txt: dependencies

Install (Windows PowerShell):
  python -m pip install -r requirements_train3_windows.txt

Run (prompts for downsampling percentage):
  python train3_downsampling_prompt.py

Example input:
  80.133

The entered percentage is the percentage of valid readings to REMOVE.
Allowed values: 0 <= percentage < 100. Decimal values are accepted.
The script uses per-bucket min/max envelopes for X_g and Y_g. This preserves
bucket peaks/troughs, but not every individual sample or exact waveform detail.
The runtime report prints source, retained and downsampled counts for each axis.

Force a fresh database query (ignore cache):
  python train3_downsampling_prompt.py --refresh

Optionally provide the percentage without an interactive prompt:
  python train3_downsampling_prompt.py --downsample-percent 80.133

Keep db_config.json beside the script. Do not share it because it contains
credentials.
