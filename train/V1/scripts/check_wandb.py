"""One online preparation-only run: scalar checks, no dataset or model upload."""
import json
from pathlib import Path
from path_config import RUN_ROOT
import time
import wandb

root = RUN_ROOT
run = wandb.init(project='heterovla-pi05', entity='huph7-sun-yat-sen-university',
                 name=time.strftime('preparation-check-%Y%m%d-%H%M%S'),
                 job_type='preparation', group='environment-check',
                 config={'purpose': 'environment_logging_check', 'training_started': False},
                 settings=wandb.Settings(init_timeout=60, disable_code=True))
run.log({'preparation/check': 1, 'preparation/training_steps': 0})
report = {'online_logging_verified': True, 'run_id': run.id, 'run_url': run.url,
          'entity': run.entity, 'project': run.project, 'training_started': False}
run.finish()
(root / 'reports').mkdir(parents=True, exist_ok=True)
(root / 'reports/wandb_check.json').write_text(json.dumps(report, indent=2))
print(json.dumps(report))
