"""One command for the complete CondOT and VP-SDE evaluation suites."""
import argparse,subprocess,sys
from pathlib import Path
def main():
 p=argparse.ArgumentParser(); p.add_argument('--device',default='cuda'); p.add_argument('--batch-size',type=int,default=1024); p.add_argument('--steps',type=int,default=30); p.add_argument('--repeats',type=int,default=100); p.add_argument('--reference-steps',type=int,default=512); p.add_argument('--output-dir',default='outputs/all'); a=p.parse_args(); root=Path(__file__).resolve().parent
 common=['--device',a.device,'--batch-size',str(a.batch_size),'--steps',str(a.steps),'--repeats',str(a.repeats),'--reference-steps',str(a.reference_steps)]
 subprocess.run([sys.executable,str(root/'benchmark'/'run_condot_all.py'),*common,'--output-dir',str(Path(a.output_dir)/'condot')],check=True)
 subprocess.run([sys.executable,str(root/'benchmark'/'run_vp_all.py'),*common,'--output-dir',str(Path(a.output_dir)/'vpsde')],check=True)
if __name__=='__main__': main()
