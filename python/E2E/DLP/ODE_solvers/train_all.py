"""One command to train matched CondOT and VP-SDE toy model suites."""
import argparse, subprocess, sys
from pathlib import Path

def main():
    p=argparse.ArgumentParser(); p.add_argument('--device',default='cuda'); p.add_argument('--train-steps',type=int,default=3200); p.add_argument('--batch-size',type=int,default=1024); p.add_argument('--checkpoint-dir',default='checkpoints'); a=p.parse_args()
    root=Path(__file__).resolve().parent; d=root/a.checkpoint_dir; d.mkdir(parents=True,exist_ok=True)
    subprocess.run([sys.executable,str(root/'train'/'train.py'),'--device',a.device,'--train-steps',str(a.train_steps),'--batch-size',str(a.batch_size),'--prediction-type','all','--checkpoint',str(d/'toy_condot.pt')],check=True)
    subprocess.run([sys.executable,str(root/'train'/'train_vp.py'),'--device',a.device,'--train-steps',str(a.train_steps),'--batch-size',str(a.batch_size),'--checkpoint',str(d/'toy_vpsde.pt')],check=True)

if __name__=='__main__': main()
