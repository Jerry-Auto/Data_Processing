import os
a = 0
folder_path=os.path.dirname(os.path.abspath(__file__))
py_path=os.path.join(folder_path,'a.py')
a = os.system(f"python {py_path}")
print(a)