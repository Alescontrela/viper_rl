import sys
import os
import subprocess

if __name__ == "__main__":
    data_type = sys.argv[1]
    domain = sys.argv[2]
    print(f'Processing {data_type} download for {domain}')
    assert data_type in ['dataset', 'checkpoint'], 'data_type must be either dataset or checkpoint'
    filepath = os.path.join(os.path.dirname(__file__), f'{data_type}_{domain}.sh')
    subprocess.run(["bash", filepath])