import sys
import yaml
from yaml import SafeLoader as yaml_Loader
from lib.training.execute import execute

if __name__ == '__main__':
    additional_configs = None
    if len(sys.argv) > 2:
        additional_configs = yaml.load('\n'.join(sys.argv[2:]), 
                                       Loader=yaml_Loader)
    execute('predict', sys.argv[1], additional_configs)
        
