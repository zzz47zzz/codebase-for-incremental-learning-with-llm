import os 
import importlib

METHOD_LIST = []

cur_path = os.path.dirname(os.path.abspath(__file__))
for method_name in os.listdir(cur_path):
    if method_name in ['__init__.py'] or os.path.isdir(os.path.join(cur_path,method_name)):
        continue
    method_name = method_name.split('.py')[0]
    try:
        importlib.import_module('models.%s'%method_name, package=method_name)
        METHOD_LIST.append(method_name)
    except Exception as e:
        print(e)
        print('Fail to import %s'%(method_name))

print('All supported models = %s'%sorted(METHOD_LIST))
