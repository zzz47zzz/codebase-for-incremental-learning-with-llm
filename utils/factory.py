import importlib

from models import METHOD_LIST
from models.Base import BaseLearner
METHOD_IMPORT_LIST = {}
for _method_name in METHOD_LIST:
    try:
        METHOD_IMPORT_LIST[_method_name] = importlib.import_module('models.%s'%_method_name, package=_method_name)
    except Exception as e:
        print(e)
        print('Fail to import %s'%(_method_name))

def get_model(params, CL_dataset, accelerator) -> BaseLearner:
    method = params.method
    assert method in METHOD_LIST, 'Not implemented for method %s'%(method)
    model = getattr(METHOD_IMPORT_LIST[method],method)(params, CL_dataset, accelerator)

    return model