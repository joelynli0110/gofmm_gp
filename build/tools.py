# This file was automatically generated by SWIG (http://www.swig.org).
# Version 3.0.8
#
# Do not make changes to this file unless you know what you are doing--modify
# the SWIG interface file instead.





from sys import version_info
if version_info >= (2, 6, 0):
    def swig_import_helper():
        from os.path import dirname
        import imp
        fp = None
        try:
            fp, pathname, description = imp.find_module('_tools', [dirname(__file__)])
        except ImportError:
            import _tools
            return _tools
        if fp is not None:
            try:
                _mod = imp.load_module('_tools', fp, pathname, description)
            finally:
                fp.close()
            return _mod
    _tools = swig_import_helper()
    del swig_import_helper
else:
    import _tools
del version_info
try:
    _swig_property = property
except NameError:
    pass  # Python < 2.2 doesn't have 'property'.


def _swig_setattr_nondynamic(self, class_type, name, value, static=1):
    if (name == "thisown"):
        return self.this.own(value)
    if (name == "this"):
        if type(value).__name__ == 'SwigPyObject':
            self.__dict__[name] = value
            return
    method = class_type.__swig_setmethods__.get(name, None)
    if method:
        return method(self, value)
    if (not static):
        if _newclass:
            object.__setattr__(self, name, value)
        else:
            self.__dict__[name] = value
    else:
        raise AttributeError("You cannot add attributes to %s" % self)


def _swig_setattr(self, class_type, name, value):
    return _swig_setattr_nondynamic(self, class_type, name, value, 0)


def _swig_getattr_nondynamic(self, class_type, name, static=1):
    if (name == "thisown"):
        return self.this.own()
    method = class_type.__swig_getmethods__.get(name, None)
    if method:
        return method(self)
    if (not static):
        return object.__getattr__(self, name)
    else:
        raise AttributeError(name)

def _swig_getattr(self, class_type, name):
    return _swig_getattr_nondynamic(self, class_type, name, 0)


def _swig_repr(self):
    try:
        strthis = "proxy of " + self.this.__repr__()
    except Exception:
        strthis = ""
    return "<%s.%s; %s >" % (self.__class__.__module__, self.__class__.__name__, strthis,)

try:
    _object = object
    _newclass = 1
except AttributeError:
    class _object:
        pass
    _newclass = 0



_tools.N_CHILDREN_swigconstant(_tools)
N_CHILDREN = _tools.N_CHILDREN
class file_to_argv(_object):
    __swig_setmethods__ = {}
    __setattr__ = lambda self, name, value: _swig_setattr(self, file_to_argv, name, value)
    __swig_getmethods__ = {}
    __getattr__ = lambda self, name: _swig_getattr(self, file_to_argv, name)
    __repr__ = _swig_repr

    def __init__(self, *args):
        this = _tools.new_file_to_argv(*args)
        try:
            self.this.append(this)
        except Exception:
            self.this = this
    __swig_destroy__ = _tools.delete_file_to_argv
    __del__ = lambda self: None

    def print_argv(self):
        return _tools.file_to_argv_print_argv(self)

    def return_argv(self):
        return _tools.file_to_argv_return_argv(self)
file_to_argv_swigregister = _tools.file_to_argv_swigregister
file_to_argv_swigregister(file_to_argv)


def call_Launchhelper(filename):
    return _tools.call_Launchhelper(filename)
call_Launchhelper = _tools.call_Launchhelper

def launchhelper_denseSPD(K, filename):
    return _tools.launchhelper_denseSPD(K, filename)
launchhelper_denseSPD = _tools.launchhelper_denseSPD

def load_denseSPD(height, width, filename):
    return _tools.load_denseSPD(height, width, filename)
load_denseSPD = _tools.load_denseSPD

def hello_world():
    return _tools.hello_world()
hello_world = _tools.hello_world

def Compress(K, NN, splitter, rkds, config):
    return _tools.Compress(K, NN, splitter, rkds, config)
Compress = _tools.Compress

def Evaluate(tree, weights):
    return _tools.Evaluate(tree, weights)
Evaluate = _tools.Evaluate

def load_denseSPD_from_console(numpyArr):
    return _tools.load_denseSPD_from_console(numpyArr)
load_denseSPD_from_console = _tools.load_denseSPD_from_console

def mul_denseSPD(K1, K2, mul_numpy):
    return _tools.mul_denseSPD(K1, K2, mul_numpy)
mul_denseSPD = _tools.mul_denseSPD
# This file is compatible with both classic and new-style classes.


