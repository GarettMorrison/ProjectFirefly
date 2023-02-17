# This file was automatically generated by SWIG (http://www.swig.org).
# Version 4.0.1
#
# Do not make changes to this file unless you know what you are doing--modify
# the SWIG interface file instead.

from sys import version_info as _swig_python_version_info
if _swig_python_version_info < (2, 7, 0):
    raise RuntimeError("Python 2.7 or later required")

# Import the low-level C/C++ module
if __package__ or "." in __name__:
    from . import _FastFireFly
else:
    import _FastFireFly

try:
    import builtins as __builtin__
except ImportError:
    import __builtin__

def _swig_repr(self):
    try:
        strthis = "proxy of " + self.this.__repr__()
    except __builtin__.Exception:
        strthis = ""
    return "<%s.%s; %s >" % (self.__class__.__module__, self.__class__.__name__, strthis,)


def _swig_setattr_nondynamic_instance_variable(set):
    def set_instance_attr(self, name, value):
        if name == "thisown":
            self.this.own(value)
        elif name == "this":
            set(self, name, value)
        elif hasattr(self, name) and isinstance(getattr(type(self), name), property):
            set(self, name, value)
        else:
            raise AttributeError("You cannot add instance attributes to %s" % self)
    return set_instance_attr


def _swig_setattr_nondynamic_class_variable(set):
    def set_class_attr(cls, name, value):
        if hasattr(cls, name) and not isinstance(getattr(cls, name), property):
            set(cls, name, value)
        else:
            raise AttributeError("You cannot add class attributes to %s" % cls)
    return set_class_attr


def _swig_add_metaclass(metaclass):
    """Class decorator for adding a metaclass to a SWIG wrapped class - a slimmed down version of six.add_metaclass"""
    def wrapper(cls):
        return metaclass(cls.__name__, cls.__bases__, cls.__dict__.copy())
    return wrapper


class _SwigNonDynamicMeta(type):
    """Meta class to enforce nondynamic attributes (no new attributes) for a class"""
    __setattr__ = _swig_setattr_nondynamic_class_variable(type.__setattr__)


class SwigPyIterator(object):
    thisown = property(lambda x: x.this.own(), lambda x, v: x.this.own(v), doc="The membership flag")

    def __init__(self, *args, **kwargs):
        raise AttributeError("No constructor defined - class is abstract")
    __repr__ = _swig_repr
    __swig_destroy__ = _FastFireFly.delete_SwigPyIterator

    def value(self):
        return _FastFireFly.SwigPyIterator_value(self)

    def incr(self, n=1):
        return _FastFireFly.SwigPyIterator_incr(self, n)

    def decr(self, n=1):
        return _FastFireFly.SwigPyIterator_decr(self, n)

    def distance(self, x):
        return _FastFireFly.SwigPyIterator_distance(self, x)

    def equal(self, x):
        return _FastFireFly.SwigPyIterator_equal(self, x)

    def copy(self):
        return _FastFireFly.SwigPyIterator_copy(self)

    def next(self):
        return _FastFireFly.SwigPyIterator_next(self)

    def __next__(self):
        return _FastFireFly.SwigPyIterator___next__(self)

    def previous(self):
        return _FastFireFly.SwigPyIterator_previous(self)

    def advance(self, n):
        return _FastFireFly.SwigPyIterator_advance(self, n)

    def __eq__(self, x):
        return _FastFireFly.SwigPyIterator___eq__(self, x)

    def __ne__(self, x):
        return _FastFireFly.SwigPyIterator___ne__(self, x)

    def __iadd__(self, n):
        return _FastFireFly.SwigPyIterator___iadd__(self, n)

    def __isub__(self, n):
        return _FastFireFly.SwigPyIterator___isub__(self, n)

    def __add__(self, n):
        return _FastFireFly.SwigPyIterator___add__(self, n)

    def __sub__(self, *args):
        return _FastFireFly.SwigPyIterator___sub__(self, *args)
    def __iter__(self):
        return self

# Register SwigPyIterator in _FastFireFly:
_FastFireFly.SwigPyIterator_swigregister(SwigPyIterator)

class vectord(object):
    thisown = property(lambda x: x.this.own(), lambda x, v: x.this.own(v), doc="The membership flag")
    __repr__ = _swig_repr

    def iterator(self):
        return _FastFireFly.vectord_iterator(self)
    def __iter__(self):
        return self.iterator()

    def __nonzero__(self):
        return _FastFireFly.vectord___nonzero__(self)

    def __bool__(self):
        return _FastFireFly.vectord___bool__(self)

    def __len__(self):
        return _FastFireFly.vectord___len__(self)

    def __getslice__(self, i, j):
        return _FastFireFly.vectord___getslice__(self, i, j)

    def __setslice__(self, *args):
        return _FastFireFly.vectord___setslice__(self, *args)

    def __delslice__(self, i, j):
        return _FastFireFly.vectord___delslice__(self, i, j)

    def __delitem__(self, *args):
        return _FastFireFly.vectord___delitem__(self, *args)

    def __getitem__(self, *args):
        return _FastFireFly.vectord___getitem__(self, *args)

    def __setitem__(self, *args):
        return _FastFireFly.vectord___setitem__(self, *args)

    def pop(self):
        return _FastFireFly.vectord_pop(self)

    def append(self, x):
        return _FastFireFly.vectord_append(self, x)

    def empty(self):
        return _FastFireFly.vectord_empty(self)

    def size(self):
        return _FastFireFly.vectord_size(self)

    def swap(self, v):
        return _FastFireFly.vectord_swap(self, v)

    def begin(self):
        return _FastFireFly.vectord_begin(self)

    def end(self):
        return _FastFireFly.vectord_end(self)

    def rbegin(self):
        return _FastFireFly.vectord_rbegin(self)

    def rend(self):
        return _FastFireFly.vectord_rend(self)

    def clear(self):
        return _FastFireFly.vectord_clear(self)

    def get_allocator(self):
        return _FastFireFly.vectord_get_allocator(self)

    def pop_back(self):
        return _FastFireFly.vectord_pop_back(self)

    def erase(self, *args):
        return _FastFireFly.vectord_erase(self, *args)

    def __init__(self, *args):
        _FastFireFly.vectord_swiginit(self, _FastFireFly.new_vectord(*args))

    def push_back(self, x):
        return _FastFireFly.vectord_push_back(self, x)

    def front(self):
        return _FastFireFly.vectord_front(self)

    def back(self):
        return _FastFireFly.vectord_back(self)

    def assign(self, n, x):
        return _FastFireFly.vectord_assign(self, n, x)

    def resize(self, *args):
        return _FastFireFly.vectord_resize(self, *args)

    def insert(self, *args):
        return _FastFireFly.vectord_insert(self, *args)

    def reserve(self, n):
        return _FastFireFly.vectord_reserve(self, n)

    def capacity(self):
        return _FastFireFly.vectord_capacity(self)
    __swig_destroy__ = _FastFireFly.delete_vectord

# Register vectord in _FastFireFly:
_FastFireFly.vectord_swigregister(vectord)

class ledLocalizationFast(object):
    thisown = property(lambda x: x.this.own(), lambda x, v: x.this.own(v), doc="The membership flag")
    __repr__ = _swig_repr

    def __init__(self, _LED_X, _LED_Y, _LED_Z, _pos_x=0, _pos_y=-500, _pos_z=1000):
        _FastFireFly.ledLocalizationFast_swiginit(self, _FastFireFly.new_ledLocalizationFast(_LED_X, _LED_Y, _LED_Z, _pos_x, _pos_y, _pos_z))

    def getPosition(self):
        return _FastFireFly.ledLocalizationFast_getPosition(self)

    def fitPositionToVectors(self, vect_X, vect_Y, vect_Z, _Vect_S, _LED_Indices):
        return _FastFireFly.ledLocalizationFast_fitPositionToVectors(self, vect_X, vect_Y, vect_Z, _Vect_S, _LED_Indices)

    def getError(self):
        return _FastFireFly.ledLocalizationFast_getError(self)

    def getRandFactor(self):
        return _FastFireFly.ledLocalizationFast_getRandFactor(self)
    __swig_destroy__ = _FastFireFly.delete_ledLocalizationFast

# Register ledLocalizationFast in _FastFireFly:
_FastFireFly.ledLocalizationFast_swigregister(ledLocalizationFast)



