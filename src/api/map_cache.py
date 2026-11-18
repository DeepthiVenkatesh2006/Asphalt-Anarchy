"""Bounded display-only cache: map GET reuses a successful inference response."""
from collections import OrderedDict
from threading import RLock

_maps=OrderedDict()
_lock=RLock()


def map_key(service,partition,trip_id,radius,use_viterbi):
    radius=service.settings.search_radius_meters if radius is None else radius
    return (service,partition,trip_id,float(radius),bool(use_viterbi))


def store(key,markup):
    with _lock:
        _maps[key]=markup
        _maps.move_to_end(key)
        while len(_maps)>16:_maps.popitem(last=False)


def lookup(key):
    with _lock:
        markup=_maps.get(key)
        if markup is not None:_maps.move_to_end(key)
        return markup


def invalidate(key):
    with _lock:_maps.pop(key,None)
