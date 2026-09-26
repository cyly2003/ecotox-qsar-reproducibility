"""Pure contract checks; no torch import and no training."""
from __future__ import annotations
import math
from .train_contract import bounds_contract, canonical_hash


def main():
    assert bounds_contract(False,3,None,None)==(3.0,3.0)
    assert bounds_contract(True,None,None,3)==(-math.inf,3.0)
    assert bounds_contract(True,None,3,None)==(3.0,math.inf)
    assert bounds_contract(True,None,2,3)==(2.0,3.0)
    for values in [(True,3,None,3),(True,None,None,None),(True,None,3,2),(True,None,3,3),(False,None,None,None)]:
        try: bounds_contract(*values)
        except ValueError: pass
        else: raise AssertionError(values)
    # p=-log10(c) flips interval endpoints; positive standardization preserves p inequalities.
    lo,hi=-math.log10(100),-math.log10(10)
    assert lo==-2 and hi==-1
    assert (lo-4)/2 < (hi-4)/2
    assert canonical_hash({'b':2,'a':1})==canonical_hash({'a':1,'b':2})
    print('PASS: point, one-sided, interval, malformed bounds, p-scale flip, positive scaling, canonical hashing')


if __name__=='__main__': main()
