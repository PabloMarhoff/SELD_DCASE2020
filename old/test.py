#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun  4 09:51:28 2020

@author: pablo
"""
from multiprocessing import Process, Lock
from time import time

def f(l, i):
    l.acquire()
    try:
        print('hello world', i,'.0')
        print('hello world', i,'.1')
        print('hello world', i,'.2')
    finally:
        l.release()

if __name__ == '__main__':
    t1=time()
    lock = Lock()

    for num in range(10):
        p = Process(target=f, args=(lock, num)).start()
    p.join()
    t2=time()
    print(t2-t1)