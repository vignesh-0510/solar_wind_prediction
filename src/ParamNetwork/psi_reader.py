import math
import numpy as np
from scipy.interpolate import RegularGridInterpolator
import gc
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import psi_io

def read_vr(instrument_path, return_scale=False):
    with psi_io.PsiData(f'{instrument_path}/vr002.hdf', model='mas') as mas_reader:
        vr, r_scale, t_scale, p_scale = mas_reader.read(unit=None)
        vr = np.array(vr).transpose(2,1,0)
        r_scale, t_scale, p_scale = np.array(r_scale), np.array(t_scale), np.array(p_scale)
        vr = (vr[:,:-1] + vr[:,1:])/2
        t_scale = (t_scale[:-1] + t_scale[1:] )/ 2
    if return_scale:
        return vr, r_scale, t_scale, p_scale
    return vr

def read_br(instrument_path, return_scale=False):
    with psi_io.PsiData(f'{instrument_path}/br002.hdf', model='mas') as mas_reader:
        br, r_scale, t_scale, p_scale = mas_reader.read(mesh='main', unit=None)
        br = np.array(br).transpose(2,1,0)
        r_scale, t_scale, p_scale = np.array(r_scale), np.array(t_scale), np.array(p_scale)
    if return_scale:
        return br, r_scale, t_scale, p_scale
    return br

def read_vt(instrument_path, return_scale=False):
    with psi_io.PsiData(f'{instrument_path}/vt002.hdf', model='mas') as mas_reader:
        vt, r_scale, t_scale, p_scale = mas_reader.read(unit=None)
        vt = np.array(vt).transpose(2,1,0)
        r_scale, t_scale, p_scale = np.array(r_scale), np.array(t_scale), np.array(p_scale)
        vt = vt[1:]
        r_scale = r_scale[1:]
    if return_scale:
        return vt, r_scale, t_scale, p_scale
    return vt # 30.74 R_Sun

def read_vp(instrument_path, return_scale=False):
    with psi_io.PsiData(f'{instrument_path}/vp002.hdf', model='mas') as mas_reader:
        vp, r_scale, t_scale, p_scale = mas_reader.read(unit=None)
        vp = np.array(vp).transpose(2,1,0)
        r_scale, t_scale, p_scale = np.array(r_scale), np.array(t_scale), np.array(p_scale)
        vp = (vp[:,:-1] + vp[:,1:])/2
        t_scale = (t_scale[1:] + t_scale[:-1])/2
        vp = vp[1:]
        r_scale = r_scale[1:]
    if return_scale:
        return vp, r_scale, t_scale, p_scale
    return vp   # 30.74 R_Sun

def read_bt(instrument_path, return_scale=False):
    with psi_io.PsiData(f'{instrument_path}/bt002.hdf', model='mas') as mas_reader:
        bt, r_scale, t_scale, p_scale = mas_reader.read(mesh='main', unit=None)
        bt = np.array(bt).transpose(2,1,0)
        r_scale, t_scale, p_scale = np.array(r_scale), np.array(t_scale), np.array(p_scale)
    if return_scale:
        return bt, r_scale, t_scale, p_scale
    return bt

def read_bp(instrument_path, return_scale=False):
    with psi_io.PsiData(f'{instrument_path}/bp002.hdf', model='mas') as mas_reader:
        bp, r_scale, t_scale, p_scale = mas_reader.read(unit=None)
        bp = np.array(bp).transpose(2,1,0)
        r_scale, t_scale, p_scale = np.array(r_scale), np.array(t_scale), np.array(p_scale)
    if return_scale:
        return bp, r_scale, t_scale, p_scale
    return bp

def read_jt(instrument_path, return_scale=False):
    with psi_io.PsiData(f'{instrument_path}/jt002.hdf', model='mas') as mas_reader:
        jt, r_scale, t_scale, p_scale = mas_reader.read(unit=None)
        jt = np.array(jt).transpose(2,1,0)
        r_scale, t_scale, p_scale = np.array(r_scale), np.array(t_scale), np.array(p_scale)
        jt = (jt[:-1] + jt[1:])/2
        r_scale = (r_scale[:-1] + r_scale[1:] )/ 2
    if return_scale:
        return jt, r_scale, t_scale, p_scale
    return jt

def read_jp(instrument_path, return_scale=False):
    with psi_io.PsiData(f'{instrument_path}/jp002.hdf', model='mas') as mas_reader:
        jp, r_scale, t_scale, p_scale = mas_reader.read(mesh='main', unit=None)
        jp = np.array(jp).transpose(2,1,0)
        r_scale, t_scale, p_scale = np.array(r_scale), np.array(t_scale), np.array(p_scale)
    if return_scale:
        return jp, r_scale, t_scale, p_scale
    return jp

def read_jr(instrument_path, return_scale=False):
    with psi_io.PsiData(f'{instrument_path}/jr002.hdf', model='mas') as mas_reader:
        jr, r_scale, t_scale, p_scale = mas_reader.read(unit=None)
        jr = np.array(jr).transpose(2,1,0)
        r_scale, t_scale, p_scale = np.array(r_scale), np.array(t_scale), np.array(p_scale)
        jr = (jr[:,:-1] + jr[:,1:])/2
        t_scale = (t_scale[:-1] + t_scale[1:] )/ 2
    if return_scale:
        return jr, r_scale, t_scale, p_scale
    return jr

def read_rho(instrument_path, return_scale=False):
    with psi_io.PsiData(f'{instrument_path}/rho002.hdf', model='mas') as mas_reader:
        rho, r_scale, t_scale, p_scale = mas_reader.read(unit=None)
        rho = np.array(rho).transpose(2,1,0)
        r_scale, t_scale, p_scale = np.array(r_scale), np.array(t_scale), np.array(p_scale)
        rho = (rho[:,:-1] + rho[:,1:])/2
        t_scale = (t_scale[:-1] + t_scale[1:] )/ 2
        rho = (rho[:-1] + rho[1:])/2
        r_scale = (r_scale[:-1] + r_scale[1:] )/ 2
    if return_scale:
        return rho, r_scale, t_scale, p_scale
    return rho

def read_p(instrument_path, return_scale=False):
    with psi_io.PsiData(f'{instrument_path}/p002.hdf', model='mas') as mas_reader:
        p, r_scale, t_scale, p_scale = mas_reader.read(unit=None)
        p = np.array(p).transpose(2,1,0)
        r_scale, t_scale, p_scale = np.array(r_scale), np.array(t_scale), np.array(p_scale)
        p = (p[:,:-1] + p[:,1:])/2
        t_scale = (t_scale[:-1] + t_scale[1:] )/ 2
        p = (p[:-1] + p[1:])/2
        r_scale = (r_scale[:-1] + r_scale[1:] )/ 2
    if return_scale:
        return p, r_scale, t_scale, p_scale
    return p

def read_t(instrument_path, return_scale=False):
    with psi_io.PsiData(f'{instrument_path}/t002.hdf', model='mas') as mas_reader:
        t, r_scale, t_scale, p_scale = mas_reader.read(unit=None)
        t = np.array(t).transpose(2,1,0)
        r_scale, t_scale, p_scale = np.array(r_scale), np.array(t_scale), np.array(p_scale)
        t = (t[:,:-1] + t[:,1:])/2
        t_scale = (t_scale[:-1] + t_scale[1:] )/ 2
        t = (t[:-1] + t[1:])/2
        r_scale = (r_scale[:-1] + r_scale[1:] )/ 2
    if return_scale:
        return t, r_scale, t_scale, p_scale
    return t

COMPONENT_READER_MAP = {
    'vr': read_vr,
    'br': read_br,
    'vt': read_vt,
    'vp': read_vp,
    'bt': read_bt,
    'bp': read_bp,
    'jt': read_jt,
    'jp': read_jp,
    'jr': read_jr,
    'rho': read_rho,
    'p': read_p,
    't': read_t
}

def read_simulation(file_names, instrument_path, return_scale=False):
    components = [k.split('002')[0] for k in file_names]
    results = [0]*len(components)
    for idx, component in enumerate(components):
        results[idx] = COMPONENT_READER_MAP[component](instrument_path, False)
    if return_scale:
        _, r_scale, t_scale, p_scale = COMPONENT_READER_MAP['vr'](instrument_path, True)
        return results, r_scale, t_scale, p_scale
    return results
