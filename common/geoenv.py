#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PROJ/GDAL 데이터 경로 격리.

이 시스템에는 PostgreSQL/PostGIS 의 구버전 proj.db 를 가리키는 PROJ_LIB/GDAL_DATA
환경변수가 걸려 있어 rasterio 내장 PROJ 9.x 와 충돌한다(LAYOUT.VERSION 오류).
**rasterio / geopandas 를 import 하기 전에** 이 모듈을 먼저 import 하면
내장 데이터 폴더로 환경변수를 강제 고정한다.

    from common import geoenv  # noqa: F401  (import 부작용용)
    import rasterio
"""
from __future__ import annotations

import os
import sysconfig

_SITE = sysconfig.get_paths()["purelib"]
for _var, _sub in (("PROJ_LIB", "proj_data"), ("PROJ_DATA", "proj_data"), ("GDAL_DATA", "gdal_data")):
    _path = os.path.join(_SITE, "rasterio", _sub)
    if os.path.isdir(_path):
        os.environ[_var] = _path
