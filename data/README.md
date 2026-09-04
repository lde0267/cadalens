# data/

용량이 큰 원본·생성 파일은 저장소에 포함하지 않는다(`.gitignore`).
아래 구조로 직접 배치하면 파이프라인이 동작한다.

```
data/
├── reference/                      ← 저장소 포함
│   ├── jimok_codes.csv             28개 지목 코드·정의·기대 지표
│   └── jimok_taxonomy.json         지목 → 시각 클래스 매핑
│
├── labels/imya_eval_150/           ← labels.csv / manifest.csv 만 저장소 포함
│   ├── labels.csv                  손라벨 150 (pnu, jibun, area_m2, label)
│   └── manifest.csv                표집 목록
│
├── satellite/                      ← 비버전. 국토지리정보원 정사영상
│   ├── (B060)정사영상_2025_37714092.tif
│   ├── (B060)정사영상메타데이터_*.xml
│   └── georef/…_georef.tif         georeference_from_metadata.py 산출
│
├── cadastral/                      ← 비버전. 연속지적도 (LSMD_CONT_LDREG shp 세트)
│   ├── LSMD_CONT_LDREG_경기_안성시/
│   └── derived/anseong_37714092_parcels_within.gpkg   extract_parcels 산출
│
├── chips/                          ← 비버전. clip_parcel_chips.py 산출
│   ├── 37714092/                   mask 모드
│   └── 37714092_imya_real/         real 모드 (현재 표준)
│
├── remoteclip/                     ← 비버전. 분류 결과 CSV·몽타주
└── output/                         ← 비버전. 방법별 임야/형질변경의심 분류 jpg
```

## 좌표계

- 정사영상·지적도 모두 중부원점 (EPSG:5186 / KGD2002 Central Belt 2010) 계열
- 정사영상 원본에 GeoTIFF 태그가 없으면 `georeference_from_metadata.py` 로 복원

## 모델 가중치

`models/remoteclip/` 는 비버전. `classify_*` 첫 실행 시 HuggingFace `chendelong/RemoteCLIP`
(ViT-B-32 / ViT-L-14) 에서 자동 다운로드·캐시.
