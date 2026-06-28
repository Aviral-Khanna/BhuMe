# Data

Village bundles are **not committed** to this repo (rasters are 6–16 MB each).

## Download

Files are already present under each village directory after the initial setup.
To re-download from the source:

```bash
BASE=https://hiring.bhume.in/data

# Vadnerbhairav
mkdir -p data/34855_vadnerbhairav_chandavad_nashik
cd data/34855_vadnerbhairav_chandavad_nashik
curl -O $BASE/34855_vadnerbhairav_chandavad_nashik/input.geojson
curl -O $BASE/34855_vadnerbhairav_chandavad_nashik/imagery.tif
curl -O $BASE/34855_vadnerbhairav_chandavad_nashik/boundaries.tif
curl -O $BASE/34855_vadnerbhairav_chandavad_nashik/example_truths.geojson

# Malatavadi
cd ../..
mkdir -p data/12429_malatavadi_chandgad_kolhapur
cd data/12429_malatavadi_chandgad_kolhapur
curl -O $BASE/12429_malatavadi_chandgad_kolhapur/input.geojson
curl -O $BASE/12429_malatavadi_chandgad_kolhapur/imagery.tif
curl -O $BASE/12429_malatavadi_chandgad_kolhapur/boundaries.tif
curl -O $BASE/12429_malatavadi_chandgad_kolhapur/example_truths.geojson
```

## Expected layout after download

```
data/
  34855_vadnerbhairav_chandavad_nashik/
    input.geojson           2.2 MB   2,457 plots
    imagery.tif            13.6 MB   7552×8680 px  ~1.2 m/px
    boundaries.tif         16.4 MB   3776×4340 px  pre-detected field edges
    example_truths.geojson  2.2 KB   6 hand-aligned reference plots
  12429_malatavadi_chandgad_kolhapur/
    input.geojson           1.5 MB   2,508 plots
    imagery.tif             6.9 MB   6400×5224 px  ~0.6 m/px
    boundaries.tif          8.4 MB   3200×2612 px  pre-detected field edges
    example_truths.geojson    997 B   3 hand-aligned reference plots
```

`predictions.geojson` is written here by `predict.py` and **is** committed.
