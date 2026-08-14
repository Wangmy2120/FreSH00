# FreSH

Frequency-Segmented Hierarchical Multi-Expert Framework for Multivariate Time Series Classification.

FreSH transforms multivariate time series into the frequency domain, divides the spectrum into segments, and processes those segments with a hierarchical mixture-of-experts architecture. Segment experts capture band-specific patterns, global experts model full-spectrum dependencies, and adaptive gating fuses the two representations before classification.

## Installation

```bash
conda env create -f environment.yml
conda activate fresh
```

or

```bash
pip install -r requirements.txt
```

## Data

Place UEA multivariate time series `.ts` files under `dataset/<DatasetName>/`. The loader expects the standard UEA `*_TRAIN.ts` and `*_TEST.ts` naming.

## Training and evaluation

```bash
python -u runFreSH.py \
  --task_name classification \
  --is_training 1 \
  --root_path ./dataset/UWaveGestureLibrary/ \
  --model_id UWaveGestureLibrary \
  --model FreSH \
  --data UEA \
  --batch_size 16 \
  --Seg_num 4 \
  --SegE_num 2 \
  --GE_num 2 \
  --learning_rate 0.002 \
  --train_epochs 100 \
  --patience 30
```

Dataset-specific launch scripts are available under `scripts/UEA/`.

## Acknowledgements

This project uses the UEA Multivariate Time Series Classification Archive. We thank Bagnall, Dau, Lines, Flynn, Large, Bostrom, Southam, Keogh, and all researchers involved in collecting, curating, and cleaning this dataset. The archive is publicly available at [http://www.timeseriesclassification.com](http://www.timeseriesclassification.com).If you use this repository or build upon this work, please also cite the original dataset paper:

```bibtex
@article{bagnall2018uea,
  title={The UEA multivariate time series classification archive, 2018},
  author={Bagnall, Anthony and Dau, Hoang Anh and Lines, Jason and Flynn, Michael and Large, James and Bostrom, Aaron and Southam, Paul and Keogh, Eamonn},
  journal={arXiv preprint arXiv:1811.00075},
  year={2018}
}

## License

See [LICENSE](LICENSE). Some bundled third-party files retain their own license headers; check the header of each source file.
