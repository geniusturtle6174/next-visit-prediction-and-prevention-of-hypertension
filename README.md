# next-visit-prediction-and-prevention-of-hypertension

Source code of a manuscript entitled "Next-visit prediction and prevention of hypertension using large-scale routine health checkup data."

## Essential Usage

### Required libs

* For training and evaluation: `numpy`, `scipy`, `scikit-learn`, `xgboost`, and `lightgbm`.
* For data or results observation: `matplotlib`.
* Other utilities: `pickle5`, `PyYAML`.
* For old deep learning implementation: `pytorch`. Comment out the "import"s if you are not going to run them.
* Versions of them are not restricted as long as they're new enough.

### Data preprocessing

Follow the instructions below in order.

```bash
python3 data_to_pkl.py
```
* Data files should be placed and named correctly. See code for details.
* May need to be run on a server with larger amount of RAM.

```bash
python3 pkl_to_fea.py save_fea_dir
```
* `save_fea_dir`: Directory for saving produced feature files.
* Previously produced pkl file is automatically loaded.
* May need to be run on a server with larger amount of RAM.

### Training

* Imputation of Missing values for training data is done is this stage. See `patch_value.patch_for_one_person` for details.
* From this stage on, you could run the code on a laptop with enough RAM.

#### XGBoost or LightGBM models

```bash
python3 train_cv_nvp_xgb.py fea_dir save_model_dir_name -classify
```
* `fea_dir`: Directory of previously produced feature files.
* `save_model_dir_name`: Directory for saving traning parameters, trained models, and intermediate results.
* Add `--model_type lgbm` to train LightGBM models.
* Add `-d dim_1 [dim_2 dim3 ...]` to train using certain factors, where `dim_*` indicates zero-based feature index. See `fea_names` for indices of factor names.

#### Random forest models

```bash
python3 train_cv_nvp_rf.py fea_dir save_model_dir_name -classify
```
* `fea_dir`: Directory of previously produced feature files.
* `save_model_dir_name`: Directory for saving traning parameters, trained models, and intermediate results.
* Add `-d dim_1 [dim_2 dim3 ...]` to train using certain factors, where `dim_*` indicates zero-based feature index. See `fea_names` for indices of factor names.

#### Obtaining feature importance

```bash
python3 get_fea_importance.py model_dir -agg
```
* `model_dir`: Directory of previously produced model files.
* Results are written to stdout.

### Evaluation

* Imputation of Missing values for test data is done is this stage.
* Use the same python files for all (XGBoost, LightGBM, and random forest) model types.
* Follow the instructions below in order.

```bash
python3 test_cv_raw_nvp_xgb.py model_dir output_result_dir
```
* `model_dir`: Directory of previously produced model files.
* `output_result_dir`: Directory for saving raw prediction results.
* Add `-mf n` to add `n` virtual visits, where `n` is in [1, 5].
  * Use `python3 plot_feature_similarity_x.py fea_dir` to obtain parameters for adding virtual visits.
  * Obtained parameters should be filled in `fea_util.modify_fea`.

```bash
python3 test_cv_merge_classifier_hybp_xgb.py result_dir
```
* `result_dir`: Directory of previously produced raw prediction results.
* Have to copy and paste the evaluation results to `plot_figures` if needed.
* Statistical test for one `result_dir` is performed here. Use `python3 compare_two_model_results.py result_dir_1 result_dir_2` for comparing two `result_dir`s.
