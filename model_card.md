# Model Card

## Model Details

This model is a scikit-learn RandomForestClassifier with 150 trees, a minimum of two samples per leaf, and random seed 42. It was developed with Python 3.12 and scikit-learn 1.8.0. Training-only OneHotEncoder and LabelBinarizer artifacts are stored alongside the classifier. The positive label is >50K and the negative label is <=50K. The artifact versions are pinned in requirements.txt.

## Intended Use

The model demonstrates a reproducible educational classification pipeline and FastAPI inference service. It estimates the historical income category of records with the Census Adult dataset schema. It is not validated for lending, employment, insurance, eligibility decisions, or predictions of current income.

## Training Data

The provided census.csv contains 32,561 records of historical Census Adult income data distributed with the Udacity starter repository. A stratified 80/20 split with random_state=42 assigns 26,048 rows to training. The target is salary. Eight categorical columns are one-hot encoded, and six numeric columns are retained. Whitespace is stripped while missing-category markers such as ? are retained as explicit categories. The encoders are fit only on training records. No external data or hyperparameter search was used.

## Evaluation Data

The remaining 6,513 records form a held-out test set. Neither the classifier nor the encoders are fit on these records. Every categorical value observed in this test set is evaluated independently in slice_output.txt. This split evaluates historical within-dataset generalization and does not establish temporal or external validity.

## Metrics

Precision measures the fraction of predicted >50K records whose true label is >50K. Recall measures the fraction of actual >50K records correctly detected. F1 is the harmonic mean of precision and recall. Held-out precision is 0.779953, recall is 0.635204, and F1 is 0.700176. Undefined metrics are reported as zero, so slices without positive examples should not be interpreted as reliable estimates of positive-class performance. Slice counts and positive counts are included to make these limitations visible.

## Ethical Considerations

The features include race and sex, and the historical labels reflect structural socioeconomic disparities. Predictions can perpetuate these disparities. Slice metrics help expose differing outcomes but do not establish fairness. Records are not logged by the API. The local service has no authentication or production security controls and should not be exposed with sensitive personal data.

## Caveats and Recommendations

Small slices have unstable metrics and may lack positive examples. The data describe historical income thresholds rather than current purchasing power. Unknown categories are ignored by the fitted encoder and can reduce reliability. Future work should evaluate calibration, subgroup uncertainty, drift, external validation, and removal of sensitive features before any broader use. Only trusted pickle files may be loaded. Retraining requires preserving compatible dependencies and regenerating all artifacts together.
