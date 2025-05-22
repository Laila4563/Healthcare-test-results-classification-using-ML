from .preprocessing import (
    handle_categorical_missing_values,
    handle_numerical_missing_values,
    display_missing_info,
    look_for_outliers,
    handle_outliers_iqr,
)
from .encoding import encode_features
from .skewness import treat_skewness
from .scaling import scale_features
from .PCA import apply_pca