from sklearn.pipeline import make_pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
import xgboost as xgb
from xgboost.sklearn import XGBClassifier

pipeline = make_pipeline(
    TfidfVectorizer(
    sublinear_tf=True,
    strip_accents='unicode',
    analyzer='word',
    token_pattern=r'\w{1,}',
    stop_words='english',
    ngram_range=(1, 1),
    norm='l2',
    min_df=0,
    smooth_idf=False,
    max_features=3000),
    XGBClassifier(max_depth=3, n_estimators=300, learning_rate=0.1),
)
