
# Import the libraries
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from google.cloud import bigquery, bigquery_storage
from sklearn.ensemble import RandomForestRegressor
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from google import auth
from scipy import stats
import numpy as np
import argparse
import joblib
import pickle
import csv
import os

# add parser arguments
parser = argparse.ArgumentParser()
parser.add_argument('--project-id', dest='project_id',  type=str, help='Project ID.')
parser.add_argument('--training-dir', dest='training_dir', default=os.getenv("AIP_MODEL_DIR"),
                    type=str, help='Dir to save the data and the trained model.')
parser.add_argument('--bq-source', dest='bq_source',  type=str, help='BigQuery data source for training data.')
args = parser.parse_args()

# data preparation code
BQ_QUERY = """
with tmp_table as (
SELECT trip_seconds, trip_miles, fare,
    tolls,  company,
    pickup_latitude, pickup_longitude, dropoff_latitude, dropoff_longitude,
    DATETIME(trip_start_timestamp, 'America/Chicago') trip_start_timestamp,
    DATETIME(trip_end_timestamp, 'America/Chicago') trip_end_timestamp,
    CASE WHEN (pickup_community_area IN (56, 64, 76)) OR (dropoff_community_area IN (56, 64, 76)) THEN 1 else 0 END is_airport,
FROM `{}`
WHERE
  dropoff_latitude IS NOT NULL and
  dropoff_longitude IS NOT NULL and
  pickup_latitude IS NOT NULL and
  pickup_longitude IS NOT NULL and
  fare > 0 and
  trip_miles > 0
  and MOD(ABS(FARM_FINGERPRINT(unique_key)), 100) between 0 and 99
ORDER BY RAND()
LIMIT 10000)
SELECT *,
    EXTRACT(YEAR FROM trip_start_timestamp) trip_start_year,
    EXTRACT(MONTH FROM trip_start_timestamp) trip_start_month,
    EXTRACT(DAY FROM trip_start_timestamp) trip_start_day,
    EXTRACT(HOUR FROM trip_start_timestamp) trip_start_hour,
    FORMAT_DATE('%a', DATE(trip_start_timestamp)) trip_start_day_of_week
FROM tmp_table
""".format(args.bq_source)
# Get default credentials
credentials, project = auth.default()
bqclient = bigquery.Client(credentials=credentials, project=args.project_id)
bqstorageclient = bigquery_storage.BigQueryReadClient(credentials=credentials)
df = (
    bqclient.query(BQ_QUERY)
    .result()
    .to_dataframe(bqstorage_client=bqstorageclient)
)
# Add 'N/A' for missing 'Company'
df.fillna(value={'company':'N/A','tolls':0}, inplace=True)
# Drop rows containing null data.
df.dropna(how='any', axis='rows', inplace=True)
# Pickup and dropoff locations distance
df['abs_distance'] = (np.hypot(df['dropoff_latitude']-df['pickup_latitude'], df['dropoff_longitude']-df['pickup_longitude']))*100

# Remove extremes, outliers
possible_outliers_cols = ['trip_seconds', 'trip_miles', 'fare', 'abs_distance']
df=df[(np.abs(stats.zscore(df[possible_outliers_cols].astype(float))) < 3).all(axis=1)].copy()
# Reduce location accuracy
df=df.round({'pickup_latitude': 3, 'pickup_longitude': 3, 'dropoff_latitude':3, 'dropoff_longitude':3})

# Drop the timestamp col
X=df.drop(['trip_start_timestamp', 'trip_end_timestamp'],axis=1)

# Split the data into train and test
X_train, X_test = train_test_split(X, test_size=0.10, random_state=123)

## Format the data for batch predictions
# select string cols
string_cols = X_test.select_dtypes(include='object').columns
# Add quotes around string fields
X_test[string_cols] = X_test[string_cols].apply(lambda x: '\"' + x + '\"')
# Add quotes around column names
X_test.columns = ['\"' + col + '\"' for col in X_test.columns]
# Save DataFrame to csv
X_test.to_csv(os.path.join(args.training_dir,"test.csv"),index=False,quoting=csv.QUOTE_NONE, escapechar=' ')
# Save test data without the target for batch predictions
X_test.drop('\"fare\"',axis=1,inplace=True)
X_test.to_csv(os.path.join(args.training_dir,"test_no_target.csv"),index=False,quoting=csv.QUOTE_NONE, escapechar=' ')

# Separate the target column
y_train=X_train.pop('fare')
# Get the column indexes
col_index_dict = {col: idx for idx, col in enumerate(X_train.columns)}
# Create a column transformer pipeline
ct_pipe = ColumnTransformer(transformers=[
    ('hourly_cat', OneHotEncoder(categories=[range(0,24)], sparse = False), [col_index_dict['trip_start_hour']]),
    ('dow', OneHotEncoder(categories=[['Mon', 'Tue', 'Sun', 'Wed', 'Sat', 'Fri', 'Thu']], sparse = False), [col_index_dict['trip_start_day_of_week']]),
    ('std_scaler', StandardScaler(), [
        col_index_dict['trip_start_year'],
        col_index_dict['abs_distance'],
        col_index_dict['pickup_longitude'],
        col_index_dict['pickup_latitude'],
        col_index_dict['dropoff_longitude'],
        col_index_dict['dropoff_latitude'],
        col_index_dict['trip_miles'],
        col_index_dict['trip_seconds']])
])
# Add the random-forest estimator to the pipeline
rfr_pipe = Pipeline([
    ('ct', ct_pipe),
    ('forest_reg', RandomForestRegressor(
        n_estimators = 20,
        max_features = 1.0,
        n_jobs = -1,
        random_state = 3,
        max_depth=None,
        max_leaf_nodes=None,
    ))
])

# train the model
rfr_score = cross_val_score(rfr_pipe, X_train, y_train, scoring = 'neg_mean_squared_error', cv = 5)
rfr_rmse = np.sqrt(-rfr_score)
print ("Crossvalidation RMSE:",rfr_rmse.mean())
final_model=rfr_pipe.fit(X_train, y_train)
# Save the model pipeline
with open(os.path.join(args.training_dir,"model.pkl"), 'wb') as model_file:
    pickle.dump(final_model, model_file)
