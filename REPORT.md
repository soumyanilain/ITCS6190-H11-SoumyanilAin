# Hands-on L11: Report

**Name:** Soumyanil Ain
**Student ID:** 801488534
**Email:** sain@charlotte.edu

---

## Seed and commands

Seed used (your student ID): **801488534**

The commands I ran, in order, in PowerShell on Windows (Docker Desktop with WSL2), from
`D:\ITCS6190-H11-SoumyanilAin`. I wrote each `spark-submit` on one line because PowerShell
does not accept the `\` line breaks from the README.

```bash
docker compose up -d
docker cp main.py spark-master:/opt/spark/work-dir/

# first try: failed with No module named 'numpy' (see Problems and fixes)
docker exec -it spark-master /opt/spark/bin/spark-submit --master spark://spark-master:7077 /opt/spark/work-dir/main.py task1 /opt/spark/work-dir/shared 801488534

# fix: install numpy in the three containers
docker exec -u root spark-master pip install numpy
docker exec -u root spark-worker-1 pip install numpy
docker exec -u root spark-worker-2 pip install numpy

# second try: the containers restarted half way (see Problems and fixes), so task 2 found no model
docker exec -it spark-master /opt/spark/bin/spark-submit --master spark://spark-master:7077 /opt/spark/work-dir/main.py task1 /opt/spark/work-dir/shared 801488534
docker exec -it spark-master /opt/spark/bin/spark-submit --master spark://spark-master:7077 /opt/spark/work-dir/main.py task2 /opt/spark/work-dir/shared 801488534
docker inspect -f "{{.Name}} OOMKilled={{.State.OOMKilled}} Restarts={{.RestartCount}} ExitCode={{.State.ExitCode}}" spark-master spark-worker-1 spark-worker-2

# third try: both tasks finished
docker exec -it spark-master /opt/spark/bin/spark-submit --master spark://spark-master:7077 /opt/spark/work-dir/main.py task1 /opt/spark/work-dir/shared 801488534
docker exec -it spark-master /opt/spark/bin/spark-submit --master spark://spark-master:7077 /opt/spark/work-dir/main.py task2 /opt/spark/work-dir/shared 801488534

docker compose down
```

Where I deviated from the README: I installed numpy inside the running containers (the
compose files are unchanged), and I cloned the template locally first and then pointed it at
my own repository created from the template, instead of cloning my repository directly.

---

## Results

### Task 1: baseline and full pipeline

```
seed 801488534: 5723 training rows, 1423 test rows
baseline                         RMSE   198.4   R2  0.144
full                             RMSE   189.1   R2  0.223
full pipeline: 7 string + 26 numeric columns, feature vector of 98 doubles
+----------------------+---------------+--------+-----+----------+
|neighbourhood_cleansed|room_type      |bedrooms|price|prediction|
+----------------------+---------------+--------+-----+----------+
|Bayview               |Entire home/apt|1.0     |130.0|-74.1     |
|Bayview               |Private room   |1.0     |70.0 |29.1      |
|Bernal Heights        |Entire home/apt|3.0     |350.0|292.2     |
|Castro/Upper Market   |Entire home/apt|3.0     |500.0|442.1     |
|Castro/Upper Market   |Entire home/apt|2.0     |199.0|222.2     |
+----------------------+---------------+--------+-----+----------+
only showing top 5 rows
full PipelineModel saved to /opt/spark/work-dir/shared/models/full
```

My seed gave **5,723 training rows and 1,423 test rows**, against 5,780 and 1,366 with seed 42.
The data is the same file with the same single partition, so the only thing that changed is
the seed. randomSplit draws a random number for every row from a generator started with the
seed, and puts the row in training when the number falls in the first 80%. A different seed
gives every row a different draw, so a different set of rows lands on each side. It is also
only about 80/20 (5,723 is 80.1% of 7,146), because each row is decided on its own rather than
by cutting the data at an exact count.

### Task 2: scoring with the saved model

```
stages: ['StringIndexerModel', 'OneHotEncoderModel', 'VectorAssembler', 'LinearRegressionModel']
loaded model RMSE 189.1
1422 predictions written for 1423 test rows
+----------------------+---------------+--------+-----+----------+
|neighbourhood_cleansed|room_type      |bedrooms|price|prediction|
+----------------------+---------------+--------+-----+----------+
|Bayview               |Entire home/apt|1.0     |130.0|-74.1     |
|Bayview               |Private room   |1.0     |70.0 |29.1      |
|Bernal Heights        |Entire home/apt|3.0     |350.0|292.2     |
|Castro/Upper Market   |Entire home/apt|3.0     |500.0|442.1     |
|Castro/Upper Market   |Entire home/apt|2.0     |199.0|222.2     |
+----------------------+---------------+--------+-----+----------+
only showing top 5 rows
```

The loaded model gives the same RMSE (189.1) and the same predictions as the model right after
training in task 1, which shows nothing was lost in the save and load.

---

## Questions

**1. Baseline versus full.**

I got **198.4 / 0.144 for the baseline and 189.1 / 0.223 for the full pipeline**. The 97 extra
features cut the error by about 9 dollars a night (around 5%) and raised R2 from 0.144 to 0.223,
so the model now explains about 22% of the variation in price instead of 14%. That is more
than the lecture got with seed 42 (221.6 to 220.6, only 1 dollar), but it is still a weak
model: always guessing the average price would give an RMSE of about 214 on my test set, so
the full model beats a constant guess by only about 25 dollars.

The price range is the main reason. RMSE squares every error before averaging, so a few
listings that are far off dominate it. In my test set the 5 worst predictions out of 1,422
make up 59% of the total squared error: three listings at 2,281, 2,282 and 4,500 dollars a
night that the model priced much lower, and two cheap listings (69 and 164 dollars) that it
priced above 1,600. The model also predicts negative prices for 33 test listings (the Bayview
home above at -74.1), because a straight line on raw dollars has no floor.

The numbers change from seed to seed because the test set is a different random 20% each time,
and with a few extreme listings it matters a lot which side they land on. With my seed the
10,000-dollar listing is in the training set and the most expensive test listing is 4,500
(14 training listings are above 2,000 against 3 in the test set). That is a big part of why my
baseline RMSE (198.4) is lower than the lecture's (221.6), even though it is the same model.
The training rows change too, so the coefficients change.

The change I would try first is to predict the logarithm of the price and convert back with
exp() before evaluating, as the book does (slide "Running Example" with the model table). It
goes straight at the outlier problem, since a 10,000-dollar listing no longer dwarfs a
100-dollar one, and the predictions can no longer be negative. After that I would try a random
forest tuned with ParamGridBuilder and CrossValidator (slides "Tuning"), which got the
lecture's RMSE down to 213.9.

**2. What is in `shared-folder/models/full`?**

```
models/full
models/full/metadata
models/full/stages
models/full/stages/0_StringIndexer_49bf6948ab93
models/full/stages/1_OneHotEncoder_cde7872c571e
models/full/stages/2_VectorAssembler_362adec61275
models/full/stages/3_LinearRegression_2f809273aea0
```

- `metadata/`: one JSON line describing the whole pipeline: the class
  (`org.apache.spark.ml.PipelineModel`), the Spark version (4.2.0), its uid, and the uids of
  the four stages in order.
- `stages/0_StringIndexer_.../`: `metadata/` with the class (`StringIndexerModel`), the 7 input
  and output columns and `handleInvalid: skip`, plus `data/`, a Parquet file with the labels
  learned for each string column, the category-to-index mapping from training.
- `stages/1_OneHotEncoder_.../`: `metadata/` (`OneHotEncoderModel`, its columns, `dropLast: true`)
  plus `data/` with the number of categories learned for each column, which fixes the size of
  each one-hot vector.
- `stages/2_VectorAssembler_.../`: only `metadata/`, the list of 33 input columns and the output
  column `features`. There is no `data/` folder because it learned nothing.
- `stages/3_LinearRegression_.../`: `metadata/` (`LinearRegressionModel` and its parameters,
  regParam 0, solver auto) plus `data/` with the intercept and the 98 coefficients.

We save the PipelineModel rather than the LinearRegressionModel alone because the regression
model only understands a ready-made vector of 98 numbers in a fixed order. It cannot read
"Private room" or "Bayview". The mapping from strings to indexes, the vector sizes and the
column order live in the other three stages. With the whole pipeline saved, the scoring job
loads one object with `PipelineModel.load` and feeds it raw listings, and they go through
exactly the same steps as in training. Saved stage by stage, I would have to load each one
with its own class and put them back together in the right order.

The fitted models are stage 0 (`StringIndexerModel`), stage 1 (`OneHotEncoderModel`) and stage
3 (`LinearRegressionModel`); they are the ones with a `data/` folder. Stage 2
(`VectorAssembler`) is a plain transformer. The folder names keep the estimator's uid
(`StringIndexer_...`), but the class saved inside is the fitted model.

**3. Only transformers at scoring time.**

Task 2 wrote **1,422 predictions for 1,423 test rows**, so one row is missing. Comparing the
test set with the training set, the missing listing is a private room in **Presidio** (1
bedroom, 105 dollars). Presidio does not appear in any of the 5,723 training listings, so the
StringIndexerModel has no index for it, and `handleInvalid="skip"` on the StringIndexer (stage
0) dropped the row. With `"keep"` it would have been scored with an extra "unknown" category,
and with the default `"error"` task 2 would have crashed.

To score a stream instead of a Parquet file, the model and the `transform` line stay exactly the
same; only the reading and writing change (slide "Scoring a Stream"):

```python
schema = spark.read.parquet(f"{data_dir}/split/test").schema   # a file stream needs the schema up front
streamDF = (spark.readStream.schema(schema)
            .option("maxFilesPerTrigger", 1)
            .parquet(f"{shared}/listings_in"))                 # folder where new listings arrive
predDF = model.transform(streamDF)                              # same as in batch, still no fit()
query = (predDF.select("neighbourhood_cleansed", "room_type", "bedrooms", "price",
                       sround("prediction", 1).alias("prediction"))
         .writeStream.format("csv").option("header", True)
         .option("path", f"{out_dir}/stream_predictions")
         .option("checkpointLocation", f"{shared}/ckpt/score")
         .trigger(availableNow=True)
         .start())
query.awaitTermination()
```

The parts of task 2 that need the whole dataset at once have to go: `coalesce(1).write`,
`count()`, `show()` and `rmse_eval.evaluate()` do not work on a streaming DataFrame. If I still
wanted the RMSE, I would compute it per micro-batch inside `foreachBatch`. This only works
because the PipelineModel is all transformers; `fit()` on a stream is refused.

---

## Problems and fixes

**1. numpy missing in the Spark image.** The first run of task 1 stopped before doing anything:

```
Traceback (most recent call last):
  File "/opt/spark/work-dir/main.py", line 24, in <module>
    from pyspark.ml import Pipeline, PipelineModel
  ...
ModuleNotFoundError: No module named 'numpy'
```

The `apache/spark:4.2.0` image has Python but not numpy, and `pyspark.ml` imports numpy.
Fix: `docker exec -u root <container> pip install numpy` on spark-master, spark-worker-1 and
spark-worker-2 (numpy 2.2.6). This is lost on `docker compose down`, since that recreates the
containers, so it has to be repeated after the next `up`.

**2. Containers restarted in the middle of task 1.** The second run of task 1 got through the
split and then the command ended with:

```
request returned 500 Internal Server Error for API route and version http://%2F%2F.%2Fpipe%2FdockerDesktopLinuxEngine/v1.56/exec/6d26a5283c25cf63708f9e93e4b0f75b5392fc41a782b8b76b0a6c1832015493/json, check if the server supports the requested API version
```

The cluster came back with new IP addresses (the master moved from 172.18.0.2 to 172.18.0.4
and both workers registered again), so Docker Desktop had dropped and restarted the
containers. Task 1 never reached the save, so task 2 failed with:

```
pyspark.errors.exceptions.captured.AnalysisException: [PATH_NOT_FOUND] Path does not exist: file:/opt/spark/work-dir/shared/models/full/metadata. SQLSTATE: 42K03
```

`docker inspect` showed `OOMKilled=false` for all three containers, so it was not a memory
problem. Fix: I ran task 1 again, it finished and saved the model, and then task 2 worked.

**3. Warnings that are not errors.**
- `netlib-blas: JNI_OnLoad: dlopen(libblas.so.3) failed`: the image has no native BLAS
  library, so Spark uses its built-in Java implementation. Same results, possibly slower.
- `regParam is zero, which might cause numerical instability and overfitting`: we fit plain
  least squares with no regularization, as the task asks.
- `Cholesky solver failed due to singular covariance matrix. Retrying with Quasi-Newton
  solver` (full pipeline only): some of the 98 feature columns are exact combinations of
  others, so the exact solver cannot invert the matrix and Spark switched to an iterative
  solver. The model still trained and was evaluated normally.
