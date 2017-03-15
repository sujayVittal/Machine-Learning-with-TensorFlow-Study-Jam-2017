# Wide and Deep Learning using TensorFlow

### A. Setup

1. Download *wide-n-deep.py* [link: https://github.com/sujayVittal/Machine-Learning-with-TensorFlow-Study-Jam-2017/blob/master/wide-n-deep.py]

2. Install the pandas data analysis library. tf.learn doesn't require pandas, but it does support it, and this tutorial uses pandas. To install pandas:
  - Use *pip* to install pandas:
    
    shell $ sudo pip install pandas
    
3. Execute the tutorial code to train the linear model:
  
    shell $ python wide-n-deep.py --model_type=wide_n_deep
  
  
### B. Define Base Feature Columns

First, let's define the base categorical and continuous feature columns that we'll use. These base columns will be the building blocks used by both the wide part and the deep part of the model.

    import tensorflow as tf

    # Categorical base columns.
    gender = tf.contrib.layers.sparse_column_with_keys(column_name="gender", keys=["Female", "Male"])
    race = tf.contrib.layers.sparse_column_with_keys(column_name="race", keys=[
      "Amer-Indian-Eskimo", "Asian-Pac-Islander", "Black", "Other", "White"])
    education = tf.contrib.layers.sparse_column_with_hash_bucket("education", hash_bucket_size=1000)
    relationship = tf.contrib.layers.sparse_column_with_hash_bucket("relationship", hash_bucket_size=100)
    workclass = tf.contrib.layers.sparse_column_with_hash_bucket("workclass", hash_bucket_size=100)
    occupation = tf.contrib.layers.sparse_column_with_hash_bucket("occupation", hash_bucket_size=1000)
    native_country = tf.contrib.layers.sparse_column_with_hash_bucket("native_country", hash_bucket_size=1000)

    # Continuous base columns.
    age = tf.contrib.layers.real_valued_column("age")
    age_buckets = tf.contrib.layers.bucketized_column(age, boundaries=[18, 25, 30, 35, 40, 45, 50, 55, 60, 65])
    education_num = tf.contrib.layers.real_valued_column("education_num")
    capital_gain = tf.contrib.layers.real_valued_column("capital_gain")
    capital_loss = tf.contrib.layers.real_valued_column("capital_loss")
    hours_per_week = tf.contrib.layers.real_valued_column("hours_per_week")
  
  
### C. The Wide Model: Linear Model with Crossed Feature Columns

The wide model is a linear model with a wide set of sparse and crossed feature columns:

    wide_columns = [
    gender, native_country, education, occupation, workclass, relationship, age_buckets,
    tf.contrib.layers.crossed_column([education, occupation], hash_bucket_size=int(1e4)),
    tf.contrib.layers.crossed_column([native_country, occupation], hash_bucket_size=int(1e4)),
    tf.contrib.layers.crossed_column([age_buckets, education, occupation], hash_bucket_size=int(1e6)) ]
    
Wide models with crossed feature columns can memorize sparse interactions between features effectively. That being said, one limitation of crossed feature columns is that they do not generalize to feature combinations that have not appeared in the training data. Let's add a deep model with embeddings to fix that.


### D. The Deep Model: Neural Network with Embeddings

The deep model is a feed-forward neural network as shown in the slide deck (refer presentation slide number 12). Each of the sparse, high-dimensional categorical features are first converted into a low-dimensional and dense real-valued vector, often referred to as an embedding vector. These low-dimensional dense embedding vectors are concatenated with the continuous features, and then fed into the hidden layers of a neural network in the forward pass. The embedding values are initialized randomly, and are trained along with all other model parameters to minimize the training loss. 

We'll configure the embeddings for the categorical columns using embedding_column, and concatenate them with the continuous columns:

    deep_columns = [
      tf.contrib.layers.embedding_column(workclass, dimension=8),
      tf.contrib.layers.embedding_column(education, dimension=8),
      tf.contrib.layers.embedding_column(gender, dimension=8),
      tf.contrib.layers.embedding_column(relationship, dimension=8),
      tf.contrib.layers.embedding_column(native_country, dimension=8),
      tf.contrib.layers.embedding_column(occupation, dimension=8),
      age, education_num, capital_gain, capital_loss, hours_per_week]
      
The higher the dimension of the embedding is, the more degrees of freedom the model will have to learn the representations of the features.


### E. Combining Wide and Deep Models into One

The wide models and deep models are combined by summing up their final output log odds as the prediction, then feeding the prediction to a logistic loss function. All the graph definition and variable allocations have already been handled for you under the hood, so you simply need to create a DNNLinearCombinedClassifier:

    import tempfile
    model_dir = tempfile.mkdtemp()
    m = tf.contrib.learn.DNNLinearCombinedClassifier(
        model_dir=model_dir,
        linear_feature_columns=wide_columns,
        dnn_feature_columns=deep_columns,
        dnn_hidden_units=[100, 50])
        
### F. Training and Evaluating The Model

Before we train the model, let's read in the Census dataset. The code for input data processing is provided here again for your convenience:

    import pandas as pd
    import urllib


    COLUMNS = ["age", "workclass", "fnlwgt", "education", "education_num",
      "marital_status", "occupation", "relationship", "race", "gender",
      "capital_gain", "capital_loss", "hours_per_week", "native_country", "income_bracket"]
    LABEL_COLUMN = 'label'
    CATEGORICAL_COLUMNS = ["workclass", "education", "marital_status", "occupation",
                          "relationship", "race", "gender", "native_country"]
    CONTINUOUS_COLUMNS = ["age", "education_num", "capital_gain", "capital_loss",
                          "hours_per_week"]


    train_file = tempfile.NamedTemporaryFile()
    test_file = tempfile.NamedTemporaryFile()
    urllib.urlretrieve("http://mlr.cs.umass.edu/ml/machine-learning-databases/adult/adult.data", train_file.name)
    urllib.urlretrieve("http://mlr.cs.umass.edu/ml/machine-learning-databases/adult/adult.test", test_file.name)


    df_train = pd.read_csv(train_file, names=COLUMNS, skipinitialspace=True)
    df_test = pd.read_csv(test_file, names=COLUMNS, skipinitialspace=True, skiprows=1)
    df_train[LABEL_COLUMN] = (df_train['income_bracket'].apply(lambda x: '>50K' in x)).astype(int)
    df_test[LABEL_COLUMN] = (df_test['income_bracket'].apply(lambda x: '>50K' in x)).astype(int)

    def input_fn(df):
  
      continuous_cols = {k: tf.constant(df[k].values)
                        for k in CONTINUOUS_COLUMNS}
  
      categorical_cols = {k: tf.SparseTensor(
          indices=[[i, 0] for i in range(df[k].size)],
          values=df[k].values,
          shape=[df[k].size, 1])
                          for k in CATEGORICAL_COLUMNS}
 
      feature_cols = dict(continuous_cols.items() + categorical_cols.items())
  
      label = tf.constant(df[LABEL_COLUMN].values)
  
      return feature_cols, label

    def train_input_fn():
      return input_fn(df_train)

    def eval_input_fn():
      return input_fn(df_test)
      
      
After reading the data, you can train and evaluate the model:
    
    m.fit(input_fn=train_input_fn, steps=200)
    results = m.evaluate(input_fn=eval_input_fn, steps=1)
    for key in sorted(results):
        print "%s: %s" % (key, results[key])
        
        
