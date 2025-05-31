# import required libraries
from helper import *


# load data
train_df = pd.read_csv("train_data.csv")
test_df = pd.read_csv("test_data.csv")

# remove nans. they cant be vectorized
train_df = train_df.dropna()
test_df = test_df.dropna()

test_df.head()

# define train test
x_list = ['clean_text', 'domain']
y_list = ['label']

X_train = train_df[x_list]
y_train = train_df[y_list]
X_test = test_df[x_list]
y_test = test_df[y_list]

y_train = y_train.values.ravel() # flaten so it's labels are 1D
y_test = y_test.values.ravel() # flaten so it's labels are 1D

print('x train shape: ', X_train.shape)
print('y train shape: ', y_train.shape)


# model prediction
# pretrained llm using transformers and datasets libraries to fine-tune a bert-base-uncased.
from transformers import BertTokenizer
from transformers import TFBertForSequenceClassification

# load tokenizer
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

# tokenize text
def tokenize_texts(texts, tokenizer, max_length=128):
    return tokenizer(
        texts.tolist(),
        padding='max_length',
        truncation=True,
        max_length=max_length,
        return_tensors="tf"
    )

train_encodings = tokenize_texts(X_train["clean_text"], tokenizer)
test_encodings = tokenize_texts(X_test["clean_text"], tokenizer)
print('tokenized')


import tensorflow as tf
from transformers import BertTokenizer, TFBertForSequenceClassification
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow.keras.metrics import SparseCategoricalAccuracy

# set batch size
BATCH_SIZE = 16

# convert tokenized inputs and labels into tf.data.Dataset
train_dataset = tf.data.Dataset.from_tensor_slices((
    dict(train_encodings),
    y_train
)).shuffle(1000).batch(BATCH_SIZE)

test_dataset = tf.data.Dataset.from_tensor_slices((
    dict(test_encodings),
    y_test
)).batch(BATCH_SIZE)

print('converted data to tf.data.Dataset and batched')

# load pretrained bert model for sequence classification
model = TFBertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2)
print('loaded BERT model')

# compile model
model.compile(
    optimizer=Adam(learning_rate=2e-5),
    loss=SparseCategoricalCrossentropy(from_logits=True),
    metrics=[SparseCategoricalAccuracy()],
)
print('compiled model')

# train model
history = model.fit(
    train_dataset,
    validation_data=test_dataset,
    epochs=3
)
print('finished training')

# evaluate model
loss, accuracy = model.evaluate(test_dataset)
print(f"test accuracy: {accuracy:.4f}")
