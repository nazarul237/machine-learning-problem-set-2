from pathlib import Path
import pandas as pd
from gensim.models.doc2vec import Doc2Vec, TaggedDocument

# =============================================================================
# STEP 19: TRAIN DOC2VEC AND GENERATE DOCUMENT VECTORS
# =============================================================================
# Purpose of this step:
# In Step 18, the train, validation, and test narratives were prepared for
# Doc2Vec by cleaning the text lightly and converting each narrative into a
# tokenised form.
#
# The goal of the present step is to move from tokenised text to dense document
# embeddings. In other words, each narrative will now be represented as a fixed-
# length numeric vector rather than a sparse TF-IDF representation.
#
# This matters because Method 2 is intended to test a genuinely different text
# representation approach from Method 1:
# - Method 1 used TF-IDF sparse word features
# - Method 2 uses Doc2Vec dense document vectors
#
# Important methodological note:
# The Doc2Vec model is trained ONLY on the training split.
# The validation and test narratives are not used to fit the embedding model.
# Instead, their vectors are inferred afterwards. This preserves the integrity
# of the time-based split and avoids information leakage from later data.
# =============================================================================


# -----------------------------------------------------------------------------
# Set up file paths
# -----------------------------------------------------------------------------
project_folder = Path(__file__).resolve().parent

train_input = project_folder / "step18_doc2vec_train_prepared.csv"
validation_input = project_folder / "step18_doc2vec_validation_prepared.csv"
test_input = project_folder / "step18_doc2vec_test_prepared.csv"

model_output = project_folder / "step19_doc2vec_model.model"
train_vectors_output = project_folder / "step19_doc2vec_train_vectors.csv"
validation_vectors_output = project_folder / "step19_doc2vec_validation_vectors.csv"
test_vectors_output = project_folder / "step19_doc2vec_test_vectors.csv"
summary_output = project_folder / "step19_doc2vec_vector_summary.csv"


# -----------------------------------------------------------------------------
# Load the prepared text files
# -----------------------------------------------------------------------------
train_df = pd.read_csv(train_input, low_memory=False)
validation_df = pd.read_csv(validation_input, low_memory=False)
test_df = pd.read_csv(test_input, low_memory=False)

print("=" * 100)
print("STEP 19: TRAIN DOC2VEC AND GENERATE VECTORS")
print("=" * 100)
print(f"Train rows loaded:      {len(train_df)}")
print(f"Validation rows loaded: {len(validation_df)}")
print(f"Test rows loaded:       {len(test_df)}")
print()


# -----------------------------------------------------------------------------
# Convert the stored token strings back into token lists
# -----------------------------------------------------------------------------
# In Step 18, tokenised text was stored as a single space-joined string so that
# it could be saved safely in CSV format. Here we convert that string back into
# a Python list of tokens, because gensim's Doc2Vec expects token lists.
# -----------------------------------------------------------------------------
def split_tokenised_text(text):
    """
    Convert a space-joined token string back into a list of tokens.

    Parameters
    ----------
    text : str
        Tokenised text stored as a single string.

    Returns
    -------
    list[str]
        List of tokens.
    """
    if pd.isna(text):
        return []

    text = str(text).strip()

    if text == "":
        return []

    return text.split(" ")


train_df["token_list"] = train_df["tokenised_text"].apply(split_tokenised_text)
validation_df["token_list"] = validation_df["tokenised_text"].apply(split_tokenised_text)
test_df["token_list"] = test_df["tokenised_text"].apply(split_tokenised_text)


# -----------------------------------------------------------------------------
# Create TaggedDocument objects for training
# -----------------------------------------------------------------------------
# Doc2Vec needs each training document to have:
# - a list of words
# - a unique tag
#
# The tags are simply identifiers for each document in the training corpus.
# They do not carry label information.
# -----------------------------------------------------------------------------
tagged_train_documents = [
    TaggedDocument(words=row["token_list"], tags=[f"TRAIN_{i}"])
    for i, row in train_df.iterrows()
]

print("Tagged training documents created.")
print(f"Number of tagged training documents: {len(tagged_train_documents)}")
print()


# -----------------------------------------------------------------------------
# Define and train the Doc2Vec model
# -----------------------------------------------------------------------------
# These are sensible starting settings for an MSc project:
# - vector_size = 100: each document becomes a 100-dimensional vector
# - window = 5: local context window
# - min_count = 3: ignore very rare words
# - dm = 1: use the distributed memory variant of Doc2Vec
# - epochs = 40: enough learning passes to produce stable vectors
#
# The model is trained only on the training corpus.
# -----------------------------------------------------------------------------
doc2vec_model = Doc2Vec(
    vector_size=100,
    window=5,
    min_count=3,
    workers=1,
    dm=1,
    epochs=40,
    seed=42
)

# Build vocabulary from training documents only
doc2vec_model.build_vocab(tagged_train_documents)

print("Doc2Vec vocabulary built.")
print(f"Vocabulary size: {len(doc2vec_model.wv)}")
print()

# Train the model on training documents only
doc2vec_model.train(
    tagged_train_documents,
    total_examples=doc2vec_model.corpus_count,
    epochs=doc2vec_model.epochs
)

# Save the trained Doc2Vec model for reproducibility
doc2vec_model.save(str(model_output))

print("Doc2Vec model training completed.")
print()


# -----------------------------------------------------------------------------
# Infer vectors for train, validation, and test
# -----------------------------------------------------------------------------
# For the training split, we also infer vectors from the token lists rather than
# directly reading internal document vectors. This keeps the process consistent
# across train, validation, and test.
#
# Each narrative is converted into a dense numeric vector of length 100.
# -----------------------------------------------------------------------------
def infer_vectors_for_split(df, split_name, model):
    """
    Infer document vectors for one dataset split.

    Parameters
    ----------
    df : pandas.DataFrame
        Prepared dataframe containing token_list and metadata.
    split_name : str
        Name of the split for logging.
    model : gensim.models.doc2vec.Doc2Vec
        Trained Doc2Vec model.

    Returns
    -------
    pandas.DataFrame
        Dataframe containing metadata and vector columns.
    """
    print(f"Inferring vectors for: {split_name}")

    vectors = []
    for tokens in df["token_list"]:
        vec = model.infer_vector(tokens, epochs=30)
        vectors.append(vec)

    vectors_df = pd.DataFrame(vectors, columns=[f"vec_{i}" for i in range(model.vector_size)])

    output_df = pd.concat(
        [
            df[["ACN", "incident_year", "label_unstable", "data_split"]].reset_index(drop=True),
            vectors_df
        ],
        axis=1
    )

    print(f"{split_name} vectors shape: {output_df.shape}")
    print()

    return output_df


train_vectors_df = infer_vectors_for_split(train_df, "train", doc2vec_model)
validation_vectors_df = infer_vectors_for_split(validation_df, "validation", doc2vec_model)
test_vectors_df = infer_vectors_for_split(test_df, "test", doc2vec_model)


# -----------------------------------------------------------------------------
# Save the vector files
# -----------------------------------------------------------------------------
train_vectors_df.to_csv(train_vectors_output, index=False)
validation_vectors_df.to_csv(validation_vectors_output, index=False)
test_vectors_df.to_csv(test_vectors_output, index=False)


# -----------------------------------------------------------------------------
# Save a compact summary table
# -----------------------------------------------------------------------------
summary_df = pd.DataFrame([
    {
        "split": "train",
        "rows": len(train_vectors_df),
        "unstable_cases": int(train_vectors_df["label_unstable"].sum()),
        "vector_dimensions": 100
    },
    {
        "split": "validation",
        "rows": len(validation_vectors_df),
        "unstable_cases": int(validation_vectors_df["label_unstable"].sum()),
        "vector_dimensions": 100
    },
    {
        "split": "test",
        "rows": len(test_vectors_df),
        "unstable_cases": int(test_vectors_df["label_unstable"].sum()),
        "vector_dimensions": 100
    }
])

summary_df.to_csv(summary_output, index=False)


# -----------------------------------------------------------------------------
# Print final output summary
# -----------------------------------------------------------------------------
print("=" * 100)
print("DOC2VEC VECTOR GENERATION SUMMARY")
print("=" * 100)
print(summary_df.to_string(index=False))
print()

print("FILES CREATED")
print("-" * 100)
print(model_output.name)
print(train_vectors_output.name)
print(validation_vectors_output.name)
print(test_vectors_output.name)
print(summary_output.name)
print()

print("=" * 100)
print("STEP 19 FINISHED")
print("The Doc2Vec model has been trained on the training split only, and document")
print("vectors have been generated for train, validation, and test.")
print("No classifier has been trained yet.")
print("=" * 100)