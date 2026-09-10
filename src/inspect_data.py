import pandas as pd

df = pd.read_csv(
    "data/twcs/twcs.csv",
    nrows=100000
)

# IDs ko same type: string
df["tweet_id"] = df["tweet_id"].astype(str)
df["response_tweet_id"] = df["response_tweet_id"].astype(str)

# Customer tweets
customers = df[df["inbound"] == True].copy()

# AmazonHelp replies
amazon = df[df["author_id"] == "AmazonHelp"].copy()

# Customer → AmazonHelp reply
conversations = customers.merge(
    amazon,
    left_on="response_tweet_id",
    right_on="tweet_id",
    suffixes=("_customer", "_amazon")
)

print("Customer tweets:", len(customers))
print("AmazonHelp replies:", len(amazon))
print("Matched conversations:", len(conversations))

print("\nCustomer → AmazonHelp examples:\n")

print(
    conversations[
        [
            "tweet_id_customer",
            "text_customer",
            "tweet_id_amazon",
            "text_amazon"
        ]
    ].head(10).to_string(index=False)
)  
print("\nRandom 20 matched conversations:\n")

print(
    conversations[
        [
            "tweet_id_customer",
            "text_customer",
            "tweet_id_amazon",
            "text_amazon"
        ]
    ].sample(20, random_state=42).to_string(index=False)
)
# Obvious conversational noise ko identify karna
noise_words = [
    "thanks",
    "thank you",
    "you're welcome",
    "you are welcome",
    "okay",
    "ok",
]

# Customer message ko lowercase mein convert karke check karenge
customer_text_lower = conversations["text_customer"].str.lower()

# Agar message mein upar wale words hain to usko noise maanenge
noise_mask = customer_text_lower.str.contains(
    "|".join(noise_words),
    na=False
)

clean_conversations = conversations[~noise_mask].copy()

# Customer messages ko lowercase mein convert kar rahe hain
text = clean_conversations["text_customer"].str.lower()

# Common support-related words ki frequency
keywords = [
    "order",
    "delivery",
    "delivered",
    "refund",
    "return",
    "payment",
    "account",
    "prime",
    "cancel",
    "package",
    "shipping",
    "password",
    "login",
    "product",
    "broken",
    "damaged"
]

print("\nCommon support keywords:\n")

for word in keywords:
    count = text.str.contains(word, na=False).sum()
    print(f"{word}: {count}")

print("\nBefore noise filtering:", len(conversations))
print("After noise filtering:", len(clean_conversations))

print("\nClean examples:\n")

print(
    clean_conversations[
        [
            "text_customer",
            "text_amazon"
        ]
    ].sample(10, random_state=42).to_string(index=False)
)
print("\nGolden dataset candidates:\n")

golden_candidates = clean_conversations[
    [
        "tweet_id_customer",
        "text_customer",
        "text_amazon"
    ]
].sample(10, random_state=100)

print(
    golden_candidates.to_string(index=False)
)
# Customer message ko lowercase mein convert kar rahe hain
text = clean_conversations["text_customer"].str.lower().str.strip()

# Bahut chhote messages usually useful support request nahi hote
short_message = text.str.len() < 15

# Obvious positive/praise messages
positive_words = [
    "great",
    "awesome",
    "excellent",
    "love amazon",
    "love you amazon",
    "good job",
    "well done",
    "fast delivery",
    "happy with",
    "amazing service"
]

positive_message = text.str.contains(
    "|".join(positive_words),
    na=False
)

# Dono types ko remove karenge
quality_mask = short_message | positive_message

quality_conversations = clean_conversations[
    ~quality_mask
].copy()

print("Before quality filtering:", len(clean_conversations))
print("After quality filtering:", len(quality_conversations))
print("Removed:", quality_mask.sum())
print("\nSTEP 13 - Golden dataset:\n")

golden = quality_conversations[
    [
        "tweet_id_customer",
        "text_customer",
        "text_amazon"
    ]
].sample(20, random_state=123)

print(
    golden.to_string(index=False)
)
# STEP 15 - Golden dataset

golden = quality_conversations[
    [
        "tweet_id_customer",
        "text_customer",
        "text_amazon"
    ]
].sample(20, random_state=123).copy()

# Manual labels
labels = [
    "order_delivery",
    "INVALID",
    "INVALID",
    "other",
    "order_delivery",
    "INVALID",
    "order_delivery",
    "INVALID",
    "other",
    "order_delivery",
    "other",
    "order_delivery",
    "other",
    "product_issue",
    "order_delivery",
    "INVALID",
    "order_delivery",
    "cancellation",
    "INVALID",
    "order_delivery"
]

# Labels ko new column mein add karna
golden["intent"] = labels

# Valid/invalid column
golden["valid"] = golden["intent"] != "INVALID"

# Column names simple bana rahe hain
golden = golden.rename(
    columns={
        "tweet_id_customer": "tweet_id",
        "text_customer": "customer_text",
        "text_amazon": "amazon_response"
    }
)

# CSV save
golden.to_csv(
    "data/golden.csv",
    index=False
)

print("\nGolden dataset saved!")
print("Rows:", len(golden))
print(golden.to_string(index=False))

print("\nSTEP 16 - Next 20 golden candidates:\n")

next_golden = quality_conversations[
    [
        "tweet_id_customer",
        "text_customer",
        "text_amazon"
    ]
].sample(20, random_state=456)

print(
    next_golden.to_string(index=False)
)
print("\nSTEP 16 - Labeled golden batch:\n")

next_golden = quality_conversations[
    [
        "tweet_id_customer",
        "text_customer",
        "text_amazon"
    ]
].sample(20, random_state=456).copy()

# Manual labels
next_labels = [
    "other",
    "order_delivery",
    "order_delivery",
    "order_delivery",
    "refund_return",
    "INVALID",
    "order_delivery",
    "order_delivery",
    "order_delivery",
    "order_delivery",
    "order_delivery",
    "cancellation",
    "other",
    "other",
    "INVALID",
    "product_issue",
    "payment_issue",
    "product_issue",
    "other",
    "other"
]

next_golden["intent"] = next_labels

next_golden["valid"] = next_golden["intent"] != "INVALID"

next_golden = next_golden.rename(
    columns={
        "tweet_id_customer": "tweet_id",
        "text_customer": "customer_text",
        "text_amazon": "amazon_response"
    }
)

print(next_golden.to_string(index=False))
print("\nSTEP 17 - Combining golden datasets:\n")

# Load existing first 20 golden examples
golden_old = pd.read_csv("data/golden.csv")

# Correct two labels from the first batch
golden_old.loc[
    golden_old["tweet_id"] == 95309,
    "intent"
] = "product_issue"

golden_old.loc[
    golden_old["tweet_id"] == 38565,
    "intent"
] = "INVALID"

# Update valid column after corrections
golden_old["valid"] = golden_old["intent"] != "INVALID"


# Step 16 dataset
golden_new = next_golden.copy()

# Combine both
golden_combined = pd.concat(
    [golden_old, golden_new],
    ignore_index=True
)

# Save final combined dataset
golden_combined.to_csv(
    "data/golden.csv",
    index=False
)

print("Total golden examples:", len(golden_combined))
print("\nIntent distribution:")
print(golden_combined["intent"].value_counts())

print("\nSaved to data/golden.csv")
print("\nSTEP 18 - Next 20 golden candidates:\n")

next_golden = quality_conversations[
    [
        "tweet_id_customer",
        "text_customer",
        "text_amazon"
    ]
].sample(20, random_state=789)

print(next_golden.to_string(index=False))


print("\nSTEP 18 - Labeled golden batch:\n")

next_golden = quality_conversations[
    [
        "tweet_id_customer",
        "text_customer",
        "text_amazon"
    ]
].sample(20, random_state=789).copy()

next_labels = [
    "payment_issue",
    "INVALID",
    "other",
    "payment_issue",
    "order_delivery",
    "order_delivery",
    "INVALID",
    "order_delivery",
    "order_delivery",
    "order_delivery",
    "order_delivery",
    "other",
    "other",
    "payment_issue",
    "order_delivery",
    "other",
    "order_delivery",
    "refund_return",
    "product_issue",
    "other"
]

next_golden["intent"] = next_labels
next_golden["valid"] = next_golden["intent"] != "INVALID"

next_golden = next_golden.rename(
    columns={
        "tweet_id_customer": "tweet_id",
        "text_customer": "customer_text",
        "text_amazon": "amazon_response"
    }
)

print(next_golden.to_string(index=False))
print("\nSTEP 19 - Adding Step 18 to golden dataset:\n")

# Load existing 40 examples
golden_old = pd.read_csv("data/golden.csv")

# Combine old + new
golden_combined = pd.concat(
    [golden_old, next_golden],
    ignore_index=True
)

# Remove accidental duplicates, if any
golden_combined = golden_combined.drop_duplicates(
    subset=["tweet_id"]
)

# Save
golden_combined.to_csv(
    "data/golden.csv",
    index=False
)

print("Total golden examples:", len(golden_combined))

print("\nIntent distribution:")
print(golden_combined["intent"].value_counts())

print("\nSaved to data/golden.csv")