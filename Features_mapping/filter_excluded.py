import json
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

OUTPUT_FILENAME = "features_excluding.json"

# ==================================================
# Load excluded features
# ==================================================

input_filename = "excluded_features.json"
with open(input_filename, 'r', encoding='utf-8') as f:
        data = json.load(f)
        logger.info("Loaded %d features to be excluded from %s", len(data), input_filename)

excluded_features = set(data)
logger.info("Excluded features set created with %d entries.", len(excluded_features))

# ==================================================
# Load all features
# ==================================================
with open("features.json", 'r', encoding='utf-8') as f:
    features = json.load(f)
    logger.info("Loaded %d total features from features.json", len(features))

feature_keys = set(feat['key_name'].lower() for feat in features)

# ==================================================
# count unmatched excluded features
# ==================================================
unmatched_features = excluded_features.difference(feature_keys)
if unmatched_features:
    logger.warning("The following excluded features were not found in the features list:")
    for feat in unmatched_features:
        logger.warning(" - %s", feat)

# ==================================================
# Filter features
# ==================================================

filtered_features = [feat for feat in features if feat['key_name'].lower() not in excluded_features]
logger.info("Filtered features count: %d", len(filtered_features))
with open(OUTPUT_FILENAME, 'w', encoding='utf-8') as f:
    json.dump(filtered_features, f, indent=2)
    logger.info("Filtered features written to %s", OUTPUT_FILENAME)