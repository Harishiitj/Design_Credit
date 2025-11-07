import json
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

OUTPUT_FILENAME = "features_including.json"

# ==================================================
# Load included features
# ==================================================
input_filename = "included_features.json"
with open(input_filename, 'r', encoding='utf-8') as f:
        data = json.load(f)
        logger.info("Loaded %d features to be included from %s", len(data), input_filename)

included_features = set(data)
logger.info("Included features set created with %d entries.", len(included_features))


# ==================================================
# Load all features
# ==================================================
with open("features.json", 'r', encoding='utf-8') as f:
    features = json.load(f)
    logger.info("Loaded %d total features from features.json", len(features))

feature_keys = set(feat.get('key_name', '').lower() for feat in features)

# ==================================================
# warn unmatched included features
# ==================================================
unmatched_included_features = included_features.difference(feature_keys)
if unmatched_included_features:
    logger.warning("The following included features were not found in the features list:")
    for feat in unmatched_included_features:
        logger.warning(" - %s", feat)

# ==================================================
# Filter features
# ==================================================

filtered_features = [feat for feat in features if feat.get('key_name', '').lower() in included_features]
logger.info("Filtered features count: %d", len(filtered_features))
with open(OUTPUT_FILENAME, 'w', encoding='utf-8') as f:
    json.dump(filtered_features, f, indent=2)
    logger.info("Filtered features written to %s", OUTPUT_FILENAME)

# ==================================================
# load features excluding
# ==================================================
with open("features_excluding.json", 'r', encoding='utf-8') as f:
    features_excluding = json.load(f)
    logger.info("Loaded %d features from features_excluding.json", len(features_excluding))

features_excluding_keys = set(feat.get('key_name', '').lower() for feat in features_excluding)

# ==================================================
# remaining after removing included otherwise
# =================================================
remaining_features = [feat for feat in features_excluding if feat.get('key_name', '').lower() not in included_features]
logger.info("Remaining features count after excluding included: %d", len(remaining_features))
with open("features_remaining.json", 'w', encoding='utf-8') as f:
    json.dump(remaining_features, f, indent=2)
    logger.info("Remaining features written to features_remaining.json")