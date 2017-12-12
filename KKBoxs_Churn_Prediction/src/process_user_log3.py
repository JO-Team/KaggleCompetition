import pandas as pd

user_log_v1 = pd.read_csv('../input/processed_features_log_final_v2.csv')

user_log_v1 = user_log_v1.drop_duplicates()

user_log_v1.to_csv("../input/processed_features_log_final_v2_nondup.csv", index=False)