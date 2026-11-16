import pandas as pd

FEATURE_COLUMNS = ['Age', 'Height', 'Weight', 'Duration', 'Heart_Rate', 'Body_Temp',
       'Sex_male', 'intensity', 'scope_Of_Work', 'Weight_X_Duration']

def encode_sex(DF):
    df = DF.copy()
    df['Sex_male'] = (df['Sex'] == 'male').astype(int) 
    df = df.drop(columns=['Sex'])
    return df


def add_features(DF):
    df = DF.copy()
    df['intensity'] = df['Heart_Rate'] / (220 - df['Age'])
    df['scope_Of_Work'] = df['Duration'] * df['Heart_Rate']
    df['Weight_X_Duration'] = df['Weight'] * df['Duration']
    return df

def prepare_features(DF):
    # Преобразуем Sex_male
    df = encode_sex(DF)
    df = add_features(df)

    return df[FEATURE_COLUMNS]