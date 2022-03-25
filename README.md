# DemoCodeRepo

## Simple Usage

Fetch data from dvc using tag, example:

    # DVC Setting
    path = 'data/wine-quality.csv'
    repo = 'D:\WorkSpace\gitFolder\Demo\DVCTutorial'
    version = 'v2'

    data_url = dvc.api.get_url(
        path=path,
        repo=repo,
        rev=version
    )

Log parameters using mlflow api:

    with mlflow.start_run():
        # Log data params
        mlflow.log_param('data_url', data_url)
        mlflow.log_param('data_version', version)
        mlflow.log_param('input_rows', data.shape[0])
        mlflow.log_param('input_cols', data.shape[1])
