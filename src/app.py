import click

from src.dataset.make_dataset import Dataset
from src.features.build_features import label_encode
from src.model.train_model import train_xgb
from src.model.predict_model import write_submission

@click.command()
@click.option('--nrows_train', type=int, default=None, help='Number of training rows')
@click.option('--name', type=str, default='submission.csv', help='Submission name')
def run_experiment(nrows_train, name):
    ds = Dataset()
    ds.load_raw(nrows_train)
    label_encode(ds)
    xgb_clf = train_xgb(ds)
    write_submission(ds, xgb_clf, name)

if __name__ == '__main__':
    run_experiment(1000, 'main_submission')