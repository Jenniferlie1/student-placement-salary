from data_ingestion import ingest_data
from preprocessing import preprocess
from train import train
from evaluation import evaluate

ACCURACY_THRESHOLD = 0.8
R2_THRESHOLD = 0.5

def run_pipeline():
    print("Step 1: Data Ingestion")
    ingest_data()

    print("Step 2: Preprocessing")
    preprocess()
    print("Step 3: Training")
    run_id = train()

    print("Step 4: Evaluation")
    acc, r2 = evaluate(run_id)

    if acc >= ACCURACY_THRESHOLD and r2 >= R2_THRESHOLD:
        print("Model approved for deployment")
    else:
        print("Model rejected")


if __name__ == "__main__":
    run_pipeline()