import os
from huggingface_hub import snapshot_download

def download_hf_dataset():
    # Nome del repository su Hugging Face
    dataset_name = "huyva/MIND-small"

    # Cartella locale dove salvare il dataset sul tuo computer/HPC
    output_dir = "./data"

    print(f" Avvio del download per il dataset: '{dataset_name}'...")

    try:
        # MIND-small è composto da file grezzi (news.tsv, behaviors.tsv,
        # popularity_data.csv) con schemi diversi tra loro: usare
        # load_dataset() li fonderebbe erroneamente in un'unica tabella.
        # Scarichiamo quindi i file così come sono, mantenendo la struttura
        # train/dev del repository.
        local_path = snapshot_download(
            repo_id=dataset_name,
            repo_type="dataset",
            local_dir=output_dir,
        )

        print("\n Download completato con successo!")
        print(f" Completato! Path assoluto del dataset: {os.path.abspath(local_path)}")

    except Exception as e:
        print(f"\n Si è verificato un errore durante il download: {e}")

if __name__ == "__main__":
    download_hf_dataset()