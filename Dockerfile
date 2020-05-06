FROM python:3

WORKDIR /app

ADD criteria.py denoising_revolutions_gui.py helpers.py hypersearch.py Main.py mibi_dataloader.py mibi_pickle.py modules.py prep_datasets.py rmodules.py run_trial.py utils.py models /app/
RUN pip install torch torchvision scikit-image scipy numpy kornia