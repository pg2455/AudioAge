export KMP_DUPLICATE_LIB_OK=TRUE
python -m pdb transfer.py -cuda=False \
      -base_modeldir=/Users/pgupta/Workspace/audio/exp1_male_anom \
      -modeldir=/Users/pgupta/Workspace/audio/neurolex/models/transfer \
      -gender=male \
      -anomaly=True
