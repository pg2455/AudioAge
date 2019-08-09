export KMP_DUPLICATE_LIB_OK=TRUE
python -m pdb train_anom.py -train_datadir=/Users/pgupta/Workspace/audio/Common_voice/Common_Voice_train/male/ages/ \
      -val_datadir=/Users/pgupta/Workspace/audio/Common_voice/Common_Voice_train/male/ages/ \
      -cuda=False \
      -modeldir=/Users/pgupta/Workspace/audio/neurolex/models/exp1
# python -m pdb data.py
