python train_model.py --disable-dp
python train_model.py
cp ./.cache/privatemodel_best.pth.tar ./models/privatemodel_best.pth.tar
cp ./.cache/nonprivatemodel_best.pth.tar ./models/nonprivatemodel_best.pth.tar
