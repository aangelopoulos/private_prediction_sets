python train_model.py > ./outputs/privatemodel_terminal.txt 
cp ./.cache/privatemodel_best.pth.tar ./models/privatemodel_best.pth.tar
python train_model.py --disable-dp > ./outputs/nonprivatemodel_terminal.txt  
cp ./.cache/nonprivatemodel_best.pth.tar ./models/nonprivatemodel_best.pth.tar
