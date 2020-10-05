# **oza's custom FBNet (cifar10 DataSet)**

This repository reproduces the results of the following paper:

[**FBNet: Hardware-Aware Efficient ConvNet Design via Differentiable Neural Architecture Search**](https://arxiv.org/pdf/1812.03443.pdf)  
Bichen Wu1, Xiaoliang Dai, Peizhao Zhang, Yanghan Wang, Fei Sun, Yiming Wu, Yuandong Tian, Peter Vajda, Yangqing Jia, Kurt Keutzer (Fasebook Research)

Layers to Search are from a [FacebookResearch repository](https://github.com/facebookresearch/maskrcnn-benchmark/tree/master/maskrcnn_benchmark/modeling/backbone)
Utils stuff is taken from [DARTS repository](https://github.com/quark0/darts/blob/master/cnn/utils.py)

# FBNet_train(FBNet_tmp)

* 基本的にはこちらで学習を行い, FBNet_load(FBNet)に学習履歴やパラメータを格納
* ただし, 混乱を防ぐために両者に格納をしている
* FBNet_learn/supernet_functions/logsには, FBNetで探索したモデル(パラメータ)の結果を格納
* python3 supernet_main_file.py --train_or_sample trainでFBnetを学習し, モデルを探索(学習)
* 学習の結果, 獲得されたパラメータ(モデル)はFBNet_learn/supernet_functions/logsに格納
* その後, python3 supernet_main_file.py --train_or_sample sample --architecture_name my_unique_name_for_architecture --hardsampling_bool_value True でFBNet_learn/fbnet_building_blocks/fbnet_modeldef.pyにモデル構造を書き込み(logs内のbest_model.pthを元に書き込むので配置に注意)
* そして, python3 architecture_main_file.py --architecture_name [自分で指定, fbnet_modeldef.pyのアーキテクチャ]で学習し, cifar10でのaccuracy算出


# FBNet_load(FBNet)
* cifar10でのaccuracy, latency, number of parametersを算出するために, 学習済みのparameterを配置しているが, 実態はFBNet_trainと同様






