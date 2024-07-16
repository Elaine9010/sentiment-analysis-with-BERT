#!/bin/bash


python bert_cls_finetuning.py --batch_size 32 --learning_rate 2e-5 --epochs 3
python bert_cls_finetuning.py --batch_size 16 --learning_rate 2e-5 --epochs 3
python bert_cls_finetuning.py --batch_size 32 --learning_rate 3e-5 --epochs 3
python bert_cls_finetuning.py --batch_size 16 --learning_rate 3e-5 --epochs 3
python bert_cls_finetuning.py --batch_size 32 --learning_rate 5e-5 --epochs 3
python bert_cls_finetuning.py --batch_size 16 --learning_rate 5e-5 --epochs 3
