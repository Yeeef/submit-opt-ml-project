# SGD
# trivially increasing the batch size
python main.py --opt sgd --lr 0.025 --bs 32
python main.py --opt sgd --lr 0.025 --bs 64
python main.py --opt sgd --lr 0.025 --bs 128
python main.py --opt sgd --lr 0.025 --bs 256
python main.py --opt sgd --lr 0.025 --bs 512
python main.py --opt sgd --lr 0.025 --bs 1024
python main.py --opt sgd --lr 0.025 --bs 2048

# linear scale rule + warmup
python main.py --opt sgd --lr 0.025 --bs 32 --warmup_epochs 10
python main.py --opt sgd --lr 0.05 --bs 64 --warmup_epochs 10
python main.py --opt sgd --lr 0.1 --bs 128 --warmup_epochs 10
python main.py --opt sgd --lr 0.2 --bs 256 --warmup_epochs 10
python main.py --opt sgd --lr 0.4 --bs 512 --warmup_epochs 10
python main.py --opt sgd --lr 0.8 --bs 1024 --warmup_epochs 10
python main.py --opt sgd --lr 1.6 --bs 2048 --warmup_epochs 10

# simulation of multi-worker training
python main.py --opt sgd --lr 0.025 --bs 32 --warmup_epochs 10 --psuedo 32
python main.py --opt sgd --lr 0.05 --bs 64 --warmup_epochs 10 --psuedo 32
python main.py --opt sgd --lr 0.1 --bs 128 --warmup_epochs 10 --psuedo 32
python main.py --opt sgd --lr 0.2 --bs 256 --warmup_epochs 10 --psuedo 32
python main.py --opt sgd --lr 0.4 --bs 512 --warmup_epochs 10 --psuedo 32
python main.py --opt sgd --lr 0.8 --bs 1024 --warmup_epochs 10 --psuedo 32
python main.py --opt sgd --lr 1.6 --bs 2048 --warmup_epochs 10 --psuedo 32




