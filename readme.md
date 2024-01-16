# Neural Network for Chess Position Evaluation 

In this project we tried to create a simple neural network, which predicts who is currently winning in a chess game. The project also involves creating our own datasets with the help of [Stockfish](https://stockfishchess.org) the strongest chess engine at the time, and the ability to play against the trained neural network. Because this is a student project, and we have limited resources, we decided to focus on "mini chess", a chess variation with less pieces to make the training process easier and faster. 
 
Neural Network vs. Random Moves            |  Neural Network vs. Stockfish (Elo=200)
:-------------------------:|:-------------------------:
![Variation one](images/random.gif)  |  ![Variation two](images/stockfish200.gif)

## Disclaimer 
This project is a University Project for the Course [Neural Networks](https://www.unisi.it/ugov/degreecourse/480727) at the [Università degli Studi di Siena](https://www.unisi.it)

HAUPTORJECT MINICHESS
EXPERIMENTS NOT OPTIMIZED AND NOT PREPARED
RL CHERRY ON TOP

## Getting Started

To start creating your own data you can execute the dataCreation.py file.

```
python dataCreation.py --amount=10 --random=0.5
```

After creating the data you can train the model with the train.py file.

```
python train.py --epoch=100 --batch=100 --dataset=data/miniChess.csv --name=model.pt --lr=0.1
```

The final step is now to play against the trained model, or let the trained model play against a bot.

```
python play.py --model=models/model.pt --play=y
```

### Installing

To install all the libraries that we are using run pip install with the requirements.txt

```
pip install -r requirements.txt
```


## Authors

* **Silas Ueberschaer** - [Deischox](https://github.com/Deischox)

* **Benjamin Pöhlmann** - [Bepo1337](https://github.com/Bepo1337)

## Acknowledgments

* 
* Inspiration
* etc
