#Portfolio Manager AI

###Introduction
Simulation Environment for stock market and agent which takes the input as Markov decision process. The input is
mathematical indicators deriven from price and volume of stock which passes through a neural network that tries to 
optimize alpha generated and minimize risk. 

### Installation Guide



### TODO
- [x] Simulation environment
- [x] Agent interaction
- [ ] W&B data and model versioning
- [ ] Streamlit for analysis for training - **25%**
- [x] Dynamic Clustering of stocks
- [ ] UI for agent training 
- [x] Pretrain from date to date
- [ ] Pretrain with Expert dataset
- [ ] Multi input dictionary from env
- [x] Time window input 
- [x] Train process with time window input
- [ ] Dictionary observation for prices / indicators
- [ ] Take flag days out of environment
- [ ] Cirriculum learning
  - [ ] Learn not sell for loss for generalization
- [ ] Validation of model for different time horizons for PBT
- [ ] Day Trading Environment
- [ ] MARL
- [ ] Ensemble voting for trading strategy
- Data Selection Add to Preprocess
  - [ ] Correlation Table as input
  - [x] Past prices for the input
  - [ ] Avg. bought price of stocks in environment as a fuction
  - [x] Dynamic Clustering 
  - [x] Normalized inputs for the model
  - [ ] Switching stocks and keeping the same stock holding if is in same list
  
### To start
1. pip install -r requirements.txt
2. StockTradingBot.ipynb notebook is for training process