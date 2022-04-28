# Reinforcement Learning to play Mario

## CLICK BELOW TO SEE the DOUBLE DQN ALORITHM PLAY MARIO:!!!! 
[![CLICK HERE](https://github.com/aCStandke/ReinforcementLearning/blob/main/mario.png)](https://www.youtube.com/watch?v=r3Y_ryFYPNg)


## Reference:
Quick implementation of Pytorch's 'Train a Mario-Playing Agent as found at [Reinforcement](https://pytorch.org/tutorials/intermediate/mario_rl_tutorial.html).

## Results of Training the Model for 10000 episodes:
The results of training the model for 10000 episodes can be found in the code section: [Train Mario 10000]()


## Code:
Code can be found here and expirmented with Colab by clicking the Colab button in the upper left corner of the notebook: [Train Mario](https://github.com/aCStandke/ReinforcementLearning/blob/main/DoubleDQN_Reinforement_Learning.ipynb)

Note: If you want to display a dumb Mario agent playing the game for one episode do the following:
  1. In the runtime tab at the top of the Colab notebook, select 'Runtime all' (very easy)

## Double DQN Learning Algorithm:
Mario's Action policy for solving sequential decision problems (ie., the machine brain, lol) is the DDQN algorithm as detailed in https://arxiv.org/pdf/1509.06461.pdf[1] The idea of Double Q-learning is to reduce overestimations by decomposing the max operation in the target into action selection and action evaluation[1]. The algorithm evaluates the greedy policy according to the online network and uses the target network to estimate its value[1].

![](https://github.com/aCStandke/ReinforcementLearning/blob/main/download.png)

The weights of the second network contain the weights of the target network Θₜ-for the evaluation of the current greedy policy. The update to the target network stays unchanged from DQN, and remains a periodic copy of the online network[1].Two values are involved in learning: TD Estimate - the predicted optimal 𝑸* for a given state s:

![](https://github.com/aCStandke/ReinforcementLearning/blob/main/download%20(1).png)


And TD Target - aggregation of current reward and the estimated 𝑸* in the next state s':

![](https://github.com/aCStandke/ReinforcementLearning/blob/main/download%20(2).png)

![](https://github.com/aCStandke/ReinforcementLearning/blob/main/download%20(3).png)

As Mario samples inputs from his replay buffer, we compute TDₜ and TDₑ and backpropagate this loss down Qₒₗᵢₙₑ to update its parameters θₒₗᵢₙₑ (α is the learning rate) as follows:

![](https://github.com/aCStandke/ReinforcementLearning/blob/main/download%20(4).png)

target weights do not get updated during backpropogation, instead rather weights from online are assigned to the weights to the target as follows:

![](https://github.com/aCStandke/ReinforcementLearning/blob/main/download%20(5).png)
