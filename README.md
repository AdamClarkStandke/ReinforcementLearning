# CLICK BELOW TO SEE the DOUBLE DQN ALORITHM PLAY MARIO and WIN:!!!! 
[![CLICK HERE](https://github.com/aCStandke/ReinforcementLearning/blob/main/mario.png)](https://youtu.be/iucn3RA2bWc)


## Reference:
Quick implementation of Pytorch's 'Train a Mario-Playing Agent as found at [Reinforcement](https://pytorch.org/tutorials/intermediate/mario_rl_tutorial.html).

## Results of Training the Model for 10000 episodes:
The results of training the model for 10000 episodes can be found in the code section of: [Train Mario 10000](https://github.com/aCStandke/ReinforcementLearning/blob/main/Training_Mario_10000Episodes.ipynb) 

After 10000 episodes, an average score of 334.503 was achieved (which is well below the scores documented by others using simple movements for Mario). Investigation of the reason why, will be done in a future project!

## Results of warm starting model for 100 episodes:
After training for another 100 episodes (i.e. warm starting) using the same memory max length and batch size of 32, only parameters relating to: 1) the experiences added to the cache before training; 2) the number of experiences added to the cache before updating the weights of the online DQN network and 3)`the frequency of synchronizing the model weights of the target DQN network were changed. The values and code can be found at [Warm-Start 100](https://github.com/aCStandke/ReinforcementLearning/blob/main/Experimental_Notebook.ipynb) 

## Theory of Double DQN Learning Algorithm:
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
