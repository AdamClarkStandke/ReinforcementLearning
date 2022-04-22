# ReinforcementLearning

## Training Mario after 20,000 episodes using a batch size of 64:
[![Mario Game](https://github.com/aCStandke/ReinforcementLearning/blob/main/mario.png)](https://www.youtube.com/watch?v=r3Y_ryFYPNg)


## Reference:
Quick implementation of Pytorch's TRAIN A MARIO-PLAYING RL AGENT as found at [Reinforcement](https://pytorch.org/tutorials/intermediate/mario_rl_tutorial.html)

## Code:
Code can be found here (future comments will be added): [Train Mario](https://github.com/aCStandke/ReinforcementLearning/blob/main/ReinforementLearning.ipynb)


## Double DQN Learning Algorithm
Mario's Action policy for solving sequential decision problems (ie., the machine brain, lol) is the DDQN algorithm as detailed in https://arxiv.org/pdf/1509.06461.pdf[1] The idea of Double Q-learning is to reduce overestimations by decomposing the max operation in the target into action selection and action evaluation[1]. The algorithm evaluates the greedy policy according to the online network and uses the target network to estimate its value[1].

Screenshot 2022-04-22 3.02.36 PM.png

The weights of the second network contain the weights of the target network Θₜ-for the evaluation of the current greedy policy. The update to the target network stays unchanged from DQN, and remains a periodic copy of the online network[1].Two values are involved in learning: TD Estimate - the predicted optimal 𝑸* for a given state s

Screenshot 2022-04-22 3.37.47 PM.png

And TD Target - aggregation of current reward and the estimated 𝑸* in the next state s'

Screenshot 2022-04-22 3.37.59 PM.png

Screenshot 2022-04-22 3.38.06 PM.png

As Mario samples inputs from his replay buffer, we compute TDₜ and TDₑ and backpropagate this loss down Qₒₗᵢₙₑ to update its parameters θₒₗᵢₙₑ (α is the learning rate)

​Screenshot 2022-04-22 4.18.50 PM.png

theta_target weights does not get updated during backpropogation, instead rather weights from online are assigned to the weights to the target 
Screenshot 2022-04-22 4.23.55 PM.png
