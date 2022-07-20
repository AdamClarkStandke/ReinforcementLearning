# Theory of DQN and the Double DQN Learning Algorithm:
As detailed by [Deep Reinforcement Learning with Double Q-learning](https://arxiv.org/pdf/1509.06461.pdf) 

> A deep Q network (DQN) is a multi-layered neural network that for a given state s outputs a vector of action values Q(s, · ; θ), where θ are the parameters of the network. For an n-dimensional state space and an action space containing m actions, the neural network is a function from R<sup>n</sup> to R<sup>m</sup>.[^1] In DQN there is an online and target network. The target network, has parameters θ<sup>−</sup>, and they are the same as the online network except that its parameters are copied every τ steps from the
online network, and kept fixed on all other steps.[^1] The target value used by DQN is:

![]()[^1]

However, research scientist/professor/genius Hado van Hasselt found that the max operator in the above formula led to overestimation.  

> Namely, the max operator in standard Q-learning and DQN, uses the same values both to select and to evaluate an action. This makes it more likely to select overestimated values, resulting in overoptimistic value estimates.[^1] Thus, to lower this overestimation in Double Q-learning the max operation is decomposed in the target value into an action selection and an action evaluation. Although not fully decoupled, the target network in the DQN architecture provides a natural candidate for the second value function, without
having to introduce additional networks. We therefore propose to evaluate the greedy policy according to the online
network, but using the target network to estimate its value.[^1] Hence, the target value becomes the following:

![](https://github.com/aCStandke/ReinforcementLearning/blob/main/download.png)[^1]

## Implementation
The weights of the second network contain the weights of the target network Θₜ- for the evaluation of the current greedy policy. The update to the target network stays unchanged from DQN, and remains a periodic copy of the online network[2].Two values are involved in learning: TD Estimate - the predicted optimal 𝑸* for a given state s:

![](https://github.com/aCStandke/ReinforcementLearning/blob/main/download%20(1).png)[^2]

And TD Target - aggregation of current reward and the estimated 𝑸* in the next state s':

![](https://github.com/aCStandke/ReinforcementLearning/blob/main/download%20(2).png)[^2]

![](https://github.com/aCStandke/ReinforcementLearning/blob/main/download%20(3).png)[^2]

The algorithm also samples inputs from a replay buffer, to compute TDₜ and TDₑ and backpropagate this loss down Qₒₗᵢₙₑ to update its parameters θₒₗᵢₙₑ (α is the learning rate) as follows:

![](https://github.com/aCStandke/ReinforcementLearning/blob/main/download%20(4).png)[^2]

Target weights do not get updated during backpropogation, instead rather weights from the online network are assigned to the weights of the target network as follows:

![](https://github.com/aCStandke/ReinforcementLearning/blob/main/download%20(5).png)[^2]

## Example 1: AI-powered Mario Agent using the Double Deep Q Network (DDQN) Algorithm 
The results of training the model for 10000 episodes can be found in the code section of: [Train Mario 10000](https://github.com/aCStandke/ReinforcementLearning/blob/main/Training_Mario_10000Episodes.ipynb) After 10000 episodes, an average score of 334.503 was achieved (which is well below the scores documented by others using right movements for Mario). I decided to  warm start  the model for another 100 episodes using the same memory max length and batch size of 32. Only parameters relating to: 1) the experiences added to the cache before training; 2) the number of experiences added to the cache before updating the weights of the online network and 3)`the frequency of synchronizing the model weights of the target network were changed. The values and code can be found at [Warm-Start 100](https://github.com/aCStandke/ReinforcementLearning/blob/main/DDQN_Algorithm.ipynb). Click the bottom image, if you dare, to see the DDQN in action in the Super Mario environment!!  

[![CLICK HERE](https://github.com/aCStandke/ReinforcementLearning/blob/main/mario.png)](https://youtu.be/iucn3RA2bWc)

------------------------------------------------------------------------------------------------------------------------------
# (Basic 😄) Theory of Actor-Critic (A2C) Algorithm:
All Actor-Critic Algorithms consist of two methods: an Actor/Policy and a Critic/Value. As detailed by [Actor-Critic Algorithms](https://proceedings.neurips.cc/paper/1999/file/6449f44a102fde848669bdd9eb6b76fa-Paper.pdf)

> Actor-Critic methods aim at combining the strong points of Actor/Policy-only and Critic/Value-only methods. The Critic uses an approximation architecture and simulation to learn a Value function, which is then used to update the Actor's Policy parameters in a direction of performance improvement. Such methods, as long as they are gradient-based, may have desirable convergence properties, in contrast to Critic-only methods for which convergence is guaranteed in very limited settings. They hold the promise of delivering faster convergence (due to variance reduction), when compared to Actor-only methods.[^3]

Since the Actor-Critic methods rely on an Actor/Policy and a Critic/Value, you need to select both an Actor/Policy and a Critic/Value method. In this case,the Actor/Policy method used was a Policy-Gradient Method called the REINFORCE algorithm and the Critic/Value method used was based on On-Policy Value learning. Then both methods are combined to create the Actor-Critic Algorithm used in the Implementation portion.   

**Actor/Policy**
> The method REINFORCE is built upon trajectories instead of episodes because maximizing expected return over trajectories (instead of episodes) lets the method search for optimal policies for both episodic and continuing tasks.The method samples trajectories using the policy and then uses those trajectories only to estimate the gradient.[^5] The pseudocode for this method is the following:
 
![](https://github.com/aCStandke/ReinforcementLearning/blob/main/REINFORCE.png)[^5]

**Critic/Value**
> An On-Policy Value Function gives the expected return if you start in a given state s<sub>t</sub> and always act according to the policy π and is represented by the following formula:  

![](https://github.com/aCStandke/ReinforcementLearning/blob/main/OnPolicyValueFunc.png)[^4]

**Actor-Critic-Combination**
> By viewing the problem as a stochastic gradient problem on the Actor/Policy parameter space represented by line (4) in the above pseudocode,the job of the critic is to compute an approximation of the projection of the policy π onto the subspace Ψ paramerterized by θ [which leads to the two actor-critic algorithms described in the paper][^3].The actor uses this approximation to update its policy in an approximate gradient direction.[^3] This is done by using the critic as a state-dependent baseline. To do so, an *Advantage Function* is calulated by subtracting the expected return(i.e. G) by the estimated value(i.e.V) and the following gradient is computed: 

![](https://github.com/aCStandke/ReinforcementLearning/blob/main/gradientPwCasB.png)[^4]

## Implementation:
The implementation of the above Actor-Critic Combination was taken directly from [^6]. To model functions π and V one shared neural network was used with one hidden layer of 128 units, detailed by the following code snippet: 
```
class ActorCritic(tf.keras.Model):
  def __init__(self, num_actions: int, num_hidden_units: int):
    super().__init__()
    self.common = layers.Dense(num_hidden_units, activation="relu")
    self.actor = layers.Dense(num_actions)
    self.critic = layers.Dense(1)

  def call(self, inputs: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
    x = self.common(inputs)
    return self.actor(x), self.critic(x)
```
The to train this Actor-Critic Model the gradient of the loss function has to be calculated and backpropogated. 

**Actor Loss**

![](https://github.com/aCStandke/ReinforcementLearning/blob/main/actor_loss.png)[^6]

where G-V is the [Advantage Function](https://spinningup.openai.com/en/latest/spinningup/rl_intro.html#advantage-functions). 

**Critic Loss**

![](https://github.com/aCStandke/ReinforcementLearning/blob/main/HuberLoss.png)[^6]

where L<sub>𝛿</sub> is the [Huber Loss](https://en.wikipedia.org/wiki/Huber_loss). 

Each loss is calculated independently and then combined to get the total loss. The following code snippet details this combination:

```
huber_loss = tf.keras.losses.Huber(reduction=tf.keras.losses.Reduction.SUM)

def compute_loss(action_probs: tf.Tensor, values: tf.Tensor,  returns: tf.Tensor):
  """Computes the combined actor-critic loss."""

  advantage = returns - values

  action_log_probs = tf.math.log(action_probs)
  actor_loss = -tf.math.reduce_sum(action_log_probs * advantage)

  critic_loss = huber_loss(values, returns)

  return actor_loss + critic_loss
```

## Example 2: AI-powered Discrete Lunar Lander Agent using the Actor-Critic (A2C) Algorithm
I decided to test out the Implementaion of the A2C Algorithm on the Lunar Lander enviornment as found here [Lunar Lander](https://www.gymlibrary.ml/environments/box2d/lunar_lander/). This enviornment has two possible action spaces to choose from. One is continous and another one is discrete. I choose the discrete enironment this time, since it was very easy to set up.[^7] As detailed in the Lunar Lander documentation the action, state, reward space were the following:
> There are four discrete actions available: do nothing, fire left orientation engine, fire main engine, fire right orientation engine. There are 8 states: the coordinates of the lander in x & y, its linear velocities in x & y, its angle, its angular velocity, and two booleans that represent whether each leg is in contact with the ground or not.The landing pad is always at coordinates (0,0). The coordinates are the first two numbers in the state vector.Rewards for moving from the top of the screen to the landing pad and zero speed are about 100 to 140 points.If the lander moves away from the landing pad it loses reward. The episode finishes if the lander crashes or comes to rest, receiving an additional -100 or +100 points. Each leg with ground contact is +10 points.Firing the main engine is -0.3 points each frame. Firing the side engine is -0.03 points each frame.Solved is 200 points.

I used all the default parameters from the tutorial and ended on the 10000-th episode with an episode reward of 221 (i.e. 🎮 numbers! Take that 🧑‍🚀 camp!) and an average reward of 115.82. Und das war alles. 

[![CLICK HERE for Code](https://github.com/aCStandke/ReinforcementLearning/blob/main/landingMoon_no_wind.gif)](https://github.com/aCStandke/ReinforcementLearning/blob/main/ActorCriticAlgorithm.ipynb)

## Reference:
[^1]: [Deep Reinforcement Learning with Double Q-learning](https://arxiv.org/pdf/1509.06461.pdf)
[^2]: [Train a Mario-Playing Agent](https://pytorch.org/tutorials/intermediate/mario_rl_tutorial.html)
[^3]: [Actor-Critic Algorithms](https://proceedings.neurips.cc/paper/1999/file/6449f44a102fde848669bdd9eb6b76fa-Paper.pdf)
[^4]: [Actor-Critic Algorithms Slides](http://rail.eecs.berkeley.edu/deeprlcourse-fa17/f17docs/lecture_5_actor_critic_pdf)
[^5]: [Policy-Gradient Methods: REINFORCE Algorithm](https://towardsdatascience.com/policy-gradient-methods-104c783251e0#:~:text=The%20method%20REINFORCE%20is%20built,both%20episodic%20and%20continuing%20tasks.)
[^6]: [Playing CartPole with the Actor-Critic Method](https://www.tensorflow.org/tutorials/reinforcement_learning/actor_critic)
[^7]: I tried at first the continous environment, but the interface between numpy and tensorflow's graph was giving me some trouble when using tensorflow's wrapper [tf.numpy_function](https://www.tensorflow.org/api_docs/python/tf/numpy_function)   
