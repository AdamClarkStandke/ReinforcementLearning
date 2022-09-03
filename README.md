# Theory of DQN and the Double DQN Learning Algorithm:

As detailed by [Deep Reinforcement Learning with Double Q-learning](https://arxiv.org/pdf/1509.06461.pdf) 

> A deep Q network (DQN) is a multi-layered neural network that for a given state s outputs a vector of action values Q(s, · ; θ), where θ are the parameters of the network. For an n-dimensional state space and an action space containing m actions, the neural network is a function from R<sup>n</sup> to R<sup>m</sup>.[^1] In DQN there is an online and target network. The target network, has parameters θ<sup>−</sup>, and they are the same as the online network except that its parameters are copied every τ steps from the
online network, and kept fixed on all other steps.[^1] The target value used by DQN is:

![](https://github.com/aCStandke/ReinforcementLearning/blob/main/Screenshot%202022-08-04%2011.38.00%20AM.png)[^1]

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

> The method REINFORCE is built upon trajectories instead of episodes because maximizing expected return over trajectories (instead of episodes) lets the method search for optimal policies for both episodic and continuing tasks.The method samples trajectories using the policy and then uses those trajectories only to estimate the gradient.[^5] The pseudocode for the REINFORCE algorithm is the following:
 
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

I used all the default parameters from the tutorial and ended on the 10000-th episode with an episode reward of 221 (i.e. 🎮 numbers! Take that 🧑‍🚀 camp!) and an average reward of 115.82. Und das war alles fur heute. 

[![CLICK HERE for Code](https://github.com/aCStandke/ReinforcementLearning/blob/main/landingMoon_no_wind.gif)](https://github.com/aCStandke/ReinforcementLearning/blob/main/ActorCriticAlgorithm.ipynb)

------------------------------------------------------------------------------------------------------------------------------
# Theory of Dueling Network Architecture:

As detailed by [Dueling Network Architectures for Deep Reinforcement Learning](https://arxiv.org/pdf/1511.06581.pdf)

>  In this paper, we present a new neural network architecture for model-free reinforcement learning. Our
dueling network represents two separate estimators: one for the state value function and one for
the state-dependent action advantage function.The main benefit of this factoring is to generalize learning across actions without imposing any
change to the underlying reinforcement learning  algorithm.[^8]

> To bring this insight to fruition, we design a single Qnetwork architecture, as illustrated in Figure 1, which we
refer to as the dueling network. The lower layers of the dueling network are convolutional as in the original DQNs
(Mnih et al., 2015). However, instead of following the convolutional layers with a single sequence of fully connected
layers, we instead use two sequences (or streams) of fully connected layers. The streams are constructed such that
they have they have the capability of providing separate estimates of the value and advantage functions. Finally, the
two streams are combined to produce a single output Q function. As in (Mnih et al., 2015), the output of the network is a set of Q values, one for each action.[^8].

![](https://github.com/aCStandke/ReinforcementLearning/blob/main/duel_network.png.png)[^8].

The forumula that combines these two streams to output Q-values is the following: 

![](https://github.com/aCStandke/ReinforcementLearning/blob/main/dueling_architecture_formula.png.png)[^8].

where V represents the value function and A represents the advantages function for each action.

## Implementation

The implementation of the above formula came from the book [Deep Reinforcement Learning Hands-On: Apply modern RL methods to practical problems of chatbots, robotics, discrete optimization, web automation, and more, 2nd Edition](https://www.amazon.com/Deep-Reinforcement-Learning-Hands-optimization/dp/1838826998/ref=asc_df_1838826998/?tag=hyprod-20&linkCode=df0&hvadid=416741343328&hvpos=&hvnetw=g&hvrand=7234438034400691228&hvpone=&hvptwo=&hvqmt=&hvdev=c&hvdvcmdl=&hvlocint=&hvlocphy=9008183&hvtargid=pla-871456510229&psc=1&tag=&ref=&adgrpid=93867144477&hvpone=&hvptwo=&hvadid=416741343328&hvpos=&hvnetw=g&hvrand=7234438034400691228&hvqmt=&hvdev=c&hvdvcmdl=&hvlocint=&hvlocphy=9008183&hvtargid=pla-871456510229) by Maxim Lapan. And is shown by the following PyTorch code snippet:

```
def __init__(self, shape, actions_n):
        self.conv = nn.Sequential(
            nn.Conv1d(shape[0], 128, kernel),
            nn.ReLU(),
            nn.Conv1d(128, 128, kernel),
            nn.ReLU(),
        )
        o =  self.conv(torch.zeros(1, *shape))
        out_size =i nt(np.prod(o.size()))
        self.fc_val = nn.Sequential(
            nn.Linear(out_size, 512),
            nn.ReLU(),
            nn.Linear(512, 1)
        )
        self.fc_adv = nn.Sequential(
            nn.Linear(out_size, 512),
            nn.ReLU(),
            nn.Linear(512, actions_n)
        )
    def forward(self, x):
        conv_out = self.conv(x).view(x.size()[0], -1)
        val = self.fc_val(conv_out)
        adv = self.fc_adv(conv_out)
        return val + adv - adv.mean(dim=1, keepdim=True)
```

## Example 3: Stock Trading Agent using the Dueling Network Architecture 

In chapter 8 of [^9], the author details a practical domain problem of reinforcement learning; namely stock trading! The first thing that had to be done was create the stock trading environment. The type of *actions* the agent could take in this environment were the following: 1) hold-off on trading;  2) buy the given stock; and 3) sell the given stock. When the agent bought or sold the given stock, it had to pay a commission of 0.1%. The agent's *state-space* consisted of the following items: 1) 5 past trading days of opening price data; 2) 5 past trading days of high, low, and closing prices in relation to the opening price; 3) volume for the current trading day; 4) whether the agent bought stock and 5) the relative close of the stock for the current trading day. The *reward* for the agent is a two reward scheme, as detailed in [^9]. Namely the reward is of "either/or form"; if the variable reward_on_close is True, the agent will receive a reward only on selling its stock position, else the agent will recive a reward only when buying and holding its stock position (i.e. not selling). The first form amounts to the trading strategy of [active investing](https://www.investopedia.com/terms/a/activeinvesting.asp#:~:text=Active%20investing%20refers%20to%20an,activity%20to%20exploit%20profitable%20conditions.), while the second form amounts to the trading strategy of [passive investing](https://www.investopedia.com/terms/p/passiveinvesting.asp#:~:text=Passive%20investing's%20goal%20is%20to,price%20fluctuations%20or%20market%20timing.). 

In [^9] the author uses stock data from the Russian stock market from the period ranging from 2015-2016 for the technology company [Yandex](https://en.wikipedia.org/wiki/Yandex). While the dataset contained over 130,000  rows of data, in which every row represented a single minute of price data, I decided to take a more longer term approach and chose for the agent to trade using the [SPY ETF](https://www.etf.com/SPY#:~:text=SPY%20is%20the%20best%2Drecognized,US%20index%2C%20the%20S%26P%20500.). Each row in the dataset represented one trading day of the etf, and ranged from 2005 to 2022. The years 2005-2014 was used for training and the years 2015-2020 was used for validation.The Source Code for the SPY Trading agent can be found here: [SpyTradingAgent](https://github.com/aCStandke/ReinforcementLearning/blob/main/SpyTradingAgent.ipynb) 

### Passive Investing Results:
The Y-axis is given in percentages and the X-axis is given by the number of steps executed. 

**Mean Value Reward**
![](https://github.com/aCStandke/ReinforcementLearning/blob/main/values_mean_buy.png)

**Mean Value Reward per 100 Trading Windows**
![](https://github.com/aCStandke/ReinforcementLearning/blob/main/steps_per_100_tradingWindow_buy.png)

**Validation: Mean Value Reward per Episode**
![](https://github.com/aCStandke/ReinforcementLearning/blob/main/episode_reward_val_buy.png)

**Test: Mean Value Reward per Episode**
![](https://github.com/aCStandke/ReinforcementLearning/blob/main/episode_reward_test_buy.png)

### Active Investing Results:
The Y-axis is given in percentages and the X-axis is given by the number of steps executed. 

**Mean Value Reward**
![](https://github.com/aCStandke/ReinforcementLearning/blob/main/values_mean_sell.png)

**Mean Value Reward per 100 Trading Windows**
![](https://github.com/aCStandke/ReinforcementLearning/blob/main/reward_per_tradingWindow_sell.png)

**Validation: Mean Value Reward per Episode**
![](https://github.com/aCStandke/ReinforcementLearning/blob/main/episode_reward_val_sell.png)

**Test: Mean Value Reward per Episode**
![](https://github.com/aCStandke/ReinforcementLearning/blob/main/sell_episode_reward_test.png)

------------------------------------------------------------------------------------------------------------------------------
# Basic Theory of Deep Deterministic Policy Gradient (DDPG) and Soft Actor Critic (SAC)

## Deep Deterministic Policy Gradient (DDPG)
As stated by the authors of [CONTINUOUS CONTROL WITH DEEP REINFORCEMENT
LEARNING](https://arxiv.org/pdf/1509.02971.pdf):

> While DQN solves problems with high-dimensional observation spaces, it can only handle discrete and low-dimensional action spaces. Many tasks of interest, most notably physical control tasks, have continuous (real valued) and high dimensional action spaces. DQN cannot be straightforwardly applied to continuous domains since it relies on a finding the action that maximizes the action-value function, which in the continuous valued case requires an iterative optimization process at every step. In this work we present a model-free, off-policy actor-critic algorithm using deep function approximators that can learn policies in high-dimensional, continuous action spaces. Here we combine the actor-critic approach with insights from the recent success of Deep Q Network [DQN](https://arxiv.org/pdf/1312.5602.pdf)[^10]

**The key take aways from the algorithm are the following:** 

1.   It uses an actor-critic approach based on the [DPG algorithm](http://proceedings.mlr.press/v32/silver14.pdf) 
2.   A replay buffer can be used to store transitions for sampling, since it is an off-policy algorithm
3.   A copy of the actor and critic networks, $µ_{\theta}(s|θ)$ and $Q_{\phi}(s,a|{\phi})$ respectively, are used for calculating the target values. The weights of these target networks are then updated by having them slowly track the learned networks
4.   Batch normalization can be used 
5.   An exploration policy is used by adding noise sampled from a noise process N to the actor policy  $µ_{\theta}(s_t)= µ(s_t |{\theta}, µ_t) + N$ 

*N* can be chosen to suit the environment, I used Ornstein Uhlenbeck Noise as provided by Stable-Baselines.The pseudocode for the DDPG algorithm is the following:

![](https://github.com/aCStandke/ReinforcementLearning/blob/main/DDPG%20algorithm.png)[^10]

## Soft Actor Critic (SAC)
As stated by the authors of [Soft Actor-Critic:
Off-Policy Maximum Entropy Deep Reinforcement
Learning with a Stochastic Actor](https://arxiv.org/pdf/1801.01290.pdf):

> In this framework, the actor aims to maximize expected reward while also maximizing entropy. That is, to succeed at the task while acting as randomly as possible.[^11]

**The key take aways from the algorithm are the following:** 

1.   Uses a maximum entropy objective rather than  the standard maximum
expected reward objective 
2.   Uses a soft policy iteration, which is a general algorithm for learning optimal maximum entropy policies that alternate between policy evaluation and policy improvement 
3.   SAC concurrently learns a policy  $\pi_{\theta}$ and two Q-functions  $Q_{\phi_1}$,  $Q_{\phi_2}$

The pseudocode for the SAC algorithm is the following:

![]()[^12]

## Implementation:

## Example 4: Continuous Stock/ETF Trading Agent 

[![CLICK HERE](https://github.com/aCStandke/ReinforcementLearning/blob/main/agentTradingscreen.png)](https://youtu.be/jKH295P-r-8)




The Source Code for the Second SPY Trading agent can be found here: [Second Spy Trading Agent](https://github.com/aCStandke/ReinforcementLearning/blob/main/SecondStockEnivornment.ipynb)

The SPY data that the Second SPY Trading agent operated in can be found here: [SPY](https://github.com/aCStandke/ReinforcementLearning/blob/main/spy.us.txt) 

------------------------------------------------------------------------------------------------------------------------------


## Reference:
[^1]: [Deep Reinforcement Learning with Double Q-learning](https://arxiv.org/pdf/1509.06461.pdf)
[^2]: [Train a Mario-Playing Agent](https://pytorch.org/tutorials/intermediate/mario_rl_tutorial.html)
[^3]: [Actor-Critic Algorithms](https://proceedings.neurips.cc/paper/1999/file/6449f44a102fde848669bdd9eb6b76fa-Paper.pdf)
[^4]: [Actor-Critic Algorithms Slides](http://rail.eecs.berkeley.edu/deeprlcourse-fa17/f17docs/lecture_5_actor_critic_pdf)
[^5]: [Policy-Gradient Methods: REINFORCE Algorithm](https://towardsdatascience.com/policy-gradient-methods-104c783251e0#:~:text=The%20method%20REINFORCE%20is%20built,both%20episodic%20and%20continuing%20tasks.)
[^6]: [Playing CartPole with the Actor-Critic Method](https://www.tensorflow.org/tutorials/reinforcement_learning/actor_critic)
[^7]: I tried at first the continous environment, but the interface between numpy and tensorflow's graph was giving me some trouble when using tensorflow's wrapper [tf.numpy_function](https://www.tensorflow.org/api_docs/python/tf/numpy_function)   
[^8]: [Dueling Network Architectures for Deep Reinforcement Learning](https://arxiv.org/pdf/1511.06581.pdf)
[^9]: [Deep Reinforcement Learning Hands-On: Apply modern RL methods to practical problems of chatbots, robotics, discrete optimization, web automation, and more, 2nd Edition](https://www.amazon.com/Deep-Reinforcement-Learning-Hands-optimization/dp/1838826998/ref=asc_df_1838826998/?tag=hyprod-20&linkCode=df0&hvadid=416741343328&hvpos=&hvnetw=g&hvrand=7234438034400691228&hvpone=&hvptwo=&hvqmt=&hvdev=c&hvdvcmdl=&hvlocint=&hvlocphy=9008183&hvtargid=pla-871456510229&psc=1&tag=&ref=&adgrpid=93867144477&hvpone=&hvptwo=&hvadid=416741343328&hvpos=&hvnetw=g&hvrand=7234438034400691228&hvqmt=&hvdev=c&hvdvcmdl=&hvlocint=&hvlocphy=9008183&hvtargid=pla-871456510229) 
[^10]: [CONTINUOUS CONTROL WITH DEEP REINFORCEMENT
LEARNING](https://arxiv.org/pdf/1509.02971.pdf)
[^11]:[Soft Actor-Critic:
Off-Policy Maximum Entropy Deep Reinforcement
Learning with a Stochastic Actor](https://arxiv.org/pdf/1801.01290.pdf)
[^12]:
