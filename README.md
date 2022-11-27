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

## Example 3: Discrete Action Space for Stock/ETF Trading Agent: Part I  

In chapter 8 of [^9], the author details a practical domain problem of reinforcement learning; namely stock trading! The first thing that had to be done was create the stock trading environment. The type of *actions* the agent could take in this environment were the following: 1) hold-off on trading;  2) buy the given stock; and 3) sell the given stock. When the agent bought or sold the given stock, it had to pay a commission of 0.1%. The agent's *state-space* consisted of the following items: 1) 5 past trading days of opening price data; 2) 5 past trading days of high, low, and closing prices in relation to the opening price; 3) volume for the current trading day; 4) whether the agent bought stock and 5) the relative close of the stock for the current trading day. The *reward* for the agent is a two reward scheme, as detailed in [^9]. Namely the reward is of "either/or form"; if the variable reward_on_close is True, the agent will receive a reward only on selling its stock position, else the agent will recive a reward only when buying and holding its stock position (i.e. not selling). The first form amounts to the trading strategy of [active investing](https://www.investopedia.com/terms/a/activeinvesting.asp#:~:text=Active%20investing%20refers%20to%20an,activity%20to%20exploit%20profitable%20conditions.), while the second form amounts to the trading strategy of [passive investing](https://www.investopedia.com/terms/p/passiveinvesting.asp#:~:text=Passive%20investing's%20goal%20is%20to,price%20fluctuations%20or%20market%20timing.). 

In [^9] the author uses stock data from the Russian stock market from the period ranging from 2015-2016 for the technology company [Yandex](https://en.wikipedia.org/wiki/Yandex). While the dataset contained over 130,000  rows of data, in which every row represented a single minute of price data, I decided to take a more general approach and chose for the agent to trade using the [SPY ETF](https://www.etf.com/SPY#:~:text=SPY%20is%20the%20best%2Drecognized,US%20index%2C%20the%20S%26P%20500.). Each row in the dataset represented one trading day of the etf, and ranged from 2005 to 2022. The years 2005-2014 was used for training and the years 2015-2020 was used for validation.The Source Code for the SPY Trading agent can be found here: [SpyTradingAgent](https://github.com/aCStandke/ReinforcementLearning/blob/main/SpyTradingAgent.ipynb) 

### Passive Investing Results:
The Y-axis is given in percentages and the X-axis is given by the number of steps executed. 

**Mean Value Reward per 100 Trading Windows**
![](https://github.com/aCStandke/ReinforcementLearning/blob/main/steps_per_100_tradingWindow_buy.png)

### Active Investing Results:
The Y-axis is given in percentages and the X-axis is given by the number of steps executed. 

**Mean Value Reward per 100 Trading Windows**
![](https://github.com/aCStandke/ReinforcementLearning/blob/main/reward_per_tradingWindow_sell.png)

------------------------------------------------------------------------------------------------------------------------------
# Basic Theory of Proximal Policy Optimization(PPO)

## Proximal Policy Optimization

We propose a new family of policy gradient methods for reinforcement learning, which alternate between sampling data through interaction with the environment, and optimizing a
“surrogate” objective function using stochastic gradient ascent. Whereas standard policy gradient methods perform one gradient update per data sample, we propose a novel objective
function that enables multiple epochs of minibatch updates. The new methods, which we call
proximal policy optimization (PPO), have some of the benefits of trust region policy optimization (TRPO), but they are much simpler to implement, more general, and have better sample
complexity (empirically). Our experiments test PPO on a collection of benchmark tasks, including simulated robotic locomotion and Atari game playing, and we show that PPO outperforms
other online policy gradient methods, and overall strikes a favorable balance between sample
complexity, simplicity, and wall-time.[^10]

**CPI Loss**
![](https://github.com/aCStandke/ReinforcementLearning/blob/main/Screenshot%202022-09-14%2011.56.30%20PM.png)

**CLIP Loss**
![](https://github.com/aCStandke/ReinforcementLearning/blob/main/Screenshot%202022-09-14%2011.59.27%20PM.png)

**TOTAL Loss**
![](https://github.com/aCStandke/ReinforcementLearning/blob/main/Screenshot%202022-09-14%2011.54.58%20PM.png)

**Algorithm**

![](https://github.com/aCStandke/ReinforcementLearning/blob/main/Screenshot%202022-09-14%2011.54.10%20PM.png)

## Implementation:

To train the Trading Agent the package [Stable-Baselines3](https://stable-baselines3.readthedocs.io/en/master/index.html) was used. As stated in the docs: 

> Stable Baselines3 (SB3) is a set of reliable implementations of reinforcement learning algorithms in PyTorch. It is the next major version of Stable Baselines. And steems from the paper [Stable-Baselines3: Reliable Reinforcement Learning Implementations](https://jmlr.org/papers/volume22/20-1364/20-1364.pdf)
The algorithms in this package will make it easier for the research community and industry to replicate, refine, and identify new ideas, and will create good baselines to build projects on top of. We expect these tools will be used as a base around which new ideas can be added, and as a tool for comparing a new approach against existing ones. We also hope that the simplicity of these tools will allow beginners to experiment with a more advanced toolset, without being buried in implementation details.[^11]

Furthermore to visualize the trading agent's observation space when trading, I used Adam King's brilliant implementation of a stock trading environment as found detailed here [Rendering elegant stock trading agents using Matplotlib and Gym](https://towardsdatascience.com/visualizing-stock-trading-agents-using-matplotlib-and-gym-584c992bc6d4).

## Example 4: Continuous Action Space for Stock/ETF Trading Agent: Part II (:warning::warning::warning: :warning: WARNING!!!!!! Realize this environment is not realistic!!!! WARNING!!! WARNING!!!, Again WARNING!!!, OKAY, Time for the MIT software disclaimer, READ IT: :point_down:
THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE. 

This second stock/etf environment is based on Adam King's article as found here:[Create custom gym environments from scratch — A stock market example](https://towardsdatascience.com/creating-a-custom-openai-gym-environment-for-stock-trading-be532be3910e). Similar to the first stock trading environment based on Maxim Lapan's implementation as found in chapter eight of his book [Deep Reinforcement Learning Hands-On: Apply modern RL methods to practical problems of chatbots, robotics, discrete optimization, web automation, and more, 2nd Edition](https://www.amazon.com/Deep-Reinforcement-Learning-Hands-optimization/dp/1838826998) and as  implemented above in Example 3, the agent is trading in the environment of the [SPY ETF](https://www.etf.com/SPY?L=1) except in this trading environment the agent is taking continuous actions, rather than discrete actions and is tasked with managing a [trading account](https://www.investopedia.com/terms/t/tradingaccount.asp#:~:text=A%20trading%20account%20is%20an,margin%20requirements%20set%20by%20FINRA.).

In this trading environment, the agent's reward is based on managing its trading account/balance. The agent can take two actions: 1) either buying or selling the SPY ETF[^12] and 2) what percentage of the SPY ETF to buy or sell, which ranges from [0,1] (i.e. 0% to 100%).The agent's *state/observation-space* consists of the following items: 1) 5 past trading days of high, low, and closing price data in relation to the opening price; 2) volume for the current trading day; 3) the Agent's trading account/balance; 4) the number of shares held; 5) the number of shares sold; 6) the Agent's net worth which consists of the agent's account balance and the shares current price value; 7) the value of the shares sold; and 8) the cost basis of buying new shares as compared to buying shares in the previous period. All of these items were normalized to be in the interval of [-1,1]. The trading agent begins with 10,000 US dollars to trade with and can accumulate a max trading account/balence of 2,147,483,647 US dollars.


### Trading Results

The PPO Agent was trained for 50 thousand steps of SPY data ranging from 2005 to mid-2017 and was tested on SPY data ranging from the end-of-2017 to 2022. Furthermore, 4 parallel environments were used for the agent to gather samples/experience to train on. All of the default hyper-parameters from stable-Baselines3 were used except for the number of epochs when optimizing the surrogate loss which I set to 20. I also set the entropy coefficient to 0.01 and used stable-Baselines3's limit implementation regarding the KL divergence between updates and set it to 0.2. Unfotunely, I did not keep track of the learning cures and/or other important metrics(i.e. total loss, value loss etc) that should be tracked during training. The video down below shows this under-trained trading agent in action :point_down:

[![PPO Trading Agent](https://github.com/aCStandke/ReinforcementLearning/blob/main/16ktradingsceme.png)](https://youtu.be/QBWMEu9GrHE)
 
| Top Subplot Legend | |
| ------------- | ------------- |
| ![](https://github.com/aCStandke/ReinforcementLearning/blob/main/balance%20line%20plot.png) | Agent's trading balance/account  |

| Bottom Subplot Legend  | |
| ------------- | ------------- |
| ![](https://github.com/aCStandke/ReinforcementLearning/blob/main/Trade%20done%20by%20agent.png) | Trade executed by agent, colored either green or red, depending on agent buying or selling shares, and annotated with the total amount transacted|
| ![](https://github.com/aCStandke/ReinforcementLearning/blob/main/volume.png) | Volume for the trading day colored either green or red, depending on whether the price moved up or down  |
| ![](https://github.com/aCStandke/ReinforcementLearning/blob/main/candlestick%20.png) | OHCL data in candlestick form colored either red or green, depending on whether the stock/etf closed lower or higher than its open |

The Source Code for the Second Trading agent can be found here: [Second Spy Trading Agent](https://github.com/aCStandke/ReinforcementLearning/blob/main/SecondStockEnivornment.ipynb). The SPY data that the Trading agent used for training data can be found here: [SPY_train](https://github.com/aCStandke/ReinforcementLearning/blob/main/spy.us.txt).And the SPY data that the  Trading agent used for testing data can be found here: [SPY_test](https://github.com/aCStandke/ReinforcementLearning/blob/main/test.csv)

## Example 5: Multi-Discrete Action Space for Stock/ETF Trading Agent: Part III (:warning::warning::warning: :warning: WARNING!!!!!! Realize this environment is not realistic!!!! WARNING!!! WARNING!!!, Again WARNING!!!, OKAY, Time for the MIT software disclaimer, READ IT: :point_down:
THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE. 

This third stock trading environment is based on Adam King's articles as found here:[Creating Bitcoin trading bots don’t lose money](https://medium.com/towards-data-science/creating-bitcoin-trading-bots-that-dont-lose-money-2e7165fb0b29) and here:[Optimizing deep learning trading bots using state-of-the-art techniques](https://towardsdatascience.com/using-reinforcement-learning-to-trade-bitcoin-for-massive-profit-b69d0e8f583b)
Furthermore, the random offset in the reset method and the if/and control flow is based on Maxim Lapan's implementation as found in chapter eight of his book [Deep Reinforcement Learning Hands-On: Apply modern RL methods to practical problems of chatbots, robotics, discrete optimization, web automation, and more, 2nd Edition](https://www.amazon.com/Deep-Reinforcement-Learning-Hands-optimization/dp/1838826998).

Similar to the first and second stock trading environment, the agent is trading in the [SPY ETF](https://www.etf.com/SPY?L=1) environment; trading is in a Multi-Discrete action space of [3, 10] where 0==buy, 1==sell, and 2==hold and the values [0, .1, .2, .3, .4, .5, .6,.7,.8,.9,] represent the number of shares held/sold by the agent ie. 10%, 20%, etc.; and the observation space is a continious observation space from [-inf,inf])(*note: however, in the second stock trading environment this space ranged from [-1,1]*).Also unlike the second stock trading environment, an additional observation was added to the agent's observations space of an account history/ledger of the agent's past networth, balance, and shares from trading (*note: this window is set by the variable LOOKBACK_WINDOW_SIZE and its default is 30 days*). And  to make it semi-realistic, a commision parameter is used in the cost and sales calculation (*note: default is 0.1%*). 

Additionally, three different ways of calculating the agent's reward were added, namely: 
* [sortinoRewardRatio](https://www.investopedia.com/terms/s/sortinoratio.asp) $\frac{R_p-r_f}{\sigma_d}$ where $R_p$ is actual or expected portfolio return, $r_f$ is the risk free rate  and ${sigma_d}$ is the std of the downside
* [omegaRewardRatio](https://www.wallstreetmojo.com/omega-ratio/) $\frac{\int_{\theta}^{inf}1-F(R_p)dx}{\int_{-inf}^{\theta}F(R_p)dx}$ where $F$ is the cumulative probability distribution of returns, and ${\theta}$ is the target return threshold defining what is considered a gain versus a loss
* [Excess Return](https://ai.stackexchange.com/questions/10082/suitable-reward-function-for-trading-buy-and-sell-orders) This is a reward rule that was created by Franklin Allen and Risto Karjalainen in their paper [Using genetic algorithms to find technical trading rules](https://www.cs.montana.edu/courses/spring2007/536/materials/Lopez/genetic.pdf). This is the defualt reward function for the third trading environment. 

### Trading using StableBaseline3's MlpPolicy 

![](https://github.com/aCStandke/ReinforcementLearning/blob/main/Mlprf.png)
 
The Source Code for the Third Trading agent can be found here:[Third Spy Trading Agent](https://github.com/aCStandke/ReinforcementLearning/blob/main/modelTesting.ipynb)

------------------------------------------------------------------------------------------------------------------------------
# Robotic Control Algorithms:

## D4PG

The D4PG reinforcement algorithm is an algorithm that enhances DDPG by making the update of the critic distributional. Namely as the authors of [^14] detail:

![](https://github.com/aCStandke/ReinforcementLearning/blob/main/enhancements_to_DDPG.png)[^14]

To do this,they manapulate the traditional state-action value fuction by parameterizing the policy from $\pi$:

![](https://github.com/aCStandke/ReinforcementLearning/blob/main/traditional_stateValueFunction.png)[^14]

to $\pi_{\theta}$:

![](https://github.com/aCStandke/ReinforcementLearning/blob/main/paramartized_stateAction_DPGgradient.png)[^14]


## Implementation:







**Continious A2C**

[![Watch the video](https://github.com/aCStandke/ReinforcementLearning/blob/main/mq1.png)](https://youtu.be/PIgWhBXI7Ks)

**DDPG**

[![Watch the video](https://github.com/aCStandke/ReinforcementLearning/blob/main/mq2.png)](https://youtu.be/2SSgQbgGD_0)



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
[^10]:[PPO](https://arxiv.org/pdf/1707.06347.pdf)
[^11]:[Stable-Baselines3](https://stable-baselines3.readthedocs.io/en/master/index.html)
[^12]: Note: eventhough this decision is discrete in nature, it is being modeled as a continous action by making values less than 0 as a buy action and values greater than or equal to 0 as a sell action
[^13]: [Create custom gym environments from scratch — A stock market example](https://towardsdatascience.com/creating-a-custom-openai-gym-environment-for-stock-trading-be532be3910e)
[^14]:[D4PG](https://arxiv.org/pdf/1804.08617.pdf)
