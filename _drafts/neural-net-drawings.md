---
layout: post
title:  "Neural Network Drawings"
---

Can you draw a penguin in less than 20 seconds?

That's what the little game
[Google Quick Draw!](https://quickdraw.withgoogle.com/)
asks you to do. And if it's not an penguin, it will be one the many
other proposed items, such as Effeil towers, cats, faces or carrots.

If like mine, your drawing is so pathetic it wouldn't compete with that
of your 6-years-old nephew which is pinned to the fridge door, then you
are in luck, because this article is about teaching a neural network
to draw for you.

![](images/pathetic_penguin.png)

If not, well, I let you be the sole judge of whether you beat the
machine, or the machine beats you. The (Pytorch) code is available
on [Github](https://github.com/quentin-auge/draw/).

![](images/generated/gmm512_20_penguin.gif)

In 2017, Google released tens of millions of drawings of various
quality from *Quick Draw!*. In my opinion, a lot of then don't beat
they machine that used them to learn drawing. The student has
surpassed the teacher.

![](images/dataset/selected_penguin.png)

<small>Note: Throughout the article, everytime I include a series
of drawings that I obviously selected carefully, I link to the larger
series they were choosen from. Feel free to click the asterisk at the
right of the drawings to see how much I'm cheating you (or not).</small>

In the dataset, drawings are represented as sequences of
points, as opposed to image pixels. Since the natural fit for modelling
sequences is the so-called "recurrent neural networks" (RNNs), we're
going to use them to draw and ... fail.

As we'll discover quickly, generic recurrent neural networks, however
powerful, are not enough to succeed in regaining control over the
fridge door. In order to capture the subtilities of human drawing,
we'll need to introduce an exciting extension of neural networks called
"mixture density networks" (MDN), that involves adding a probabilistic
layer on top of the neural network.

The article is structured as follow: we start off with simple and
potentially incomplete data representation and model, and progress naturally through failures and successes to the model outlined in
the Google Brain paper
[A Neural Representation of Sketch Drawings](https://arxiv.org/abs/1704.03477). We focus on what the paper refers to as
"unconditional generation of drawings" (put bluntly: "please draw me
a penguin"). The authors go one step further to achieve conditional
generation ("please draw a penguin *like this one*") and interpolation
("please draw a penguin that is *half like this one, and half that
one*"), but that's a subject for another article.

Some familiarity with machine learning (in particular the concept of
training supervised regression and classification models by loss
minimization) would let the reader get through more easily. We won't
get too deep into the inner workings of neural nets, as there is
already a wealth of easily available information about them, and in
any case, they can always be treated as a function consisting of a
arbitrary bunch of linearities (matrix multiplications) and
non-linearities stacked together. Perhaps more critically, being
comfortable with basic probability theory would be of great help later
in the article, but as long as you know what a gaussian looks like
(a multi-dimensional one, if possible), you're halfway there. If you
know about maximum likelihood estimation, you're done, basically.

All set? Let's dive in.

# Data

The data consist in 50 million drawings available through
[Github](https://github.com/googlecreativelab/quickdraw-dataset)
across 345 categories. Let's pick 3 of them (*Effeil tower*,
*face* and *firetruck*), and draw some of the examples that
appear in the dataset.

![](images/dataset/selected_effeil.png)
![](images/dataset/selected_face.png)
![](images/dataset/selected_firetruck.png)

All three categories provide a different set of challenges. Effeil towers
are mainly composed of straight lines with some sharp angles, while
faces are mostly smooth curves and circles. Both have a moderate number
of strokes, though it is probably more difficult to position them at
the right spot with respect to each other in order to generate decent
faces than it is for Effeil towers. Firetrucks are definitely the most
difficult to draw, combining all the previous difficulties with even
more strokes.

Drawings are presented in the dataset in their most obvious shape:
sequences of points
$\begin{bmatrix} \mathbf{x}\_{i},~\mathbf{y}\_{i} \end{bmatrix}$. But
it's not the only way to represent them. What about sequences of vectors
$\begin{bmatrix} \Delta \mathbf{x}\_{i},~\Delta \mathbf{y}\_{i} \end{bmatrix}$
from each points to the next, or — let's get crazy — sequences of
polar-coordinates vectors
$\begin{bmatrix} \mathbf{r}\_{i},~\mathbf{\theta}\_{i} \end{bmatrix}$
from one such vector to the next?

![](images/representations_plot.png)

Well, it turns out the
$\begin{bmatrix} \Delta \mathbf{x}\_{i}~\Delta \mathbf{y}\_{i} \end{bmatrix}$
representation is the most interesting for the three following reasons:
 * It is trivial and inexpensive to compute:
 $\begin{bmatrix} \Delta \mathbf{x}\_{i},~\Delta \mathbf{x}\_{i} \end{bmatrix} = \begin{bmatrix} \mathbf{x}\_{i+1} - \mathbf{y}\_{i},~\mathbf{y}\_{i+1} - \mathbf{y}\_{i} \end{bmatrix}$
 , as opposed to the $\begin{bmatrix} \mathbf{r}\_{i},~\mathbf{\theta}\_{i} \end{bmatrix}$ representation I'll spare you
 the trigonometry of.
 * It allows us to define a compelling object: a *no-displacement*
 vector $\overrightarrow 0 = \begin{bmatrix} 0,~0 \end{bmatrix}$.
 It has no equivalent in the
 $\begin{bmatrix} \mathbf{x}\_{i},~\mathbf{y}\_{i} \end{bmatrix}$
 system where $\begin{bmatrix} 0,~0 \end{bmatrix}$ is just a
 regular point. It does in the
 $\begin{bmatrix} \mathbf{r}\_{i},~\mathbf{\theta}\_{i} \end{bmatrix}$
 representation, but not without a few caveats. For instance, what
 should the angle beween $\overrightarrow 0$ and another vector be?
 * It's the only representation in which the points follow a
 distribution, which although too spread out to be gaussian gaussian,
 is at least symmetrical.

   ![](images/representations_distplot.png)

   In  order for the neural net to learn more effectively, we are going
   to standarize each point by the mean
   $\begin{bmatrix} \mathbf{\mu}\_1,~\mathbf{\mu}\_2 \end{bmatrix}$
   and standard deviation
   $\begin{bmatrix} \mathbf{\sigma}\_1,~\mathbf{\sigma}\_2 \end{bmatrix}$
   of all point in the whole dataset:
    
   $$
   \begin{bmatrix} \Delta \mathbf{x}\_{i},~\Delta \mathbf{y}\_{i} \end{bmatrix}=
   \begin{bmatrix} \frac{\Delta \mathbf{x}\_{i} - \mu_1}{\sigma_1},
   ~\frac{\Delta \mathbf{y}\_{i} - \mu_2}{\sigma_2} \end{bmatrix}
   $$

   It makes much sense when the mean and variance correspond to the
   actual center and spread of the points (symmetrical) distribution.

So we pick the
$\begin{bmatrix} \Delta \mathbf{x}\_{i},~\Delta \mathbf{y}\_{i} \end{bmatrix}$
sytem and carry on.

However, in order not to clutter all subsequent equations with
$\Delta$'s, please allow me to conveniently discard them back by setting
$\begin{bmatrix} \mathbf{x}_i,~\mathbf{y}_i \end{bmatrix} = \begin{bmatrix} \Delta \mathbf{x}_i,~\Delta \mathbf{y}_i \end{bmatrix}$
and refer to them as "point" when it's convenient.

# Drawings as trajectories

As per the previous paragraph, each drawing $\mathcal{X}$ is a sequence
of 2-dimensional displacement vectors
$\mathcal{X}_i = \begin{bmatrix} \mathbf{x}\_{i},~\mathbf{y}\_{i} \end{bmatrix}$
with its own length $N-1$, since each vector compacts every successive
points of the original drawing into one.

Our goal is to create a model that can generate such drawings.
Concretely, it means we are looking for a function $f$ that takes a
vector $\mathcal{X}_i = \begin{bmatrix} \mathbf{x}_i,~\mathbf{y}_i \end{bmatrix}$
as input, and returns a prediction
$\mathcal{\hat Y}_i = \begin{bmatrix} \mathbf{\hat x}\_{i+1},\~\mathbf{\hat y}\_{i+1} \end{bmatrix}$ for what the next vector
should be. The hat $\hat~$ denotes the output of the model, as opposed
to actual vectors from the dataset.

With such a model at hand, generating a drawing can be approached
as an iterative process:
 1. pick up an initial vector $\mathcal{X}_{i=1}$ as the current one
 2. predict the next vector $\mathcal{\hat Y}_i = f(\mathcal{X}_i)$
 3. use the prediction as the current vector
 4. go back to step 2

![](images/predict.png)

In order to skip picking up the first point, let's make each
drawing $\mathcal{X}$ start with the same vector
$\mathcal{X}_1 = \overrightarrow 0$, ending up with nicely-aligned
sequences $\mathcal{X}$ and $\hat{\mathcal{Y}}$ of length $N$.

$f$ is not an arbitrary function. Let's make it a feedforward neural
network ($f = nn\_{\_W}$) with a hidden layer of size $H = 128$
and hyperbolic tangent ($tanh$) activation function. If you're fuzzy
on what it means, don't run away. What's inside $nn\_{\_W}$ is much less
relevant than how we interact with it from the outside. Put another
way: feel free to consider it a black box.

![](images/nn_equations.png)

All there is to understand is that given a bunch of weight $W$, the
neural network is a function $nn\_{\_W}$ that associate to each input
$X_i$ a given output $\mathcal{\hat Y}_i$. On the way, it computes an
internal vector $\mathbf{h}$ whose size $H$ conditions the complexity
of $nn\_{\_W}$. Since matrix multiplications and sums are linear
operations, throwing a $tanh$ into the mix ensures the resulting
$nn\_{\_W}$ is more than just a linear model.

To achieve any kind of meaningful generation with $nn\_{\_W}$, we first
need to train it. Concretely, it means that we are looking for a set of
weights $W$ that make each prediction $\mathcal{\hat Y}_i$
as close as possible from the "true next vector"
$\mathcal{Y}_i = \mathcal{X}\_{i+1}$ as available in the dataset.
These vectors are are our labels:
$\mathcal{Y} = \mathcal{Y}_1\~...\~\mathcal{Y}\_N$.

That leaves us with a supervised auto-regression problem: *supervised*
because there are labels, *regression* because these labels are
real-valued, and *auto* because they are essentially the same as the
data, but shifted by one position: $\mathcal{Y}\_i = \mathcal{X}\_{i+1}$.

![](images/train.png)

In order to quantify how close the predictions are from the labels,
we need a *loss function* $\mathscr{L}(\mathcal{\hat{Y}},\mathcal{Y})$.
For instance, it could be the the sum of the distances between each
prediction $\mathcal{\hat Y_i}$ and its corresponding label
$\mathcal{Y}_i$

$$
\mathscr{L}(\mathcal{\hat{Y}},\mathcal{Y}) =
\sum_{i=1}^{N-1}
\sqrt{(\mathbf{\hat{x}}\_{i+1} - \mathbf{x}\_{i+1}) ^ 2 +
(\mathbf{\hat{y}}\_{i+1} - \mathbf{y}\_{i+1}) ^ 2}
$$

The smaller the value of
$\mathscr{L}(\mathcal{\hat Y},\mathcal{Y})$, the closest the
generated drawing $\mathcal{\hat Y}$ is from the original drawing
$\mathcal{Y}$. If it is $0$, then $\mathcal{\hat Y} = \mathcal{Y}$.
Overfitting is not an issue here, since we do not care about
generalization, but about accurate generation.
 
Now, it turns out nobody uses this loss, for two reasons. First, the
square root is expensive to compute, so let's get rid of it. Computing
the sum of **squared** distances between the predictions and the
labels comes with the added benefit that larger discrepancies cost more.
Second, normalizing the sum to an average makes the loss for a given
drawing and its generated version independent of its size, which makes
it possible to compare the losses across drawings.

The resulting loss function is well-known and has its own name: it
is the *mean squared error* ($MSE$):

$$
MSE(\mathcal{\hat Y},\mathcal{Y}) =
\frac{1}{N - 1} \sum_{i=1}^{N-1}
\left\[
(\mathbf{\hat{x}}\_{i+1} - \mathbf{x}\_{i+1}) ^ 2 +
(\mathbf{\hat{y}}\_{i+1} - \mathbf{y}\_{i+1}) ^ 2
\right\]
$$

To summarize, given a dataset of $M$ drawings $\mathcal{X}$ and their
corresponding labels $\mathcal{Y}$, training $nn\_{\_W}$ means finding
a set of weights $W_{optimal}$ such as:

$$
W_{optimal} = \underset{W}{\argmin}~ \frac{1}{M} \sum_{(\mathcal{X},~\mathcal{Y})} MSE(nn\_{\_W}(\mathcal{X}),~\mathcal{Y})
$$

In practice, $W_{optimal}$ is computed by gradient descent. Broadly,
the strategy is to build iteratively a sequence of weights $W_t$ that
hopefully converges to $W_{optimal}$. It's made possible for reasonable
values of a parameter called the "learning rate" $\eta$, wich controls
how aggressively $W_t$ is updated at each iteration:

$$
\begin{aligned}
W_{t+1} = W_t - \eta \times \frac{\partial \mathscr{L}}{\partial W_t}(\mathcal{\hat Y}, \mathcal{Y})
\\\\\[3pt]
\text{where }\mathcal{\hat Y} = nn\_{\_{~W_t}}(\mathcal{X})
\end{aligned}
$$

This equation (and a variety of subtler variations) powers the whole
edifice of deep learning edifice, effectively allowing neural networks
to learn from data.

Although diving into the details of gradient descent and the exact
gradient descent equations for $f_W$ is out of the scope of this
article the key insight to gain from the equation is as follow: the
loss needs to be derivable with respect to the model parameters in
order for the model to be trainable. Naturally, it is the case for
$MSE$ and feedforward neural nets.

So far, so good. We have a model capable of learning from sequences
of points and generating new ones. There is however, a minor
caveat: it doesn't work. You can train it, pleasantly watch the loss
go down, and the generation will fail at producing anything even
remotely satisfying. Worse, you'd probably reach better (smaller)
losses by shuffling the drawings vectors. How on Earth is that
possible?

I think you can see where it is going. The feedforward neural network
merely takes the current vector as input, and it's a rather poor
predictor for the next one. In order to take full advantage of the
sequential structure of the data, we'll need a recurrent neural network.

The key idea that leads to RNNs is as follow: instead of feeding only
the current vector $\mathcal{X}_i$ as input to the model, feed it all
the previous vectors $\mathcal{X}\_1~...~\mathcal{X}\_{i-1}$ as well.
Since a neural network accepts a fixed number of numbers as input,
we will have to get clever and encode the previous vectors as a single
vector $\mathbf{h}_i$ (called hidden state) of size $H$ (yes, the same
$H$ wich is also the size of the hidden layer in our feedforward
network).

The best is yet to come. How do we transform previous vectors
$\mathcal{X}\_1~...~\mathcal{X}\_{i-1}$ to the hidden state
$\mathbf{h}_i$? We don't. The model does, and make it available
to the next step by outputing it. 

![](images/rnn_equations.png)

The blue parts highlight the differences with the feedforward neural
network. Interestingly, it mainly comes down to updating and exposing
as input the internal vector $\mathbf{h}_i$ that was already in the
feedforward network as $\mathbf{h}$.

Again, the exact matrix multiplications and non-linearities at play
inside the network does not matter as much as how the model is used
concretely.

The generation process is now as follow:
 1. pick up an initial point $\mathcal{X}\_{i=1} = \overrightarrow 0$
    as the current one, and an initial vector
    $\mathbf{h}_{i=1} = \overrightarrow 0$ as the current hidden state
 2. predict the next point and hidden state vector
    $\begin{bmatrix} \mathcal{\hat Y}\_i,~\mathbf{h}_{i+1} \end{bmatrix} = rnn\_{\_W}(\mathcal{X}_i,~\mathbf{h}_i)$
 3. use them as current point and current hidden state
 4. go back to step 2

![](images/predict_rnn.png)

Unlike feedforward neural networks, the training now exhibits the same
kind of iterative structure as generation.

![](images/train_rnn.png)

That's all a RNN is: a regular neural network that carries along
a hidden state. It's trained the same way: by minimizing
$\mathscr{L}(\mathcal{\hat{Y}},\mathcal{Y})$ using gradient descent.

By now, I emphasized multiple times we don't really care about the
internals of the neural network. That's thanksbecause nowadays, the
presented equations have been  they have been superseeded by slighly
more complex ones that form the base for the so-called "long short
term memory" (LSTM) models. LSTMs present a RNNs interface, with
superior ability to remember and forget relevant pieces of
information about the sequences being modelled, even when these pieces
are far apart from each other within the sequences (long range
dependencies).

Let's illustrate by taking a *face* drawing as example:

![](images/dataset/face.gif)

In order to generate such a drawing, the neural network needs to know
how to draw a circle, most importantly how to end drawing it where
it started (long range dependency between the first and last
point of the circle). It then has to mostly forget about the circular
shape, and focus on the eyes and the mouth, while retaining information
about their relative position with respect to the enclosing circle.
That's where vanilla RNNs fail and LSTMs shine.

To be honest, it still blows my mind that a few thousand weights can
hold such high-level information, and that a few matrix multiplication
is enough to apply it. Welcome to deep learning.

In the rest of this post, I'll abusively refer to LSTMs as just "RNNs".
They are indeed the de facto RNN models for any machine learning
practitioner attempting to model sequences.

# Fist generated drawings

There is good news! We've finally got everything we need to generate
our first drawings.

For each category mentioned earlier (*Effeil tower*, *faces* and
*firetruck*), let's grab a RNN, feed it ~11.000 drawings $\mathcal{X}$,
output predictions $\mathcal{\hat Y}$, score them against the labels
$\mathcal{Y}$ using the $MSE$, let the gradients flow back through the
network by gradient descent, repeat multiple times (around 200 to 300
epochs) and, after half an hour of training per model on GPU ...

Tada!!!

![](images/generated/selected_trajectory_effeil.png)
![](images/generated/selected_trajectory_face.png)
![](images/generated/selected_trajectory_firetruck.png)

Pretty disappointing, right? 

Well, not quite. At least the model identifies the primitive shape of
each class: the Effeil towers are triangle-shaped, circles somewhat
start to appear in faces, and upon closer inspection, you might
distinguish rectangles in the generated firetruck drawings.

Before we find a way of improving the drawings trajectory, let's focus
on a more immediate problem: the model is unable to decide when to lift
the pencil to start a new stroke, left alone when to stop drawing. For
its defence, it's not it's fault. We simply didn't teach it how to.

![](images/generated/trajectory_effeil.gif)

The previous generated drawings have been limited to 25 points in order
not to get out of hand.

# Stroke state of drawings

I wrote earlier that drawings are represented in the dataset
as sequences of vectors, and proceeded to represent a drawing
as a contiguous sequence. But did I mention anywhere that the
sequence had to be contiguous?

Since I value your sanity (and mine), let's consider a simple drawing
as example, and omit the initial $\overrightarrow 0$ vector It will save us the indices nightmare.

![](images/effeil.gif)

That fake Effeil tower would be represented in the dataset as a list
of three strokes between which the pencil is lifted
$\mathcal{X} = \mathcal{S}_1,~\mathcal{S}_2,~\mathcal{S}_3$.

$$
\mathcal{S}_1 = \mathcal{X}\_{1}~...~\mathcal{X}\_{7}~~~~~~~~~~~
\mathcal{S}_2 = \mathcal{X}\_{8}~...~\mathcal{X}\_{10}~~~~~~~~~~~
\mathcal{S}_3 = \mathcal{X}\_{11}~...~\mathcal{X}\_{14}
$$

![](images/effeil_annotated.png)

While this shape of data is satisfying in terms of representational
power, it is much less so in terms of model input. Indeed, a
recurrent neural network expects a potentially-variable-length
sequences of points as input, not variable-length sequences of
variable-length sequences of points.

So we'll have to flatten this representation somehow.

We already went the most naive way, concatenating all strokes as one
big stroke, and it did not go too well:

$$
\mathcal{X} =
\begin{bmatrix} \mathbf{x}_1 & ... & \mathbf{x}_7 & \mathbf{x}_8 & ... & \mathbf{x}\_{10} & \mathbf{x}\_{11} & ... & \mathbf{x}\_{14}
\\\\
\mathbf{y}_1 & ... & \mathbf{y}_7 & \mathbf{y}_8 & ... & \mathbf{y}\_{10} & \mathbf{y}\_{11} & ... & \mathbf{y}\_{14}
\end{bmatrix}
$$

![](images/effeil_continuous.png)

So let's insert a special value $\mathbf{\delta}$ between each stroke
to inform the model where the pencil should be lifted.
$\mathbf{\delta}$ should be big enough in absolute value so that is
does not conflict with regular points components.

$$
\mathcal{X} =
\begin{bmatrix} \mathbf{x}_1 & ... & \mathbf{x}_7 & \mathbf{\delta} & \mathbf{x}_8 & ... & \mathbf{x}\_{10} & \mathbf{\delta} & \mathbf{x}\_{11} & ... & \mathbf{x}\_{14}
\\\\
\mathbf{y}_1 & ... & \mathbf{y}_7 & \mathbf{\delta} & \mathbf{y}_8 & ... & \mathbf{y}\_{10} & \mathbf{\delta} & \mathbf{y}\_{11} & ... & \mathbf{y}\_{14}
\end{bmatrix}
$$

That could potentially work, but the weird non-continuous behaviour
introduced would almost certainly confuse the model. Moreover, how
are we supposed to deal with the model output when it predicts
$\delta$ on one dimension, but a regular value on the other one?
That sounds like a source of endless complication.

We'd be better off selecting a third and last approach: introducing
an input dimension $\mathbf{p_2}$ that signals the end of of a stroke.
While we're at it, let's use another dimension $\mathbf{p_3}$ to
indicate the end of the drawing, and set $\mathbf{p_2} = 0$ when
that occurs. Finally, since it is nice to have all additional
dimensions summing to 1, let's intercalate the complementary "regular
point" dimension $\mathbf{p_3}$, better described as "neither end of
stroke nor end of drawing":

$$
\mathcal{X} =
\begin{bmatrix} \mathbf{x}_1 & ... & \mathbf{x}_7 & \mathbf{x}_8 & ... & \mathbf{x}\_{10} & \mathbf{x}\_{11} & ... & \mathbf{x}\_{14}
\\\\
\mathbf{y}_1 & ... & \mathbf{y}_7 & \mathbf{y}_8 & ... & \mathbf{y}\_{10} & \mathbf{y}\_{11} & ... & \mathbf{y}\_{14}
\\\\
1 & 1 & 0 & 1 & 1 & 0 & 1 & 1 & 0
\\\\
0 &  0 & 1 & 0 & 0 & 1 & 0 & 0 & 0
\\\\
0 &  0 & 0 & 0 & 0 & 0 & 0 & 0 & 1
\end{bmatrix}
\begin{matrix}
\\\\
\\\\
\leftarrow\footnotesize{\mathbf{p_1}\text{: is regular point?~~~~~~~~~~~~~~}}
\\\\
\leftarrow\footnotesize{\mathbf{p_2}\text{: is end-of-stroke point?\~\~~}}
\\\\
\leftarrow\footnotesize{\mathbf{p_3}\text{: is end-of-drawing point?}}
\end{matrix}
$$

We name $\begin{bmatrix} \mathbf{x}_i,~\mathbf{y}_i \end{bmatrix}$
the *trajectory* and
$\begin{bmatrix} \mathbf{p_1}_i,~\mathbf{p_2}_i,~\mathbf{p_3}_i \end{bmatrix}$
the *stroke state*.

Modelling the former is a regression problem with real-valued labels
$\begin{bmatrix} \mathbf{x}_i,~\mathbf{y}_i \end{bmatrix}$, and
the latter a classification problem, with labels
$\begin{bmatrix} \mathbf{p_1}\_{i+1},~\mathbf{p_2}\_{i+1},~\mathbf{p_3}\_{i+1} \end{bmatrix}$
pertaining to one of three classes:
*regular point* $\begin{bmatrix} 1, 0, 0 \end{bmatrix}$,
*end of stroke* $\begin{bmatrix} 0, 1, 0 \end{bmatrix}$ and
*end of drawing* $\begin{bmatrix} 0, 0, 1 \end{bmatrix}$.

Our model will solve both problems jointly, by taking 5-dimensional
vectors
$\mathcal{X}_i = \begin{bmatrix} \mathbf{x}_i,~\mathbf{y}_i,~\mathbf{p_1}_i,~\mathbf{p_2}_i,~\mathbf{p_3}_i \end{bmatrix}$,
and outputing similarly-shaped predictions
$\mathcal{\hat Y}_i = \begin{bmatrix} \mathbf{\hat x}\_{i+1},~\mathbf{\hat y}\_{i+1},~\tilde{\mathbf{p_1}}\_{i+1},~\tilde{\mathbf{p_2}}\_{i+1},~\tilde{\mathbf{p_3}}\_{i+1} \end{bmatrix}$, scored against labels
$\mathcal{Y}_i = \begin{bmatrix} \mathbf{x}\_{i+1},~\mathbf{y}\_{i+1},
\mathbf{p_1}\_{i+1},~\mathbf{p_2}\_{i+1},~\mathbf{p_3}\_{i+1} \end{bmatrix}$ using the 5-dimensional $MSE$:

$$
\begin{aligned}	
MSE&(\mathcal{\hat{Y}},\mathcal{Y})=	
\frac{1}{N}\sum\_{i=1}^N	
(\mathbf{\hat{x}}\_{i+1} - \mathbf{x}\_{i+1}) ^ 2 +	
(\mathbf{\hat{y}}\_{i+1} - \mathbf{y}\_{i+1}) ^ 2~+	
\\\\	
&+ \frac{1}{N}\sum\_{i=1}^N	
(\tilde{\mathbf{p_1}}\_{i+1} - \mathbf{p_1}\_{i+1}) ^ 2 +	
(\tilde{\mathbf{p_2}}\_{i+1} - \mathbf{p_2}\_{i+1}) ^ 2 +	
(\tilde{\mathbf{p_3}}\_{i+1} - \mathbf{p_3}\_{i+1}) ^ 2	
\end{aligned}
$$

At this point, the reader accustomed to fitting classification models
may wonder "what's this guy even doing? MSE as classification loss?
Nobody does that". Sure, but can you name the flaky assumption we're
making when doing so? If you're thinking likelihood of some normal
distribution, you're on the right path. If not, that's a question the
article will clearly answer later on. For the time being, please accept
the MSE.

There is a more pressing issue: the values
$\begin{bmatrix} \tilde{\mathbf{p_1}}\_{i+1},~\tilde{\mathbf{p_2}}\_{i+1},~\tilde{\mathbf{p_3}}\_{i+1} \end{bmatrix}$, as direct outputs of the
model, are arbritrary numbers, and by no mean valid stroke states with
one and only one component equal to $1$ while the others are $0$.
Sure, we could just make the highest component goes to $1$ and the
others to $0$; an effective strategy for forming a non-derivable model
incompatible with gradient descent. Instead, let's introduce a
much smarter idea.

The idea is a follow: let's normalize the model output so that
$\tilde{\mathbf{p_1}}\_{i+1} + \tilde{\mathbf{p_2}}\_{i+1} + \tilde{\mathbf{p_3}}\_{i+1} = 1$, and have it define a probability
mass function for a distribution from which  we can sample the real
stroke state predictions
$\hat{\mathcal{Y}}\_i = \begin{bmatrix} \hat{\mathbf{p_1}}\_{i+1},~\hat{\mathbf{p_2}}\_{i+1},~\hat{\mathbf{p_3}}\_{i+1} \end{bmatrix}$
as such:
* Draw "regular point" event $\hat{\mathcal{Y}}\_i = \begin{bmatrix} 1,0,0 \end{bmatrix}$ with probability $\tilde{\mathbf{p\_{1}}\_{i+1}}$
* Draw "end of stroke" event $\hat{\mathcal{Y}}\_i = \begin{bmatrix} 0,1,0 \end{bmatrix}$ with probability $\tilde{\mathbf{p\_{2}}\_{i+1}}$
* Draw "end of drawing" event $\hat{\mathcal{Y}}\_i = \begin{bmatrix} 0,0,1 \end{bmatrix}$ with probability $\tilde{\mathbf{p\_{3}}\_{i+1}}$

![](images/mdn_stroke_state.png)

For normalization, we're going to use the softmax function for
normalization:

$$
\text{softmax}\_{\_{T_\mathbf{p}}}(\tilde{\mathbf{p\_k}}) = \frac{\exp(\tilde{\mathbf{p}\_{k}}~/~T_\mathbf{p})}{\sum\limits\_{k=1}^3 \exp(\tilde{\mathbf{p\_k}}~/~T_\mathbf{p})},~~k=1..3
$$

$T_\mathbf{p}$ is a generation parameter called *temperature*. It defines how
harsh the softmax is at amplifying the difference between the
$\tilde{\mathbf{p\_k}}$s. Since it can be conceptually difficult to
gauge its influence $T_\mathbf{p}$ from the equation, hereafter are samples of
15 stroke states for model outputs
$\begin{bmatrix} \tilde{\mathbf{p_1}}\_{i+1},~\tilde{\mathbf{p_2}}\_{i+1},~\tilde{\mathbf{p_3}}\_{i+1} \end{bmatrix} = \begin{bmatrix} 3, 2, 1 \end{bmatrix}$
at various softmax temperatures:

![](images/softmax.png)

Or more concretely with actual Effeil towers generation:

![](images/generated/base_effeil_temperature_stroke_state.png)

At low temperature ($T_\mathbf{p} = 0.1$), the model does not take any risk
and consistently outputs the stroke state most represented in the data
: *regular point*, or $\begin{bmatrix} 1,~0,~0 \end{bmatrix}$, ending
up with a drawing that continues indefinitely without lifting the
pencil once, much like our former trajectory-only model. When
temperature goes up, the model dares sampling more and more of the
other stroke states, making the drawings more fragmented (more
*end-of-stroke* points, alias $\begin{bmatrix} 0,~1,~0 \end{bmatrix}$)
and shorter (higher probability of eventually generating
*end-of-drawing* event $\begin{bmatrix} 0,~0,~1 \end{bmatrix}$).

Now that the model can lift pencil and stop drawing by itself, let's
train it with $T_\mathbf{p}=1$ — as will always be the case, $T_\mathbf{p}$ being a
generation parameter only — and generate some some drawings with
$T_\mathbf{p} = 0.8$:

![](images/generated/selected_base_effeil.png)
![](images/generated/selected_base_face.png)
![](images/generated/selected_base_firetruck.png)

Hey! The generated Effeil towers and faces are starting to look like
ones. To say the least, the firetrucks are still pretty disappointing,
but not so much for their stroke state than for their trajectories.
They will get better as we push further the idea of turning our LSTM
from yet-too-deterministic to fully probabilistic.

# Trajectory as a random variable

Let's put the stroke state aside for one moment, and consider how the
trajectory
$\mathcal{X}_i = \begin{bmatrix} \mathbf{x}\_{i},~\mathbf{y}\_{i} \end{bmatrix}$
is modelled.

Currently, ignoring the hidden state, the model deterministically
outputs a prediction for the next vector:

$$
\mathcal{\hat{Y}_i} = lstm\_{\_W}(\mathcal{X_i})
$$

Guess what? We are going to apply the same idea as for the stroke
state. Instead of flat out predicting the next vector
$\hat{\mathcal{Y}}_i$, let's have the model output a 5-dimensional
vector $\tilde{\mathcal{Y}}_i$:

Except this
time, we're also going to put the distribution right at the heart of
training by building a loss based on it. This fundamental idea leads to
an extension of neural networks known as "mixture density networks"
(MDNs).

$$
\mathcal{\tilde{Y}\_i} = \begin{bmatrix} \mathbf{\mu\_x}\_{\_{i+1}},~\mathbf{\mu\_y}\_{\_{i+1}},~\mathbf{\sigma\_x}\_{\_{i+1}},~\mathbf{\sigma\_y}\_{\_{i+1}},~\mathbf{\rho\_{xy}}\_{\_{i+1}}
\end{bmatrix} = lstm\_{\_W}(\mathcal{X_i})
$$

In order to make the next equations a little tidier, let's temporarily
omit the annoying $\_{i+1}$ indices by setting
$\mathcal{\tilde{Y}\_i} = \begin{bmatrix} \mu_x,~\mu_x,~\sigma_x,~\sigma_y,~\rho_{xy} \end{bmatrix}$.

You may recognize these symbols. They are the parameters of a
2-dimensional normal distribution:
 * $\begin{bmatrix} \mu\_x,~\mu\_y \end{bmatrix}$ is the center of
   the distribution.
 * $\sigma_x$ and $\sigma_y$ are the variances in the direction of x
   and y, both greater than 0.
 * $\rho_{xy}$ is the correlation coefficient between x and y, ranging
   between -1 and 1 (excluded). In particular, when
   $\mathbf{\rho_{xy}} = 0$, the distribution is equivalent to two
   independent normal distributions along x and y.

![](images/normals.png)

We can make sure that $\mathbf{\sigma_x}$ and $\mathbf{\sigma_y}$ are
greater than 0 by passing them through an exponential, and that
$\mathbf{\rho\_{xy}}$ is between -1 and 1 by passing it through an
hyperbolic tangent, much like we enforced
$\tilde{\mathbf{p_1}} + \tilde{\mathbf{p_2}} + \tilde{\mathbf{p_3}} = 1$,
for the output stroke state.

Finally, we sample the actual prediction for the next vector
$\hat{\mathcal{Y}}\_i = \begin{bmatrix} \hat{\mathbf{x}}\_{i+1},~\hat{\mathbf{y}}\_{i+1} \end{bmatrix}$
from the normal
$\mathcal{N}(\mathbf{\mu}\_{i+1}, \mathbf{\Sigma}\_{i+1})$
centered at
$\mathbf{\mu}\_{i+1} = \begin{bmatrix} \mu\_x,~\mu\_y \end{bmatrix}$
with covariance matrix 
$\Sigma_{i+1} = \begin{bmatrix}
\sigma\_x^2 & \rho\_{xy} \sigma_x \sigma_y
\\\\
\rho\_{xy} \sigma\_x \sigma\_y & \sigma\_x^2
\end{bmatrix}$.
.

![](images/mdn_trajectory.png)

$$
\begin{aligned}
&\begin{bmatrix}
\mathbf{\mu\_x},~\mathbf{\mu\_y},~\mathbf{\sigma\_x},~\mathbf{\sigma\_y},~\mathbf{\rho\_{xy}}
\end{bmatrix},~\mathbf{h}_{i+1} =
lstm\_{\_W}(\mathcal{X_i},~\mathbf{h}_i)
\\\\\[5pt]
&\hat{\mathcal{Y}}_i \sim \mathcal{N}(\mathbf{\mu}\_{i+1},~\mathbf{\Sigma}\_{i+1})
\\\\\[5pt]
&\text{ where }
\begin{cases}
\mathbf{\mu}\_{i+1} = \begin{bmatrix}
\mathbf{\mu}\_x,~\mathbf{\mu\_y}
\end{bmatrix}
\\\\\[5pt]
\mathbf{\Sigma}\_{i+1} = T\_{xy} \times \begin{bmatrix}
\mathbf{\sigma\_x}^2 & \mathbf{\rho\_{xy}} \mathbf{\sigma_x} \mathbf{\sigma_y}
\\\\
\mathbf{\rho\_{xy}} \mathbf{\sigma\_x} \mathbf{\sigma\_y} & \mathbf{\sigma\_x}^2
\end{bmatrix}
\end{cases}
\end{aligned}
$$

It order gain the same flexibility for trajectory generation as we had
for stroke state generation, let's scale the spread of the distribution
by a temperature parameter $T_{xy}$ so that
$\mathbf{\Sigma}\_{i+1} = T_{xy} \times \mathbf{\Sigma}\_{i+1}$.
The temperature determines how far from the center $\mathbf{\mu}\_{i+1}$
the model is allowed to sample
$\mathcal{\hat{Y}_i}$. When $T\_{xy} \to 0$, we end up with a model
ressembling our former deterministic, consistently outupting
$\begin{bmatrix} \mathbf{\mu\_{x}},~\mathbf{\mu\_{y}} \end{bmatrix}$.
When $T\_{xy} \to \infty$, we end up with fully random trajectories.

Taking a leap into the future, let's attempt to generate Effeil towers
at various temperatures. The highest the temperature, the most
chaotic the trajectory.

![](images/generated/gmm128_1_effeil_temperatures_gmm.png)

Great! We've now got a fully probabilistic for both the trajectory
and stroke state.

The battle is not over, though. We still have to train it. While
we cowardly got away with the MSE for the stroke state, good luck
applying it to trajectory outputs of dimension 5 and labels of
mismatching dimension 2.

More seriously, how do we define a loss
$\mathscr{L}(\begin{bmatrix} \mathbf{\mu}\_{i+1},~\mathbf{\Sigma}\_{i+1} \end{bmatrix}, \mathcal{Y}_i)$
that quantifies the adequation of the model outputs with their
corresponding labels?

It turns out probability theory provides a neat answer to that question:
just use the density
$p\_{\mathbf{\mu}\_{i+1},~\mathbf{\Sigma}\_{i+1}}$
of normal distribution
$\mathcal{N}(\mathbf{\mu}\_{i+1}, \mathbf{\Sigma}\_{i+1})$:

$$
p_{\mathbf{\mu},~\mathbf{\Sigma}}(x, y) =
  \frac{1}{2 \pi \mathbf{\sigma_x} \mathbf{\sigma_y} \sqrt{1-\mathbf{\rho_{xy}}^2}}
  \exp\left(
    -\frac{1}{2(1-\mathbf{\rho_{xy}}^2)}\left\[
      \frac{(x-\mathbf{\mu_x})^2}{\mathbf{\sigma_x}^2} +
      \frac{(y-\mathbf{\mu_y})^2}{\mathbf{\sigma_y}^2} -
      \frac{2\mathbf{\rho_{xy}}(x-\mathbf{\mu_y})(y-\mathbf{\mu_y})}{\mathbf{\sigma_x} \mathbf{\sigma_y}}
    \right]
  \right)
$$

By the very definition of continuous probability distribution, the
larger the value of
$p\_{\mathbf{\mu}\_{i+1},~\mathbf{\Sigma}\_{i+1}}(\mathcal{Y}\_i)$,
the better the chance of sampling $\mathcal{Y}\_i$ from
$\mathcal{N}(\mathbf{\mu}\_{i+1},~\mathbf{\Sigma}\_{i+1})$, though that
is relevant for generation, not training. Taking it backward,
however, the higher the value of
$p\_{\mathbf{\mu}\_{i+1},~\mathbf{\Sigma}\_{i+1}}(\mathcal{Y}\_i)$,
the better the chance — precisely, the *likelihood* — that model output
$\mathbf{\mu}\_{i+1},~\mathbf{\Sigma}\_{i+1}$ form a normal
$\mathcal{N}(\mathbf{\mu}\_{i+1},~\mathbf{\Sigma}\_{i+1})$
from which label $\mathcal{\mathcal{Y}\_i}$ could have been sampled.

If you read the last sentence in diagonal, I suggest you go back and
read it again. It is the crux to switching from generation to 
training, effectively providing us with a way to score model output
$\mathcal{\mathcal{Y}}_i = \begin{bmatrix} \mathbf{\mu}\_{i+1},~\mathbf{\Sigma}\_{i+1} \end{bmatrix}$
given label $\mathcal{Y}_i$.

The concept is easily generalized to all labels
$\mathcal{Y} = \mathcal{Y}_1,~...,~\mathcal{Y}_N$
and models outputs
$\mathcal{Y} = \begin{bmatrix} \mathbf{\mu}\_1,~\mathbf{\Sigma}\_1 \end{bmatrix}, ..., \begin{bmatrix} \mathbf{\mu}\_N,~\mathbf{\Sigma}\_N \end{bmatrix}$ by defining the likelihood function:

$$
\mathcal{L}(\mathcal{Y}\~;\~\mu, \mathbf{\Sigma}) =
\prod\_{i=1}^N p\_{\mathbf{\mu}\_{i+1},~\mathbf{\Sigma}\_{i+1}}(\mathcal{Y}_i)
$$

At this point, we're almost done, since — assuming independence of
the $\mathcal{Y}_i$'s — maximizing this quantity (a process known as
*maximum likelihood estimation*) maximizes the adequation of the model
outputs to the labels.

However, since we much prefer means to products for reasons of
numerical stability, and $\argmin$ to $\argmax$ in order to define
a loss, please consider the two following properties of $\argmax$:
 * wrapping what's inside $\argmax$ into whatever strictly increasing
   function (such as $\log$) does not change the value of the $\argmax$.
 * maximizing a function is the same as minimizing its opposite:
   $\argmax f = \argmin -f$.

So:

$$
\begin{aligned}
W_{optimal} =~&
\underset{W}{\argmax}
\prod\_{i=1}^N p\_{\mathbf{\mu}\_{i+1},~\mathbf{\Sigma}\_{i+1}}(\mathcal{Y}_i)
\\\\
\underset{\log}{=}~&
\underset{W}{\argmax}~
\log \left( \prod\_{i=1}^N p\_{\mathbf{\mu}\_{i+1},~\mathbf{\Sigma}\_{i+1}}(\mathcal{Y}_i)
\right)
\\\\
=~&
\underset{W}{\argmax}
\sum\_{i=1}^N \log p\_{\mathbf{\mu}\_{i+1},~\mathbf{\Sigma}\_{i+1}}(\mathcal{Y}_i)
\\\\
\underset{\times \frac{1}{N}}{=}~&
\underset{W}{\argmax} \frac{1}{N}
\sum\_{i=1}^N \log p\_{\mathbf{\mu}\_{i+1},~\mathbf{\Sigma}\_{i+1}}(\mathcal{Y}_i)
\\\\
W\_{optimal}
\underset{\times \text{-}1}{=}~&
\underset{W}{\argmin} \frac{1}{N}
\sum\_{i=1}^N \log p\_{\mathbf{\mu}\_{i+1},~\mathbf{\Sigma}\_{i+1}}(\mathcal{Y}_i)
\end{aligned}
$$

We're left with the minimization of a function with respect to
its parameters given the labels and model. Does that ring a bell?

It's a loss function!

Victory!

Omitting the hidden state:

$$
\mathscr{L}(\tilde{\mathcal{Y}},\mathcal{Y}) =
\frac{1}{N} \sum\_{i=1}^N \log p\_{\mathbf{\mu}\_{i+1},~\mathbf{\Sigma}\_{i+1}}(\mathcal{Y}_i)
\\\\\[5pt]
\text{where }
\tilde{\mathcal{Y}} = \begin{bmatrix} \mathbf{\mu},~\mathbf{\Sigma} \end{bmatrix} =
lstm\_{\_W}(\mathcal{X})
$$

Or, replacing
$p\_{\mathbf{\mu}\_{i+1},~\mathbf{\Sigma}\_{i+1}}$
with its actual equation:

$$
\mathscr{L}(\tilde{\mathcal{Y}},\mathcal{Y}) =
\frac{1}{N}\sum\_{i=1}^N
log(2 \pi \mathbf{\sigma_x}\_{\_{i+1}} \mathbf{\sigma_y}\_{\_{i+1}} \sqrt{1-{\mathbf{\rho_{xy}}\_{\_{i+1}}}^2}) + 
\frac{(\mathbf{x}\_{i+1}-\mathbf{\mu_x}\_{\_{i+1}})^2}{\mathbf{\sigma_x^2}\_{\_{i+1}}} +
\frac{(\mathbf{y}\_{i+1}-\mathbf{\mu_y}\_{\_{i+1}})^2}{\mathbf{\sigma_y^2}\_{\_{i+1}}} -
\frac{2 \mathbf{\rho\_{xy}}\_{\_{i+1}}(\mathbf{x}\_{i+1}-\mathbf{\mu_y}\_{i+1})(\mathbf{y}\_{i+1}-\mathbf{\mu_y}\_{i+1})}{\mathbf{\sigma_x} \_{i+1}\mathbf{\sigma_y}\_{i+1}}
\\\\\[5pt]
\text{where }
\begin{bmatrix} \mathbf{\mu_x},~\mathbf{\mu_y},~\mathbf{\sigma_x},~\mathbf{\sigma_{y}},~\mathbf{\rho_{xy}}
\end{bmatrix} =
lstm\_{\_W}(\mathcal{X})
$$

Rather scary, heh? Especially as compared to our previous MSE. But now,
not only are we optimizing the center of the predictions
$\mathbf{\mu}\_{i+1}$, we are also optimizing their variance
$\mathbf{\Sigma}\_{i+1}$. In fact, the MSE is a special case of that
loss, and clarifying in what way will provide valuable insight for the
rest of the article.

So, let's assume independant normal distributions ($\mathbf{\rho_{xy}} = 0$)
and equal variance ($\sigma_x = \sigma_y = \sigma$) along x and y
across all model outputs. We end up with an isotropic
covariance matrices
$\Sigma = \begin{bmatrix} \mathbf{\sigma}^2 & 0 \\\\ 0 & \mathbf{\sigma}^2 \end{bmatrix}$
and the following loss:

$$
\mathscr{L}(\tilde{\mathcal{Y}},~\mathcal{Y}) =
\frac{1}{N}\sum\_{i=1}^N
log(2 \pi) +
\frac{1}{\sigma} \left\[
(\mathbf{x}\_{i+1} - \mathbf{\mu_x}\_{\_{i+1}})^2 +
(\mathbf{y}\_{i+1} - \mathbf{\mu_y}\_{\_{i+1}})^2
\right\]
\\\\\[5pt]
\text{where }
\tilde{\mathcal{Y}}_i = \begin{bmatrix} \mathbf{\mu\_x}\_{\_{i+1}},~\mathbf{\mu\_y}\_{\_{i+1}}
\end{bmatrix} =
lstm\_{\_W}(\mathcal{X_i})
$$

Since we are minimizing $\mathscr{L}(\mathcal{\hat{Y}},\mathcal{Y})$,
the constant terms $log(2 \pi)$ and $\frac{1}{\sigma}$ can be safely
removed. Furthermore, assuming the predictions are directly the output
centers from the model ($\tilde{\mathcal{Y}} = \hat{\mathcal{Y}}$):

$$
\begin{aligned}
&\mathscr{L}(\mathcal{\hat{Y}},\mathcal{Y}) =
\frac{1}{N}\sum\_{i=1}^N
(\mathbf{x}\_{i+1} - \mathbf{x}\_{i+1})^2 +
(\mathbf{y}\_{i+1} - \mathbf{y}\_{i+1})^2
\\\\
&\mathscr{L}(\mathcal{\hat{Y}},\mathcal{Y}) =
MSE(\mathcal{\hat{Y}},\mathcal{Y})
\end{aligned}
$$

We're coming full circle, but with a fundamental insight: the mean
squared error is not just the average of the squared euclidian distance
between the predictions and the labels. It has a more profound meaning:
**minimizing the MSE between the predictions and the labels is
equivalent to maximizing the likelihood of the labels under a normal
distribution with isotropic covariance matrix**.

# Stroke state

Let's now apply the same principles as in the last section to derive an
adequate loss function for the stroke state.

We previously used MSE as loss, and I challenged you to tell why it
wasn't a good idea. The last section provides a neat answer to that
question: it assume a normal distribution over the stroke states.

Does it seem like a reasonable assumption for a set of three possible events?

Nop.

The actual distribution is much simpler that that. Leaving the
trajectory aside, the output of the model
$\tilde{\mathcal{Y}}\_{i} = \tilde{\mathbf{p}}\_{i+1} = \begin{bmatrix} \tilde{\mathbf{p_1}}\_{i+1},~\tilde{\mathbf{p_2}}\_{i+1},~\tilde{\mathbf{p_3}}\_{i+1} \end{bmatrix}$
defines a probability mass function $p_{\tilde{\mathbf{p}}\_{i+1}}$
(the equivalent of the density function, but for discrete distributions)
over the three possible stroke states, which can be evaluated against
the corresponding label
$\mathcal{Y}\_{i} = \mathbf{p}\_{i+1} = \begin{bmatrix} \mathbf{p_1}\_{i+1},~\mathbf{p_2}\_{i+1},~\mathbf{p_3}\_{i+1} \end{bmatrix}$
in a straigtforward manner:

$$
p_{\tilde{\mathbf{p}}\_{i+1}}(\mathcal{Y}_i) =
\begin{cases}
{\tilde{\mathbf{p_1}}\_{i+1}} \text{ if } \mathcal{Y}_i = \begin{bmatrix} 1,0,0 \end{bmatrix}
\\\\
{\tilde{\mathbf{p_2}}\_{i+1}} \text{ if } \mathcal{Y}_i = \begin{bmatrix} 0,1,0 \end{bmatrix}
\\\\
{\tilde{\mathbf{p_3}}\_{i+1}} \text{ if } \mathcal{Y}_i = \begin{bmatrix} 0,0,1 \end{bmatrix}
\end{cases}
$$

In plain English, $p_{\tilde{\mathbf{p}}\_{i+1}}(\mathcal{Y}_i)$
is the probability outputed by the model for the correct class. The
higher, the better.

It can be pleasantly rewritten as:

$$
p_{\tilde{\mathbf{p}}\_{i+1}}(\mathcal{Y}\_i) = \prod_{k=1}^3 \tilde{\mathbf{p_k}}\_{i+1}^{\mathbf{p_k}\_{i+1}}
$$

By the same derivation as for the trajectory, assuming independance of
the $\mathcal{Y}_i$'s, we end up with a loss as standard as the MSE,
but for classification. It is the *cross-entropy* loss:

$$
\begin{aligned}
\mathscr{L}(\tilde{\mathcal{Y}}, \mathcal{Y}) = - \frac{1}{N} \sum_{i=1}^N \sum_{k=1}^3 \mathbf{p_k}\_{i+1} \log(\tilde{\mathbf{p_k}}\_{i+1})
\end{aligned}
$$

# Third generated drawings

Putting it all together, we are left with a RNN that takes 5 number
$\mathcal{X}\_i = \begin{bmatrix} \mathbf{x}\_i,~\mathbf{y}\_i,~\mathbf{p_1}\_i,~\mathbf{p_2}\_i,~\mathbf{p_3}\_i \end{bmatrix}$
and a hidden state vector of size $H = 128$ as input, and outputs 8
numbers
$\tilde{\mathcal{Y}}\_i = \begin{bmatrix} \mathbf{\mu\_x}\_{\_{i+1}},~\mathbf{\mu\_y}\_{\_{i+1}},~\mathbf{\sigma\_x}\_{\_{i+1}},~\mathbf{\sigma\_y}\_{\_{i+1}},~\mathbf{\rho\_{xy}}\_{\_{i+1}},~\tilde{\mathbf{p_1}}\_{\_{i+1}},~\tilde{\mathbf{p_2}}\_{\_{i+1}},~\tilde{\mathbf{p_3}}\_{\_{i+1}}
\end{bmatrix}$
and a hidden state vector:

$$
\tilde{\mathcal{Y}}_i,~\mathbf{h}\_{i+1} = lstm\_{\_W}(\mathcal{X_i}, \mathbf{h}_i)
$$

It is trained by matching its outputs against the corresponding labels
$\mathcal{Y}\_i = \begin{bmatrix} \mathbf{x}\_{i+1},~\mathbf{y}\_{i+1},~\mathbf{p_1}\_{i+1},~\mathbf{p_2}\_{i+1},~\mathbf{p_3}\_{i+1} \end{bmatrix}$
using the half-regression half-regression loss:

$$
\begin{aligned}
\mathscr{L}(\tilde{\mathcal{Y}}\_i,~\mathcal{Y}\_i) =&
-\frac{1}{N}\sum\_{i=1}^N
log(2 \pi \mathbf{\sigma_x}\_{\_{i+1}} \mathbf{\sigma_y}\_{\_{i+1}} \sqrt{1-\mathbf{\rho_{xy}}\_{\_{i+1}}^2}) + 
\frac{(\mathbf{x}\_{i+1}-\mathbf{\mu_x}\_{\_{i+1}})^2}{\mathbf{\sigma_x}\_{\_{i+1}}^2} +
\frac{(\mathbf{y}\_{i+1}-\mathbf{\mu_y}\_{\_{i+1}})^2}{\mathbf{\sigma_y}\_{\_{i+1}}^2} -
\frac{2\mathbf{\rho_{xy}}\_{\_{i+1}}(\mathbf{x}\_{i+1}-\mathbf{\mu_x}\_{\_{i+1}})(\mathbf{y}\_{i+1}-\mathbf{\mu_y}\_{\_{i+1}})}{\mathbf{\sigma_x}\_{\_{i+1}} \mathbf{\sigma_y}\_{\_{i+1}}}
\\\\
&- \frac{1}{N} \sum_{i=1}^N \sum_{k=1}^3 \mathbf{p_k}\_{i+1} \log(\tilde{\mathbf{p_k}}\_{i+1})
\end{aligned}
$$

Generation is achieved by sampling the next trajectory and stroke state
$\hat{\mathcal{Y}}\_i = \begin{bmatrix} \hat{\mathbf{x}}\_{i+1},~\hat{\mathbf{y}}\_{i+1},~\hat{\mathbf{p_1}}\_{i+1},~\hat{\mathbf{p_2}}\_{i+1},~\hat{\mathbf{p_3}}\_{i+1} \end{bmatrix}$
from the resulting probability distributions at given temperatures
$T_\mathbf{xy}$ and $T_\mathbf{p}$:

$$
\hat{\mathcal{Y}}_i \sim \mathcal{N}(\begin{bmatrix} \mathbf{\mu\_x}\_{\_{i+1}},~\mathbf{\mu\_y}\_{\_{i+1}} \end{bmatrix}, \begin{bmatrix} \mathbf{\sigma\_x}\_{\_{i+1}},~\mathbf{\sigma\_y}\_{\_{i+1}},~\mathbf{\rho\_{xy}}\_{\_{i+1}} \end{bmatrix}) \times \mathcal{P}(\begin{bmatrix} \tilde{\mathbf{p_1}}\_{i+1},~\tilde{\mathbf{p_2}}\_{i+1},~\tilde{\mathbf{p_3}}\_{i+1} \end{bmatrix})
$$ 

![](images/mdn_full.png)

And ...

![](images/generated/selected_gmm128_1_effeil.png)
![](images/generated/selected_gmm128_1_face.png)
![](images/generated/selected_gmm128_1_firetruck.png)

Wow. Not quite as satisfying as expected after such a struggle ...
Sure, Effeil towers and faces are getting more realistic, but
firetrucks ... Something must be definitely wrong with the model.

Actually, yes.

Luckily, the last piece of maths we need to make it right is quite a
straightforward one, especially as compared to what we endured in the
last paragraphs. We're actually very close to tackling drawings
generation for good. Let's loose our sanity and upgrade the
5-parameters normal modelling the trajectory to one with 120
parameters.

# GMM

As the motto of machine learning practitioners goes: "all models are
wrong, but some are useful". While our model is useful, it still
doesn't get the trajectories right enough.

The reason is simple: we assumed a normal distribution for the
trajectory, even though we saw it wasn't the case when we plotted it
at the beginning of the article (symmetric, but too spread out):

![](images/distplot_xy.png)

So let's upgrade to a more sophisticated probability distribution.
Instead of having the model return the parameters for a single normal,
let's have it return the parameters for $K$ of them: $K$ centers
$({\mathbf{\mu}\_k}\_{\_{i+1}}) \_{k=1..K}$ and $K$
covariance matrices $({\mathbf{\Sigma}\_k}\_{\_{i+1}}) \_{k=1..K}$
$\mathbf{\mu}$. Additionally, it returns $K$ coefficients
$({\mathbf{\pi}\_k}\_{\_{i+1}}) \_{k=1..20}$ associated with the normals,
that we'll force to satisfy
$\sum_{k=1}^K {\mathbf{\pi}\_k}\_{\_{i+1}} = 1$ by passing
them through a softmax conditioned on the trajectory temperature
$T_\mathbf{xy}$:

$$
{\mathbf{\pi}\_k}\_{\_{i+1}} = \frac{\exp({\mathbf{\pi}\_k}\_{\_{i+1}}~/~T_\mathbf{xy})}{\sum\limits\_{k=1}^3 \exp({\mathbf{\pi}\_k}\_{\_{i+1}}~/~T_\mathbf{xy})},~~k=1..K
$$

Sampling goes as follow: pick up one of the $K$ normals
${\mathbf{\mu}\_k}\_{\_{i+1}}, {\mathbf{\Sigma}\_k}\_{\_{i+1}}$ and
with probability ${\mathbf{\pi}\_k}\_{\_{i+1}}$ and sample from it.
The resulting probability distribution is called a *gaussian mixture
model*, or GMM.

![](images/gmm.png)

As in the Google Brain paper, we're going to use $K=20$ normals,
resulting in the RNN outputing $K$ centers (2 numbers), $K$ covariance
matrices (3 numbers) and $K$ coefficients, for a total of
$(2 + 3 + 1) \times K = 120$ outputs by prediction, for the trajectory
alone!

![](images/mdn_gmm.png)

As a last step, we have to derive a loss for the GMM-modelled
trajectory. It turns out to be a pretty straigtforward extension of
the previous loss, given the density
{% raw %}
$p\_{{\mathbf{\mu}\_k}\_{\_{i+1}},~{\mathbf{\Sigma}\_k}\_{\_{i+1}}}(\mathcal{Y}_i)$
{% endraw %}
of the $k$-th normal for the $i$-th output:

{% raw %}
$$
\mathscr{L}(\tilde{\mathcal{Y}},~\mathcal{Y}) = \frac{1}{N} \sum\_{i=1}^N \sum\_{k=1}^K {\mathbf{\pi}\_k}\_{\_{i+1}} \log p\_{{\mathbf{\mu}\_k}\_{\_{i+1}},~{\mathbf{\Sigma}\_k}\_{\_{i+1}}}(\mathcal{Y}_i)
\\\\\[5pt]
\text{where }
\tilde{\mathcal{Y}} = \begin{bmatrix} \mathbf{\mu}\_{\_{i+1}},~\mathbf{\Sigma}\_{\_{i+1}},~\mathbf{\pi}\_{\_{i+1}} \end{bmatrix} =
lstm\_{\_W}(\mathcal{X})
$$
{% endraw %}

This time, we're definitively done. This model for unconditional
generation of drawings is the same as in the paper.

![](images/sketch_rnn.png)

Credit: [A Neural Representation of Sketch Drawings](https://arxiv.org/abs/1704.03477), David Ha and Douglas Eck, Figure 2. Retouched to remove
input vector $z$ not pertaining to unconditional generation of drawings.

# Fourth and final generated drawings

Time for some fun. Let's use our final model to produce more drawings!

![](images/generated/selected_gmm128_20_effeil.png)
![](images/generated/selected_gmm128_20_face.png)

Alright! We are done with Effeil towers and faces. What about the
long-awaited firetrucks?

![](images/generated/selected_gmm128_20_firetruck.png)

They look nice enough, but we can do better. Let's tweak the size of
the hidden state, which conditions the complexity of the RNN, passing
it to $H = 512$ instead of $H = 128$. Let's also thrown more drawings
in (between 30k and 40k drawing instead of 11k). Training now lasts
around 2 hours per class of drawings, but the firetrucks look much
better:

![](images/generated/selected_gmm512_20_firetruck.png)

Scale, flashing light, axles. They have it all.

That's enough metal for now. Let's turn to more organic stuff.

![](images/generated/selected_gmm512_20_carrot.png)

It's quite obvious what they are.

Now, what better way to close this article than by generating some
animals?

![](images/generated/selected_gmm512_20_cat.png)
![](images/generated/selected_gmm512_20_crab.png)
![](images/generated/selected_gmm512_20_penguin.png)
![](images/generated/selected_gmm512_20_giraffe.png)

In case you are unsure what the last class of drawings is (either
lamas, diplodocuses or random mammals), they are giraffes. While it is
quite complicated to recognize them, they are not too bad, considering
the original drawings ...

![](images/dataset/selected_giraffe.png)

No comment.

# Conclusion

In this article, we explored drawings generation using a recurrent
neural network by jointly predicting the trajectory, when to lift
pencil, and when to stop drawing. Since a vanilla LSTM quickly showed
its limit in such an endeavour, we had to stack a probabilistic layer
on top of it, making it a mixture density network, that we effectively
trained by maximizing the likelihood of the next point under
probability distributions shaped by the outputs of the model.

That was quite a journey, but with great results.

From shaky hand-drawn circle, triangle and square primitives in the
dataset, the final model generates smooth ones. In the process, it
manages to put together a "grammar of drawings" positioning these
primitives in such a way that they form recognizable drawings
actually surpassing quite a lot of the examples it has been trained on.
