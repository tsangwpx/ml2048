ML2048
-------------------------------

# Result

Achieved 2048 over 85% games.

![Maximum tile distribution](assets/tile-distribution.png)

# Brief description

* Actor-Critic Algorithm
* Proximal Policy Optimization
* Generalized Advantage Estimation
* Convolutional neural network frontend shared by actor and critic
* Fine tuning over epoches

The 2048 board consists of a 4x4 grid
and there are at least 16 possibilities in each cell (empty, 2, 4, 8, up to 131072).

Here the board is one-hot encoded and feed into CNN network.
Since the input is sparse, it is convoluted depthwisely with 1x4, 4x1, and 4x4 kernels separately,
which the agent hopefully learns horizontal, vertical, and overall grid information.
The intermediate result is then pointwisely convoluted, concatenated, convoluted again
to output the required number of features to actor and critic layers.

See [visualization.ipynb](notebooks/visualization.ipynb) to run the trained agent.

Explore the messy training code by starting with [run_train3.py](run_train3.py).

The 2048 environment is in [game_numba.py](src/ml2048/game_numba.py).

[1]: https://iclr-blog-track.github.io/2022/03/25/ppo-implementation-details/

# 16384

[<img src="assets/16384-step-6791.png" alt="26 mins to 16384">](https://github.com/tsangwpx/ml2048/issues/1)


There is rare chance (&lt;0.1%?) to obtain 16384 but the moves somehow seem surprisingly lucky.

# Remarks

1. The experience buffer is step-based instead of episode-based.
    There are `M` individual games and they are stepped `N` times in each epoch
    to obtain `MN` transitions.
    They are reset to the initial state when terminated.

2. Each transition `(S, A, R, S)` is used twice in current and next epoch
    to stabilize policy update.

3. Batch norm harmed the performance and was thus removed.

4. As [suggestion][1], weights are orthogonally initialized.
    All bias is initialized to zero, which deactivates ReLU by default when the input is sparse.
    Otherwise, the bias is likely a positive constant to the network
    until new number is seen (in one-hot encoding), which the network must greatly adjust itself.

5. The actor logits subtract their max values to avoid `+inf` and `NaN`.

6. Advantage is normalized to mostly fit into `[-1, 1]` range
    to avoid extreme values and clipping.

    It does not sound good when subtracting the mean from the advantage.
    The standard deviation is computed under the assumption of zero mean instead.

7. Entropy coefficient is reduced before the mid game, which may not be critical.

8. The depthwise convolution outputs more channels than input, and there is likely more computation,
    compared to the usual depthwise-separable convolution.

9. From practice, it is good idea to maintain the magnitude of entropy loss two order (base 10)
    less than policy loss.

10. The peaks (9k, 18k, 27k, 49k, etc) in the max tile distribution chart were caused
    by applying different hyper-parameters.

    In epoch 49k, potential-based reward shaping is used to grant extra bonus to the top left cell.
    Before that, the model preferred the largest tile located in the second to the top left cell.
    The agent adapted the new reward scheme and achieve better max tile.

11. Evaluating agent performance in step-based training with vectorized environment is tricky.

    Firstly, the trajectory in training is partial. 
    Secondly, the first `N` terminated games is not the first `N` games.
    Incorrect implementation may cause bias toward early ended games.