# L2RPN
The fourth place solution for Learning to Run a Power Network - ICAPS 2021 Competition

[Competition Link](https://competitions.codalab.org/competitions/33121)
## Approach
For the operational part, we train a DDQN model to select an appropriate action from the action space. When we have a overflow in power network. If there is not appropriate action, we will choose action from the additional action space.

For the alarm part, we use flow entrophy of power network to decide whether or not to trigger the alarm in the current timestep.
## License
This content is released under the [Mozilla Public License (MPL) v2.0](https://www.mozilla.org/en-US/MPL/2.0/).
