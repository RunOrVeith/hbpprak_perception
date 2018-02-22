

# Thimblerigger Challenge solution Group 5

This repository solves the "Thimblerigger" perception challenge.
It is part of the KIT course "Virtual Neurorobotics".

## Installing

Assuming that you have a local install of the [NRP](https://bitbucket.org/hbpneurorobotics/neurorobotics-platform),
no special setup is needed. Simply fork this repo into the **Experiments** folder.
When starting the NRP frontend, you should now see an experiment called __PerceptionChallengeKIT__.

## Running the solution

Assuming you have the NRP running,
start the ipython notebook `solution.ipynb` with the virtual coach:
```bash
cle-virtual-coach jupyter notebook
```

A Jupyter notebook will open in your browser. Just run all cells.
The last cell is an interactive stepper through the challenge.
You can now join the simulation in the front-end to see what is going on.
Just focus the input prompt that opens in the notebook and press enter to go to the next stage.
Don't worry if it doesn't start right away, it takes a while.

The robot will move its arms according to the prediction of the correct mug:

- Left arm up: Predict the ball is on the left
- Right arm up: Predict the ball is on the right
- Both arms up: Predict the ball is in the middle



## Additional Information
   The master branch contains a valid solution.
   The branches 'retina' and 'neural-track' contain other attempts to solve the challenge that did not end up working.

 - [Demo video](https://youtu.be/aice0elP7eI)
 - [NRP](https://bitbucket.org/hbpneurorobotics/neurorobotics-platform)
 - [NRP Forum](https://forum.humanbrainproject.eu/)
 - [Rospy service clients](http://wiki.ros.org/ROS/Tutorials/WritingServiceClient%28python%29#rospy_tutorials.2BAC8-Tutorials.2BAC8-WritingServiceClient.Writing_the_Client_Node)
