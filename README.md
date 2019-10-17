# Error rate control in statistical significance testing

[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/Pascal-Bliem/error-control-statistical-tests/master?filepath=Error_control_in_statistics.ipynb)

This project is a discussion on proper error rate control in statistical hypothesis significance testing. Understanding these aspects is crucial to ensure that the results of statistical inference are actually meaningful and that the potential error does not get bigger than what the experimenter deems as acceptable. I simulated hundreds of thousands of statistical tests to show how the results are distributed both if there is a real effect to be observed and if there is none. The results of these simulations are displayed in interactive visualizations in which you can change the parameters to see how errors can be  inflated in certain situations and how they can be controlled.

<p align="center">
<img src="https://chemicalstatistician.files.wordpress.com/2014/05/pregnant.jpg" alt="errors" height="300"/>  
</p>
<p align="center">
<em>An example of errors in statistical tests: False positives and false negatives.</em> 
</p>

## How to view this project

If you **do not** have Python and Jupyter installed on your computer or you don't want to download anything, no problem at all! Just [click here](https://mybinder.org/v2/gh/Pascal-Bliem/error-control-statistical-tests/master?filepath=Error_control_in_statistics.ipynb) to launch an interactive live environment on mybinder.org. The site will take a few seconds to load. To activate the interactive visualizations, please click on the menu tab *"Kernel"* and then on *"Restart & Run all"* as shown in the image below. You can also just press the small button with the fast-forward symbol on it.

<p align="center">
<img src="https://i.stack.imgur.com/neKoy.png" alt="errors" height="200"/>  
</p>
<p align="center">
<em>Click "Kernel" --> "Restart & Run all" so that you can use the interactivity.</em> 
</p>

If you **do** have Python, Jupyter, and all [dependencies](./environment.yml) installed, you can just download the repository and run it on your local machine. The main project is contained in a [Jupyter notebook](./Error_control_in_statistics.ipynb) but you will also need the utility scripts in this repo, and the already simulated data in the [./data](./data) directory may come in handy.

I hope you'll enjoy reading it, thanks a lot for your interest!
