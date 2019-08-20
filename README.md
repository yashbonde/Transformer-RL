# Transformer-RL
Experiments to train transformer network to master reinforcement learning environments. Before starting however I have
written a **small PoC in `pytorch`** to train a transformer network on RAM configurations of popular `gym` environments.
This was made as a quick hack and to learn `pytorch` package, however not a big fan of it, maybe I am just more used to
`tensorflow` (more details in README). This package will completely be based upon and use `tensorflow` as our NN package.

A bit more about the coding conventions used here, I like the OpenAI style of coding where everything is tried to be
mprphed into a function, this is more of a scripting convention and easier for experimentation and deployment. Plus
it makes it easier on the eyes to read and maintain the code, increasing reusability. Just look at the repos from
Google, its almost impossible to read and hack that code. 