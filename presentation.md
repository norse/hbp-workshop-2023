---
title: "Norse: Machine Learning with SNN"
author:
    - 
        name: Christian Pehle
        affiliation: Heidelberg University
    - 
        name: Jens Pedersen
        affiliation: KTH Stockholm
date: January 18, 2023
theme: 'white'
slideNumber: True
center: False
width: 1920
---

## What is Norse?

. . .

::: {.f1 .green}
Spiking Neuron Primitives in PyTorch
:::

##

:::: {.flex .flex-column .flex-row-l .flex-wrap-l .mt5}

::: {.fl .w-33 .tc .calibre .normal .f2 .mt3}
::: {.mw7}
![](triangle_v.png){.h5 .rotate-90}

Blazing fast 

:::
:::



::: {.fl .w-33 .tc .calibre .normal .f2 .mt3}
::: {.mw7}
![](square_g.png){.h5 .rotate-90}

Feels amazing to use

:::
:::

::: {.fl .w-33 .tc .calibre .normal .f2 .mt3}
::: {.mw7}
![](circle_t.png){.h5 .rotate-90}

Quick to get started
:::
:::

::::


##

:::: {.flex .flex-column .flex-row-l .flex-wrap-l}

::: {.fl .w-25 .tc .calibre .normal .f}

![](triangle_v.png){.h5 .rotate-90}

Blazing fast

:::

::: {.fl .w-75 .tc .calibre .normal .f1}


::: {.incremental .lh-copy .mt5 .fl .pl5}
- PyTorch already takes care of most of the work
- Implementation is performance aware
- Custom code for selected operations
::: 


:::
:::


##

:::: {.flex .flex-column .flex-row-l .flex-wrap-l}

::: {.fl .w-25 .tc .calibre .normal .f}

![](square_g.png){.h5 .rotate-90}

Feels amazing to use

:::

::: {.fl .w-75 .tc .calibre .normal .f1}


::: {.incremental .lh-copy .mt5 .fl .pl5 }
- Complexity is revealed gradually
- Sensible defaults
- Customisable by experts
::: 


:::
:::


##

:::: {.flex .flex-column .flex-row-l .flex-wrap-l}

::: {.fl .w-25 .tc .calibre .normal .f}

![](circle_t.png){.h5 .rotate-90}

Quick to get started

:::

::: {.fl .w-75 .tc .calibre .normal .f1}

::: {.fl .ph5}
Install the PIP or conda package.
:::

::: {.incremental .lh-copy .mt5 .fl .ph5}
- Try out one of the [jupyter notebooks](https://norse.github.io/notebooks/)
- Read our [interactive documentation](https://norse.github.io/norse/)
- End-to-end [example tasks](https://github.com/norse/norse/tree/main/norse/task)
::: 


:::
:::





## Event-Streaming

## Deep Learning with Norse

## 

:::: {.flex .flex-column .flex-row-l .flex-wrap-l .mt5}

::: {.fl .w-33 .tc .calibre .normal .f2 .mt3}
::: {.mw7}
![](triangle_c.png){.h5 .rotate-90}

Familiar to ML researchers

:::
:::



::: {.fl .w-33 .tc .calibre .normal .f2 .mt3}
::: {.mw7}
![](hexagon_r.png){.h5 .rotate-90}

Compose primitives freely

:::
:::

::: {.fl .w-33 .tc .calibre .normal .f2 .mt3}
::: {.mw7}
![](circle_y.png){.h5 .rotate-90}

Extensible by everyone
:::
:::

::::

## 

:::: {.flex .flex-column .flex-row-l .flex-wrap-l}

::: {.fl .w-25 .tc .calibre .normal}

![](triangle_c.png){.rotate-90 .h5}

Familiar to ML researchers

:::
::: {.fl .w-75 .tc .calibre .normal .f1}

::: {.incremental .lh-copy .mt5 .fl .ph5}
- Build models like you would in PyTorch
- Use dataloaders and training frameworks as is
::: 

:::
:::


## 

:::: {.flex .flex-column .flex-row-l .flex-wrap-l}

::: {.fl .w-25 .tc .calibre .normal .f}

![](hexagon_r.png){.h5 .rotate-90}

Compose primitives freely

:::
::: {.fl .w-75 .tc .normal .f1}
```{.python data-line-numbers="3,8-10"}
import torch, torch.nn as nn

from norse.torch import LICell
from norse.torch import LIFCell
from norse.torch import SequentialState

model = SequentialState(
    nn.Conv2d(1, 20, 5, 1),                # convolve from 1 -> 20 channels
    LIFCell(),                             # leaky-integrate and fire neurons
    nn.MaxPool2d(2, 2),
    nn.Conv2d(20, 50, 5, 1),               # convolve from 20 -> 50 channels
    LIFCell(),
    nn.MaxPool2d(2, 2),
    nn.Flatten(),                          # flatten to 800 units
    nn.Linear(800, 10),
    LICell(),                              # leaky-integrator
)
```


:::
:::



## 

:::: {.flex .flex-column .flex-row-l .flex-wrap-l}

::: {.fl .w-25 .tc .calibre .normal .f}

![](circle_y.png){.h5 .rotate-90}

Extensible by everyone

:::
::: {.fl .tl .w-75 .tc .normal .f1}


::: {.fl .lh-copy .ph5}
Join us on [github](https://github.com/github/) and [discord](https://discord.gg/7fGN359)!
:::

::: {.incremental .lh-copy .mt5 .fl .ph5}
- Implementation is (mostly) in Python
- Simple neuron models can be added quickly
- *Open* and *welcoming* community.
::: 

:::
:::



# Workshop

## 

:::: {.flex .flex-column .flex-row-l .flex-wrap-l}

::: {.fl .w-50 .calibre .tc .normal .f1}

![](triangle_g.png){.h6 .rotate-90}

Event streaming

:::
::: {.fl .w-50 .calibre .tc .normal .f1}

![](circle_y.png){.h6}

Deep Learning

:::
:::

## Schedule

::: {.incremental}
- 9:00 - 9:30 
    - Presentation + Questions (<- we are here)
- 9:30 - 10:30
    - ![](triangle_g.png){.h2 .v-mid .rotate-90} Event streaming (Jens)
    - ![](circle_y.png){.h2 .v-mid} Deep learning (Christian)
- 10:30 - 11:30
    - ![](triangle_g.png){.h2 .v-mid .rotate-90} Event streaming (Jens)
    - ![](circle_y.png){.h2 .v-mid} Deep learning (Christian)
:::


## Workshop 1: Event streaming

## Goals




## Workshop 2: Deep Learning

## Goals



# Let's get started!

## Funding & Acknowledgements


::: { .tl .gray}

The research has received funding from the EC Horizon 2020 Framework Programme under Grant Agreements 785907 and 945539 (HBP) and by the Deutsche Forschungsgemeinschaft (DFG, German Research Fundation) under Germany's Excellence Strategy EXC 2181/1 - 390900948 (the Heidelberg STRUCTURES Excellence Cluster).

:::

![](hexagon_r.png){.h4 .rotate-90} ![](triangle_c.png){.h4 .rotate-90}  ![](circle_y.png){.h4 .rotate-90}

::: { .tl .gray}


Icons by Thought Machine

© 2022 Thought Machine

Released under [CC-BY-v4](https://creativecommons.org/licenses/by/4.0/)
:::