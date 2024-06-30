---
layout: distill
title: Equivariant Diffusion for Molecule Generation in 3D using Consistency Models
description: <p> Introduction to the seminal papers &quot;Equivariant Diffusion for Molecule Generation in 3D&quot; and &quot;Consistency Models&quot; with an adaptation fusing the two together for fast molecule generation. </p> 
tags: equivariance, diffusion, molecule generation, consistency models
giscus_comments: true
date: 2024-06-30
featured: true

authors:
  - name: Martin Sedlacek*
    url: https://martin-sedlacek.com/
    affiliations:
      name: University of Amsterdam
  - name: Antonios Vozikis*
    url: "#"
    affiliations:
      name: Vrije Universiteit Amsterdam

bibliography: equivariant_diffusion/2024-06-30-equivariant_diffusion.bib

toc:
  - name: Introduction
  - name: Equivariant Diffusion Models (EDM)
  - name: Enhancements with Consistency Models
  - name: Conclusion


_styles: >
  .fake-img {
    background: #bbb;
    border: 1px solid rgba(0, 0, 0, 0.1);
    box-shadow: 0 0px 4px rgba(0, 0, 0, 0.1);
    margin-bottom: 12px;
  }
  .fake-img p {
    font-family: monospace;
    color: white;
    text-align: left;
    margin: 12px 0;
    text-align: center;
    font-size: 16px;
  }
---

## Introduction


In this blog post, we introduce and discuss ["Equivariant Diffusion for Molecule Generation in 3D"](https://arxiv.org/abs/2203.17003) <d-cite key="hoogeboom2022equivariant"></d-cite>, 
which first introduced 3D molecule generation using diffusion models. Their Equivariant Diffusion Model (EDM) also
incorporating an Equivariant Graph Neural Network (EGNN) architecture, effectively grounding the model with inductive
priors about the symmetries in 3D space. This work demonstrated strong improvement over other (non-diffusion) generative 
methods for molecules at the time, and inspired many subsequent works <d-cite key="anstine2023generative"></d-cite><d-cite key="corso2023diffdock"></d-cite><d-cite key="igashov2024equivariant"></d-cite><d-cite key="xu2023geometric"></d-cite>. 

Traditional diffusion is unfortunately bottle-necked by the sequential denoising process, which can be slow and computationally expensive <d-cite key="song2023consistency"></d-cite>.
Hence, we also aim to demonstrate that an EDM can be trained significantly faster, uncapping its potential, with the 
following two extensions:
<
1. Training EDM as a Consistency Model <d-cite key="song2023consistency"></d-cite>
2. Faster implementation of the EDM with JAX <d-cite key="bradbury2018jax"></d-cite>

Consistency models enable generating samples down to a single step, while the JAX framework has been shown to improve performance 
of models that require regular, repetitive computations such as diffusion models. It also demonstrated up to 5x improvement 
on a comparable diffusion model <d-cite key="kang2024efficient"></d-cite>.

This increase in efficiency can serve several purposes. Most importantly, increased performance enables the use of larger
models without increased need for compute. Many previous works across various domains have shown that scaling model
architectures to more parameters can significantly improve performance <d-cite key="dosovitskiy2020image"></d-cite><d-cite key="kaplan2020scaling"></d-cite><d-cite key="krizhevsky2012imagenet"></d-cite> 
in domains including language <d-cite key="brown2020language"></d-cite><d-cite key="kaplan2020scaling"></d-cite><d-cite key="touvron2023llama"></d-cite>
as well as images and video <d-cite key="liu2024sora"></d-cite><d-cite key="ramesh2022hierarchical"></d-cite><d-cite key="rombach2022high"></d-cite><d-cite key="saharia2022photorealistic"></d-cite>. 
A similar scaling effect was also observed in Graph Neural Networks (GNN) <d-cite key="sriram2022towards"></d-cite>.
We hope that, by improving the inference and training speed of these models, we can enable the use of larger GNN backbones 
without requiring more expensive compute. As a side effect, increasing the model speed also hastens development and decreases
the overall carbon footprint.

<!--- We also note that these performance improvements are, in theory at least, not exclusive to EDM or GNNs. Many other ML 
models might be improved through a JAX reimplementation and many diffusion models can be trained as a consistency model. --->

## Preliminary Concepts

#### Groups and Equivariance for molecules

Equivariance is a property of certain functions, which ensures that the function's output transforms in a predictable manner under collections of transformations. This property is valuable in molecular modeling, where it can be used to ensure that the properties of molecular structures are consistent with their symmetries in the real world. specifically, we are interested in ensuring that some structure is preserved in the representation of the molecule when three types of transformations are applied to it: translation, rotation, and reflection. 

Formally, function $f$ is said to be equivariant to the action of a group $G$ if: 

$$T_g(f(x)) = f(S_g(x)) \qquad \text{(1)}$$ 

for all $g ∈ G$, where $S_g,T_g$ are linear representations related to the group element $g$ <d-cite key="serre1977linear"></d-cite>. The three transformations we are interested in form the Euclidean group $E(3)$, for which $S_g$ and $T_g$ can be represented by a translation $t$ and an orthogonal matrix $R$ that rotates or reflects coordinates. $f$ is then equivariant to a rotation or reflection $R$ if: 

$$Rf(x) = f(Rx) \qquad \text{(2)}$$

meaning transforming its input results in an equivalent transformation of its output. <d-cite key="hoogeboom2022equivariant"></d-cite>

#### E(n) Equivariant Graph Neural Networks (EGNNs)


Generating molecules naturally leans itself into graph representation, with the nodes representing atoms within the
molecules, and edges representing their bonds. The features $\mathbf{h}_i \in \mathbb{R}^d$ of each atom, such as
element type, can then be encoded into the embedding of a node alongside it's position $\mathbf{x}_i \in \mathbb{R}^3$. The previously explained E(3) equivariance property can be used as an inductive prior that improves generalization, and EGNNs are a powerful tool which injects these priors about molecules into the model architecture itself, as the EDM paper had demonstrated <d-cite key="hoogeboom2022equivariant"></d-cite>. Their usefulness is further supported
by EGNNs beating similar non-equivariant Graph Convolution Networks on molecular generation tasks <d-cite key="verma2022modular"></d-cite>.

The E(n) EGNN is a special type of message-passing Graph Neural Network (GNN) <d-cite key="gilmer2017neural"></d-cite> with explicit rotation and translation equivariance baked in. A traditional message-passing GNN consists of several layers, each of which
updates the representation of each node, using the information in nearby nodes.

<!-- <p align="center">
  <img src="readme_material/message_passing.png" alt="Diffusion in nature" width="300" />
</p>
<p align="center">
Figure 1: Visualization of a message passing network (Credit: Yuki Asano)
</p> -->
<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
        <figure>
            {% include figure.liquid loading="eager" path="assets/img/2024-06-30-equivariant_diffusion/message_passing.png" class="img-fluid rounded z-depth-1" zoomable=true %}
            <figcaption class="text-center mt-2">Figure 1: Message passing</figcaption>
        </figure>
    </div>
</div>


The EGNN specifically contains _equivariant_ convolution layers:

$$
\mathbf{x}^{l+1},\mathbf{h}^{l+1}=EGCL[ \mathbf{x}^l, \mathbf{h}^l ] \qquad \text{(3)}
$$


The EGCL layer is defined through the formulas:


<div align="center">

$$
\mathbf{m}_{ij} = \phi_e(\mathbf{h}_i^l, \mathbf{h}_j^l, d^2_{ij}) \qquad \text{(4)}
$$

$$
\mathbf{h}_i^{l+1} = \phi_h\left(\mathbf{h}_i^l, \sum_{j \neq i} \tilde{e}_{ij} \mathbf{m}_{ij}\right) \qquad \text{(5)}
$$

$$
\mathbf{x}_i^{l+1} = \mathbf{x}_i^l + \sum_{j \neq i} \frac{\mathbf{x}_i^l \mathbf{x}_j^l}{d_{ij} + 1} \phi_x(\mathbf{h}_i^l, \mathbf{h}_j^l, d^2_{ij}) \qquad \text{(6)}
$$

</div>



where $h_l$ represents the feature $h$ at layer $l$, $x_l$ represents the coordinate at layer $l$ and 
$$d_{ij}= ||x_i^l-x^l_j||_2$$ is the Euclidean distance between nodes $$v_i$$ and $$v_j$$. A fully connected neural 
networks is used to learn the functions $$\phi_e$$, $$\phi_x$$, and $$\phi_h$$. At each layer, a message $$m_{ij}$$ 
is computed from the previous layer's feature representation. Using the previous feature and the sum of these messages, 
the model computes the next layer's feature representation.

This architecture then satisfies translation and rotation equivariance. Notably, the messages depend on the distance 
between the nodes and these distances are not changed by isometric transformations.



## Equivariant Diffusion Models

Diffusion models <d-cite key="sohl2015deep"></d-cite> are deeply rooted within the principles of physics, where the process describes particles moving 
from an area of higher concentration to an area of lower concentration - a process governed by random, stochastic 
interactions. In the physical world, this spreading can largely be traced back to the original configuration, which 
inspired scientists to create models of this behaviour. When applied to generative modelling, we usually aim to reconstruct data from some observed or sampled noise, which is an approach adopted by many powerful diffusion models. 

<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
        <figure class="custom-figure">
            {% include figure.liquid loading="eager" path="assets/img/2024-06-30-equivariant_diffusion/Diffusion_microscopic.gif" class="img-fluid rounded z-depth-1 custom-image" zoomable=true %}
            <figcaption class="text-center mt-2">Physical diffusion</figcaption>
        </figure>
    </div>
    <div class="col-sm mt-3 mt-md-0">
        <figure class="custom-figure">
            {% include figure.liquid loading="eager" path="assets/img/2024-06-30-equivariant_diffusion/Diffusion_models_flower.gif" class="img-fluid rounded z-depth-1 custom-image" zoomable=true %}
            <figcaption class="text-center mt-2">Generative modelling with diffusion</figcaption>
        </figure>
    </div>
</div>
<div class="row">
    <div class="col text-center mt-3">
        <p>Figure 2: Physical diffusion (left) and generative modelling with diffusion (right)</p>
    </div>
</div>

<style>
    .custom-figure .custom-image {
        height: 200px; /* Set a fixed height for both images */
        width: auto; /* Maintain aspect ratio and adjust width accordingly */
        max-width: 100%; /* Ensure the image doesn't exceed the container width */
    }
</style>






### Denoising Diffusion Probabilistic Models (DDPM)

One of the most widely-used and powerful diffusion models is the Denoising Diffusion Probabilistic Model (DDPM) <d-cite key="ho2020denoising"></d-cite>. In this model, the data is progressively noised and then the model learns to reverse this process, effectively "denoising". This process allows us to generate new samples from pure noise.

### Forward diffusion process ("noising")



In DDPMs the forward noising process is parameterized by a Markov process, where transition at each time step $t$ adds
Gaussian noise with a variance of $\beta_t \in (0,1)$. We formally write this transition as:

$$
\begin{align}
q\left( x_t \mid x_{t-1} \right) := \mathcal{N}\left( x_t ; \sqrt{1-\beta_t} x_{t-1}, \beta_t \mathbf{I} \right) & \qquad \text{(7)}
\end{align}
$$

The whole Markov process leading to time step $T$ is given as a chain of these transitions:

$$
\begin{align}
q\left( x_1, \ldots, x_T \mid x_0 \right) := \prod_{t=1}^T q \left( x_t \mid x_{t-1} \right) & \qquad \text{(8)}
\end{align}
$$

<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
        <figure>
            {% include figure.liquid loading="eager" path="assets/img/2024-06-30-equivariant_diffusion/ddpm_figure.png" class="img-fluid rounded z-depth-1" zoomable=true %}
            <figcaption class="text-center mt-2">Figure 3: The Markov process of forward and reverse diffusion <d-cite key="ho2020denoising"></d-cite></figcaption>
        </figure>
    </div>
</div>



### Reverse diffusion process ("denoising")



As Figure 3 shows, the reverse transitions are unknown, hence DDPM approximates them using a neural network 
parametrized by $\theta$:

$$p_\theta \left( x_{t-1} \mid x_t \right) := \mathcal{N} \left( x_{t-1} ; \mu_\theta \left( x_t, t \right), \Sigma_\theta \left( x_t, t \right) \right) \qquad \text{((9))}$$

Because we know the dynamics of the forward process, we know the variance at each $t$. Therefore, we can fix $\Sigma_\theta \left( x_t, t \right)$ to be $\beta_t \mathbf{I}$.

The network prediction is then only needed to obtain $\mu_\theta \left( x_t, t \right)$, given by:

$$\mu_\theta \left( x_t, t \right) = \frac{1}{\sqrt{\alpha_t}} \left( x_t - \frac{\beta\_t}{\sqrt{1 - \bar{\alpha}\_t}} \epsilon\_\theta \left( x_t, t \right) \right) \qquad \text{(10)}$$

where $\alpha_t = \Pi_{s=1}^t \left( 1 - \beta_s \right)$.

Hence, we can directly predict $x_{t-1}$ from $x_{t}$ using the network $\theta$ as:

$$
\begin{align}
x_{t-1} = \frac{1}{\sqrt{1 - \beta_t}} \left( x_t - \frac{\beta_t}{\sqrt{1 - \alpha_t}} \epsilon_\theta \left( x_t, t \right) \right) + \sqrt{\beta_t} v_t & \qquad \text{((11))}
\end{align}
$$

where $v_T \sim \mathcal{N}(0, \mathbf{I})$ is a sample from the pure Gaussian noise.

### Training diffusion models

The training objective of diffusion-based generative models amounts to **"maximizing the log-likelihood of the sample generated (at the end of the reverse process) which belongs to the original data distribution."**

To maximize the log-likelihood of a gaussian distribution, we need to try and find the parameters of the distribution (μ, σ²) such that it maximizes the _likelihood_ of the (generated) data belonging to the same data distribution as the original data.

To train our neural network, we define the loss function (L) as the objective function’s negative. So a high value for p_θ(x₀), means low loss and vice versa.

$$
p_{\theta}(x_{0}) := \int p_{\theta}(x_{0:T})dx_{1:T} \qquad \text{(12)}
$$

$$
L = -\log(p_{\theta}(x_{0})) \qquad \text{(13)}
$$

However, this is intractable because we need to integrate over a very high dimensional (pixel) space for continuous values over T timesteps. Instead, take inspiration from VAEs and find a new, tractable training objective using a variational lower bound (VLB), also known as _Evidence lower bound_ (ELBO). We have :

$$
\mathbb{E}[-\log p_{\theta}(x_{0})] \leq \mathbb{E}_{q} \left[ -\log \frac{p_{\theta}(x_{0:T})}{q(x_{1:T} | x_{0})} \right] = \mathbb{E}_{q} \left[ -\log p(X_{T}) - \sum_{t \geq 1} \log \frac{p_{\theta}(x_{t-1} | x_{t})}{q(x_{t} | x_{t-1})} \right] =: L \qquad \text{(14)}
$$

After some simplification, we arrive at this final $$L_{vlb}$$ - Variational Lower Bound loss term:

$$
\mathbb{E}_{q} \left[ D_{KL}(q(x_{T}|x_{0}) \parallel p(x_{T})) \bigg\rvert_{L_{T}} + \sum_{t > 1} D_{KL}(q(x_{t-1}|x_{t}, x_{0}) \parallel p_{\theta}(x_{t-1}|x_{t})) \bigg\rvert_{L_{t-1}} - \log p_{\theta}(x_{0}|x_{1}) \bigg\rvert_{L_{0}} \right] \qquad \text{(15)}
$$

We can break the above $$L_{vlb}$$ loss term into individual timesteps as follows:

$$
L_{vlb} := L_{0} + L_{1} + \cdots + L_{T-1} + L_{T} \qquad \text{(16)}
$$

$$
L_{0} := - \log p_{\theta}(x_{0}|x_{1}) \qquad \text{(17)}
$$

$$
L_{t-1} := D_{KL}(q(x_{t-1}|x_{t}, x_{0}) \parallel p_{\theta}(x_{t-1}|x_{t})) \qquad \text{(18)}
$$

$$
L_{T} := D_{KL}(q(x_{T}|x_{0}) \parallel p(x_{T})) \qquad \text{(19)}
$$

The terms ignored are:

1. **L₀** – Because the original authors got better results without this.
2. **Lₜ** – This is the _"KL divergence"_ between the distribution of the final latent in the forward process and the first latent in the reverse process. Because there are no neural network parameters we can't do anything with it so we just ignore from optimization.

So **Lₜ₋₁** is the only loss term left which is a KL divergence between the _“posterior”_ of the forward process, and the parameterized reverse diffusion process. Both terms are gaussian distributions as well.

$$
L_{vlb} := L_{t-1} := D_{KL}(q(x_{t-1}|x_{t}, x_{0}) \parallel p_{\theta}(x_{t-1}|x_{t})) \qquad \text{(20)}
$$

The term q(xₜ₋₁|xₜ, x₀) is referred to as _“forward process posterior distribution.”_

During training, our DL model learns to approximate the parameters of this posterior in order to minimize the KL divergence.

<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
        <figure>
            {% include figure.liquid loading="eager" path="assets/img/2024-06-30-equivariant_diffusion/diffusion_training.gif" class="img-fluid rounded z-depth-1" zoomable=true %}
            <figcaption class="text-center mt-2">Figure 4: Stochastic sampling process (noisy images on top, predicted images on bottom)</figcaption>
        </figure>
    </div>
</div>



## Equivariant Diffusion Models (EDM) for 3D molecule generation

We will specifically focus on an E(n) equivariant diffusion model presented by the EDM paper, 
which specialized in 3D molecular generation <d-cite key="hoogeboom2022equivariant"></d-cite>. The authors used a DDPM-based diffusion model with an EGNN backbone 
for predicting both continuous (atom coordinates) and categorical features (atom types).

As we have hinted earlier in the EGNN section, molecules are naturally equivariant to E(3) rotations and translation 
while being easily represented with a graph. By E(3) we refer to the Euclidean group in three dimensions, which includes all transformations (rotations, translations, and reflections) that preserve Euclidean distances in three-dimensional space.
 The categorical atomic properties are already invariant to E(3) 
transformations, hence they can be generated with a regular diffusion. For the generated atom positions however, 
we need to specifically ensure this equivariance to rotations and translations throughout the diffusion process.

### How to achieve equivariance during diffusion?

**Rotations**

Being equivariant to rotations, effectively meant that we want rotations applied at any given time step $t$ 
in the diffusion process to not have an effect on the likelihood of generating a corresponding rotated sample at the 
next time step $t+1$. In other words, if the best prediction is to move position of each atom $a_i$ in a certain 
diretion $\mathbf{v}_i$, after we rotate the whole molecule by some arbitrary rotation matrix $\mathbf{R}$, the 
equivariantly rotated predictions $\mathbf{R}\mathbf{v}_i$ should still be the best one to pick.

Formally, we say that for any orthogonal rotation matrix $\mathbf{R}$ the following must hold:

$$p(y|x) = p(\mathbf{R}y|\mathbf{R}x) \qquad \text{(21)}$$


To uphold this property throughout the diffusion process, the Markov chain transition probability distributions at every 
time step $t$ must be roto-invariant, otherwise rotations would alter the likelihood, breaking this desired equivariance property.

<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
        <figure class="custom-figure">
            {% include figure.liquid loading="eager" path="assets/img/2024-06-30-equivariant_diffusion/roto_symetry_gaus.png" class="img-fluid rounded z-depth-1 custom-image" zoomable=true %}
            <figcaption class="text-center mt-2">Diffusion in nature</figcaption>
        </figure>
    </div>
    <div class="col-sm mt-3 mt-md-0">
        <figure class="custom-figure">
            {% include figure.liquid loading="eager" path="assets/img/2024-06-30-equivariant_diffusion/roto_symetry_donut.png" class="img-fluid rounded z-depth-1 custom-image" zoomable=true %}
            <figcaption class="text-center mt-2">Diffusion in model</figcaption>
        </figure>
    </div>
</div>
<div class="row">
    <div class="col text-center mt-3">
        <p>Figure 5: Examples of 2D roto-invariant distributions</p>
    </div>
</div>

<style>
    .custom-figure .custom-image {
        height: 250px; /* Set a fixed height for both images */
        width: auto; /* Maintain aspect ratio and adjust width accordingly */
        max-width: 100%; /* Ensure the image doesn't exceed the container width */
    }
</style>



As the EDM authors point out, an invariant distribution composed with an equivariant invertible function results in an
invariant distribution <d-cite key="kohler2020equivariant"></d-cite>. They further point out that if $x \sim p(x)$ is invariant to a group, and the transition probabilities
of a Markov chain $y \sim p(y|x)$ are equivariant, then the marginal distribution of $y$ at any time step $t$ is also invariant
to that group <d-cite key="xu2022geodiff"></d-cite>.

In the case of EDM, the underlying EGNN ensures equivariance while the initial sampling distribution
can easily be constrained to something roto-invariant, such as a simple mean zero Gaussian with diagonal
covariance matrix seen in Figure 5 (left).

**Translations**

It has been shown, that it is impossible to have non-zero distributions invariant to translations <d-cite key="satorras2021en"></d-cite>.
Intuitively, the translation invariance property means that any point $\mathbf{x}$ results in the same assigned $p(\mathbf{x})$,
leading to a uniform distribution, which, if stretched over an unbounded space, would be approaching zero-valued probabilities
thus not integrating to one.

The EDM authors bypass this with a clever trick of always re-centering the generated samples to have center of gravity at
$\mathbf{0}$ and further show that these $\mathbf{0}$-centered distributions lie on a linear subspace that can reliably be used 
for equivariant diffusion <d-cite key="hoogeboom2022equivariant"></d-cite><d-cite key="xu2022geodiff"></d-cite>. 





We hypothesize that, intuitively, moving a coordinate from e.g. 5 to 6 on any given axis is the same as moving from 
8 to 9. But EDM predicts the actual atom positions, not a relative change, hence the objective needs to adjusted. 
By constraining the model to this "subspace" of options where the center of the molecule is always at $\mathbf{0}$, 
the absolute positions are effectively turned into relative ones w.r.t. to the center of the molecule, hence the model 
can now learn relationships that do not depend on the absolute position of the whole molecule in 3D space.


### Training the EDM

As described in the DDPM section and following other modern diffusion models, an EGNN is trained in the standard 
diffusion framework to predict the noise at each time step of the reverse process, which is then used to iteratively 
reconstruct samples on the data distribution by adding signal to pure noise sampled from the Gaussian at time $T$. 
With the caveat that the predicted noise must be calibrated to have center of gravity at $\mathbf{0}$ to ensure 
equivariance as we have described earlier. It is worth explicitly noting that the noise added during the forward 
diffusion process must also be equivariant, hence it is sampled from a $\mathbf{0}$-mean Gaussian distribution with 
a diagonal covariance matrix.

Using the KL divergence loss term introduced in DDPM with the EDM model parametrization simplifies the loss function to:

$$
\mathcal{L}_t = \mathbb{E}_{\epsilon_t \sim \mathcal{N}_{x_h}(0, \mathbf{I})} \left[ \frac{1}{2} w(t) \| \epsilon_t - \hat{\epsilon}_t \|^2 \right] \qquad \text{(22)}
$$








where 
$\( w(t) = \left(1 - \frac{\text{SNR}(t-1)}{\text{SNR}(t)}\right) \)$ and $\( \hat{\epsilon}_t = \phi(z_t, t) \)$.

However, the EDM authors found that the model had better empirical performance with a constant $w(t) = 1$, disregarding the
signal-to-noise ration (SNR). Thus, the loss term effectively simplifies to a MSE.

Since coordinates and categorical features are on different scales, the EDM authors also found they achieved better performance when scaling the inputs before prediction and then rescaling them back after.


### Consistency Models


Although diffusion Models have significantly advanced the fields of image, audio, and video generation, but they depend on an iterative de-noising process to generate samples, which can be very slow <d-cite key="song2023consistency"></d-cite>. To generate good samples, a lot of steps are often required (sometimes in the 1000s). This issue is exacerbated when dealing with high dimensional data where all operations are even more computationally expensive. As hinted in the introduction, we look at Consistency models in our work to bypass this bottleneck.

This is where Consistency Models really shine. This new family of models reduces the number of steps during de-noising up to just a single step generation, significantly speeding up this process, while allowing for a controlled trade-off between speed and sample quality.

### How does it work?

To understand consistency models, one must look at diffusion from a slightly different perspective than it's usually presented.
Consider the transfer of mass under the data probability distribution in time.

<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
        <figure>
            {% include figure.liquid loading="eager" path="assets/img/2024-06-30-equivariant_diffusion/bimodal_to_gaussian_plot.png" class="img-fluid rounded z-depth-1" zoomable=true %}
            <figcaption class="text-center mt-2">Figure 6: Illustration of a bimodal distribution evolving to a Gaussian over time</figcaption>
        </figure>
    </div>
</div>


Such process are often well described with a differential equation. In the next sections we look closely at the work of Yang Song <d-cite key="song2023consistency"></d-cite><d-cite key="song2021score"></d-cite> and others to examine how they leverage the existence of such an Ordinary Differential Equation (ODE) to generate strong
samples much faster.

<br>

**Modelling the noising process as an SDE**

Song et al. <d-cite key="song2021score"></d-cite> have shown that the noising process in diffusion can be described with a Stochastic Differential Equation (SDE)
transforming the data distribution $p_{\text{data}}(\mathbf{x})$:

$$d\mathbf{x}_t = \mathbf{\mu}(\mathbf{x}_t, t) dt + \sigma(t) d\mathbf{w}_t \qquad \text{(23)}$$

Where $t$ is the time-step, $\mathbf{\mu}$ is the drift coefficient, $\sigma$ is the diffusion coefficient,
and $\mathbf{w}_t$ is the stochastic component denoting standard Brownian motion. This stochastic component effectively
represents the iterative adding of noise to the data in the forward diffusion process and dictates the shape of the final
distribution at time $T$.

Typically, this SDE is designed such that $p_T(\mathbf{x})$ at the final time-step $T$ is close to a tractable Gaussian.

<br>

**Existence of the PF ODE**

This SDE has a remarkable property, that a special ODE exists, whose trajectories sampled at $t$ are distributed
according to $p_t(\mathbf{x})$ <d-cite key="song2023consistency"></d-cite>:

$$d\mathbf{x}_t = \left[ \mathbf{\mu}(\mathbf{x}_t, t) - \frac{1}{2} \sigma(t)^2 \nabla \log p_t(\mathbf{x}_t) \right] dt \qquad \text{(24)}$$

This ODE is dubbed the Probability Flow (PF) ODE by Song et al. <d-cite key="song2023consistency"></d-cite> and corresponds to the different view of diffusion
manipulating probability mass over time we hinted at in the beginning of the section.

A score model $s_\phi(\mathbf{x}, t)$ can be trained to approximate $\nabla log p_t(\mathbf{x})$ via score matching <d-cite key="song2023consistency"></d-cite>.
 <!-- and following Karras et al. <d-cite key="gilmer2017neural"></d-cite> it is -->
Since we know the parametrization of the final distribution $p_T(\mathbf{x})$ to be a standard Gaussian parametrized with $\mathbf{\mu}=0$ and $\sigma(t) = \sqrt{2t}$, this score model can be plugged into the equation (24) and the expression reduces itself to an empirical estimate of the PF ODE:

$$\frac{dx_t}{dt} = -ts\phi(\mathbf{x}_t, t) \qquad \text{(25)}$$

With $\mathbf{\hat{x}}_T$ sampled from the specified Gaussian at time $T$, the PF ODE can be solved backwards in time to obtain
a solution trajectory mapping all points along the way to the initial data distribution at time $\epsilon$.

<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
        <figure>
            {% include figure.liquid loading="eager" path="assets/img/2024-06-30-equivariant_diffusion/consistency_models_pf_ode.png" class="img-fluid rounded z-depth-1" zoomable=true %}
            <figcaption class="text-center mt-2">Figure 7: Solution trajectories of the PF ODE. <d-cite key="dosovitskiy2020image"></d-cite></figcaption>
        </figure>
    </div>
</div>


<br>

**Solving the PF ODE**

In Figure 7 the "Noise" distribution corresponds to $p_T(\mathbf{x})$ and the "Data" distribution is treated as one at $t=\epsilon$
very close to time zero. For numerical stability we want to avoid explicitly having $t=0$ <d-cite key="karras2022elucidating"></d-cite>.

Following Karras et al. <d-cite key="karras2022elucidating"></d-cite>, the time horizon $[\epsilon, T]$ is discretized into $N-1$ sub-intervals with
boundaries $t_1 = \epsilon < t_2 < \cdots < t_N = T$. This improves performance and stability over treating time as a
continuous variable.

In practice, the following formula is most often used to determine these boundaries <d-cite key="karras2022elucidating"></d-cite>:

$$t_i = \left(\epsilon^{1/\rho} + \frac{i - 1}{N - 1}(T^{1/\rho} - \epsilon^{1/\rho})\right)^\rho \qquad \text{(26)}$$

, where $\rho = 7$.

 Given any of-the-shelf ODE solver (e.g. Euler) and a trained score model $s_\phi(\mathbf{x}, t)$, we can solve this PF ODE.

As shown before, the score model is needed to predict the change in signal over time $\frac{dx_t}{dt}$, i.e. a
generalization of what we referred to as "predicting noise to next time step" earlier.

A solution trajectory, denoted $\\{\mathbf{x}_t\\}$, is then given as a finite set of samples $\mathbf{x}_t$ for every
discretized time-step $t$ between $\epsilon$ and $T$.

<br>

**Consistency Function**

Given a solution trajectory $${\mathbf{x}_t}$$, we define the _consistency function_ as:

<p align="center">
$f:$ $(\mathbf{x}_t, t)$ $\to$ $\mathbf{x}_{\epsilon}$ 
</p>

In other words, a consistency function always outputs a corresponding datapoint at time $\epsilon$, i.e. very close to
the original data distribution for every pair ($\mathbf{x}_t$, t).

Importantly, this function has the property of _self-consistency_: i.e. its outputs are consistent for arbitrary pairs of
$(x_t, t)$ that lie on the same PF ODE trajectory. Hence, we have $f(x_t, t) = f(x_{t'}, t')$ for all $t, t' \in [\epsilon, T]$.

The goal of a _consistency model_, denoted by $f_\theta$, is to estimate this consistency function $f$ from data by
being enforced with this self-consistency property during training.

<br>

**Boundary Condition & Function Parametrization**

For any consistency function $f(\cdot, \cdot)$, we must have $f(x_\epsilon, \epsilon) = x_\epsilon$, i.e., $f(\cdot, 
\epsilon)$ being an identity function. This constraint is called the _boundary condition_ <d-cite key="song2023consistency"></d-cite>.

The boundary condition has to be met by all consistency models, as we have hinted before that much of the training relies
on the assumption that $p_\epsilon$ is borderline identical to $p_0$. However, it is also a big architectural
constraint on consistency models.

For consistency models based on deep neural networks, there are two ways to implement this boundary condition almost
for free <d-cite key="song2023consistency"></d-cite>. Suppose we have a free-form deep neural network $F_\theta (x, t)$ whose output has the same dimensionality
as $x$.

1.) One way is to simply parameterize the consistency model as:

$$
f_\theta (x, t) =
\begin{cases}
x & t = \epsilon \\
F_\theta (x, t) & t \in (\epsilon, T]
\end{cases} \\
\qquad \text{(27)}
$$

2.) Another method is to parameterize the consistency model using skip connections, that is:

$$
f_\theta (x, t) = c_{\text{skip}} (t) x + c_{\text{out}} (t) F_\theta (x, t) \qquad \text{(28)}
$$

where $c_{\text{skip}} (t)$ and $c_{\text{out}} (t)$ are differentiable functions such that $c_{\text{skip}} (\epsilon) = 1$,
and $c_{\text{out}} (\epsilon) = 0$.

This way, the consistency model is differentiable at $t = \epsilon$ if $F_\theta (x, t)$, $c_{\text{skip}} (t)$, $c_{\text{out}} (t)$
are all differentiable, which is critical for training continuous-time consistency models.

In our work, we utilize the latter methodology in order to satisfy the boundary condition.

<br>

**Sampling**

With a fully trained consistency model $f_\theta(\cdot, \cdot)$, we can generate new samples by simply sampling from the initial
Gaussian $\hat{x_T}$ $\sim \mathcal{N}(0, T^2I)$ and propagating this through the consistency model to obtain
samples on the data distribution $\hat{x_{\epsilon}}$ $= f_\theta(\hat{x_T}, T)$ with as little as one diffusion step.

<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
        <figure>
            {% include figure.liquid loading="eager" path="assets/img/2024-06-30-equivariant_diffusion/consistency_on_molecules.png" class="img-fluid rounded z-depth-1" zoomable=true %}
            <figcaption class="text-center mt-2">Figure 8: Visualization of PF ODE trajectories for molecule generation in 3D. <d-cite key="fan2023ecconf"></d-cite></figcaption>
        </figure>
    </div>
</div>


### Training Consistency Models


Consistency models can either be trained by "distillation" from a pre-trained diffusion model, or in "isolation" as a standalone generative model from scratch. In the context of our work, we focused only on the latter because the distillation approach has a hard requirement of using a pretrained score based diffusion. 
In order to train in isolation we ned to leverage the following unbiased estimator:

$$ \nabla \log p_t(x_t) = - \mathbb{E} \left[ \frac{x_t - x}{t^2} \middle| x_t \right] \qquad \text{(29)}$$

where $x \sim p_\text{data}$ and $x_t \sim \mathcal{N}(x; t^2 I)$.

That is, given $x$ and $x_t$, we can estimate $\nabla \log p_t(x_t)$ with $-(x_t - x) / t^2$.
This unbiased estimate suffices to replace the pre-trained diffusion model in consistency distillation
when using the Euler ODE solver in the limit of $N \to \infty$ <d-cite key="song2023consistency"></d-cite>.


Song et al. <d-cite key="song2023consistency"></d-cite> justify this with a further theorem in their paper and show that the consistency training objective (CT loss)
can then be defined as:

<p align="center">
$\mathcal{L}_{CT}^N (\theta, \theta^-)$ = $\mathbb{E}[\lambda(t_n)d(f_\theta(x + t_{n+1} \mathbf{z}, t_{n+1}), f_{\theta^-}(x + t_n \mathbf{z}, t_n))]$ $\qquad \text{(30)}$
</p>

where $\mathbf{z} \sim \mathcal{N}(0, I)$.

Crucially, $\mathcal{L}(\theta, \theta^-)$ only depends on the online network $f_\theta$, and the target network
$f_{\theta^-}$, while being completely agnostic to diffusion model parameters $\phi$.

<!---
### Visualization

<p align="center">
<img src="readme_material//consistency_mnist_tiny_example.png" alt="Consistency Graph 2" width="800"/>
</p>
<p align="center">
Figure #(?): Consistency model example for the MNIST dataset.
</p>

(TBA - visualizations of the molecules if possible)


<img src="readme_material//consistency_mnist_dataset.png" alt="Consistency Graph 2" width="800"/>
*Figure #: Consistency model working for the MNIST dataset.*


We conducted the expirement of the consistency model working for the MNIST dataset. During training we sampled after
10 epochs and visualized the results. In figure # we can see the MNIST digits in different epochs. What's really
noteworthy is that those samples were generated in one step and not by doing the denoising process what takes a
very long time and a high number of steps, usually 1000-4000, which saves a lot of time and computation power
during sampling.


(TBA - maybe put back some talking about the images idk yet [Martin])
-->


### EDM Consistency Model Results

We were able to successfully train EDM as a consistency model in isolation. We achieved nearly identical training
loss curves, both in magnitude of the NLL and convergence rate as shown in figure 9: 

<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
        <figure class="custom-figure">
            {% include figure.liquid loading="eager" path="assets/img/2024-06-30-equivariant_diffusion/results_edm_orig_train_loss.png" class="img-fluid rounded z-depth-1 custom-image" zoomable=true %}
            <figcaption class="text-center mt-2">Training loss curves for original EDM</figcaption>
        </figure>
    </div>
    <div class="col-sm mt-3 mt-md-0">
        <figure class="custom-figure">
            {% include figure.liquid loading="eager" path="assets/img/2024-06-30-equivariant_diffusion/results_consistency_train_loss.png" class="img-fluid rounded z-depth-1 custom-image" zoomable=true %}
            <figcaption class="text-center mt-2">Training loss curves for consistency model EDM</figcaption>
        </figure>
    </div>
</div>
<div class="row">
    <div class="col text-center mt-3">
        <p>Figure 9: Training loss curves for original EDM (left), and consistency model EDM (right)</p>
    </div>
</div>

<style>
    .custom-figure .custom-image {
        height: 350px; /* Set a fixed height for both images */
        width: auto; /* Maintain aspect ratio and adjust width accordingly */
        max-width: 100%; /* Ensure the image doesn't exceed the container width */
    }
</style>


For validation and testing, we compared samples from an EMA model against the corresponding ground truth sample,
since consistency models are trained to produce samples directly on the data distribution. 
We achieved similar convergence rates for both val and test losses but with a different magnitude due to the 
changed objective as show on figure 10:

<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
        <figure class="custom-figure">
            {% include figure.liquid loading="eager" path="assets/img/2024-06-30-equivariant_diffusion/results_edm_orig_val_loss.png" class="img-fluid rounded z-depth-1 custom-image" zoomable=true %}
            <figcaption class="text-center mt-2">Val loss curves for original EDM</figcaption>
        </figure>
    </div>
    <div class="col-sm mt-3 mt-md-0">
        <figure class="custom-figure">
            {% include figure.liquid loading="eager" path="assets/img/2024-06-30-equivariant_diffusion/results_consistency_val_loss.png" class="img-fluid rounded z-depth-1 custom-image" zoomable=true %}
            <figcaption class="text-center mt-2">Val loss curves for consistency model EDM</figcaption>
        </figure>
    </div>
</div>
<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
        <figure class="custom-figure">
            {% include figure.liquid loading="eager" path="assets/img/2024-06-30-equivariant_diffusion/results_edm_orig_test_loss.png" class="img-fluid rounded z-depth-1 custom-image" zoomable=true %}
            <figcaption class="text-center mt-2">Test loss curves for original EDM</figcaption>
        </figure>
    </div>
    <div class="col-sm mt-3 mt-md-0">
        <figure class="custom-figure">
            {% include figure.liquid loading="eager" path="assets/img/2024-06-30-equivariant_diffusion/results_consistency_test_loss.png" class="img-fluid rounded z-depth-1 custom-image" zoomable=true %}
            <figcaption class="text-center mt-2">Test loss curves for consistency model EDM</figcaption>
        </figure>
    </div>
</div>
<div class="row">
    <div class="col text-center mt-3">
        <p>Figure 10: Val (top) and Test (bottom) loss curves for original EDM (left), and consistency model EDM (right)</p>
    </div>
</div>

<style>
    .custom-figure .custom-image {
        height: 350px; /* Set a fixed height for all images */
        width: auto; /* Maintain aspect ratio and adjust width accordingly */
        max-width: 100%; /* Ensure the image doesn't exceed the container width */
    }
</style>


These results were obtained using the same EGNN back-bone, batch-size, 
learning rate, and other relevant hyperparameters, only differing in the number of epochs completed.
However, given the displayed loss curves, we have little reason to believe that training the consistency model
for longer would be beneficial.

Using single-step sampling with consistency models, we were only able to reliably achieve around 15% atom stability in
the best case scenario with a large batch size show is figure 11. This low atom stability number could be due to the architecture and the way one step generation works in consistency models. We leave as a future expirement to try different values of sampling other than one step. We were not successful in generating any stable molecules using
the consistency model.

<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
        <figure>
            {% include figure.liquid loading="eager" path="assets/img/2024-06-30-equivariant_diffusion/results_consistency_atom_stability.png" class="img-fluid rounded z-depth-1" zoomable=true %}
            <figcaption class="text-center mt-2">Figure 11: Best results for atom stability metric using single-step sampling with consistency models trained on batch_size = 1024 for improved stability.</figcaption>
        </figure>
    </div>
</div>


The controlled trade-off between speed and sample quality should be possible with multi-step sampling,
however, all attempts to make multi-step sampling work resulted in decreased atom stability. We further 
discuss the set-up and hypothesise why this did not work in the next section.  


## Conclusion

In conclusion, we largely succeeded in reimplementing the EDM paper in JAX, leading to faster runtime, but 
un-competitive results. Similarly, we implemented and trained EDM as a consistency model, allowing us to 
generate new molecules much in a single step, however, we did not manage to make multi-step generation work. 
As such, the consistency model also did not achieve competitive results. 

Although the results are not close to state of the art, we are confident that these methods can achieve better performance 
with more development time, and in their current state, can serve as a good proof of concept. A natural direction
for future research is to continue investigating the poor performance of the current implementation and fix the
underlying issues and suspected bugs to get competitive results.

