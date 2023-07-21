<h1> Abstract </h1>
This work report examines the article "Pricing Options and Computing Implied Volatilities using Neural Networks" and its proposed deep learning model for option pricing and implied volatility estimation. The article presents a novel approach to option pricing using a feedforward neural network architecture trained on a dataset of historical stock prices and option prices. The model is shown to outperform traditional methods such as the Black-Scholes formula and Heston Model in terms of accuracy and efficiency.
<br>To validate the findings of the article, we implemented the deep learning model ourselves. We analyzed the methodology and results presented in the article and compared them with existing literature on option pricing and implied volatility estimation
<br>We hope to gain a deeper understanding of the neural network approach to option pricing and its potential applications in the financial industry.
<br>
<br>
<h1> Table of Contents </h1>
<div class="alert alert-block alert-info" style="margin-top: 20px">
    <ol>
        <li><a href="#Introduction">Introduction</a></li>
        <li><a href="#Black-Scholes Model">Black-Scholes Model</a></li>
        <li><a href="#Heston Model">Heston Model</a></li>
        <li><a href="#Artificial Neural Network (ANN)">Artificial Neural Network (ANN)</a></li>
        <li><a href="#Evaluation Metrics">Evaluation Metrics</a></li>
        <li><a href="#Reference">Reference</a></li>
    </ol>
</div>
<br>
<hr>


<div id="Introduction">
    <h1>Introduction</h1>
</div>

Computational finance is a rapidly growing field that uses advanced mathematical and computational tools to analyze and model financial markets, instruments, and risks. In this field, numerical methods are commonly used to solve complex financial problems that cannot be solved analytically. These methods involve approximating the solutions to mathematical equations using algorithms and computer simulations.
<br>One of the most important problems in computational finance is option pricing. Options are financial contracts that give the holder the right, but not the obligation, to buy or sell an underlying asset at a predetermined price (strike price) on or before a specified date (maturity). The price of an option is determined by several factors, including the current price of the underlying asset, the strike price, the time to maturity, and the volatility of the underlying asset.
<br>Traditional methods for pricing options, such as the Black-Scholes formula and the Heston model, have limitations in accurately pricing options for a wide range of strike prices and maturities. In this work report, we introduce a novel approach to option pricing using an Artificial Neural Network (ANN) trained on data obtained from the Black-Scholes and Heston models.
<br>The Black-Scholes model is a widely used model for option pricing that assumes that the underlying asset follows a geometric Brownian motion and that the volatility of the asset is constant over time. The Heston model is a popular model for option pricing that takes into account the stochastic nature of volatility, which is a key factor in option pricing. 
<br>In recent years, researchers have turned to machine learning techniques, such as neural networks, to develop more accurate and efficient methods for option pricing and implied volatility estimation. The article "Pricing Options and Computing Implied Volatilities using Neural Networks" presents a new approach to option pricing and implied volatility estimation using neural networks. The authors introduce a deep learning model based on a feedforward neural network architecture that is trained using a dataset of historical stock prices and option prices. The model is capable of accurately pricing options and estimating implied volatilities for a wide range of strike prices and maturities.
<br>Overall, this work report aims to provide a critical evaluation of the proposed approach to option pricing using the ANN and compare its performance with traditional methods such as the Black-Scholes formula and the Heston model. By introducing the Black-Scholes and Heston models and demonstrating the training of the ANN with data from these models, we hope to provide a comprehensive understanding of the approach.

<div id="Black-Scholes Model">
    <h1>Black-Scholes Model</h1>
</div>

The Black-Scholes model is a mathematical model used to calculate the theoretical price of options. It was developed by economists Fischer Black and Myron Scholes in 1973, with contributions from Robert C. Merton. The model is widely used in finance to determine the fair value of options, both in the context of stocks and other financial instruments.

The Black-Scholes model makes several assumptions:

1. Efficient markets: The model assumes that markets are efficient and there are no transaction costs or restrictions on short-selling.
2. Log-normal distribution of stock prices: The model assumes that the price of the underlying asset follows a log-normal distribution. This means that the stock price movements are assumed to be continuous and can take on any positive value.
3. Constant risk-free interest rate: The model assumes a constant risk-free interest rate over the life of the option. This rate is typically based on the prevailing risk-free rate, such as the interest rate on government bonds.
4. Constant volatility: The model assumes that the volatility of the underlying asset's returns remains constant over the life of the option. Volatility measures the magnitude of price fluctuations of the underlying asset.
5. No dividends: The model assumes that the underlying asset does not pay any dividends during the life of the option. If dividends are present, adjustments need to be made to account for their effect on the option price.

The Black-Scholes model calculates the price of a European call or put option based on several variables: the current price of the underlying asset (S), the strike price of the option (K), the time to expiration (T), the risk-free interest rate (r), and the volatility of the underlying asset (σ). The model provides a formula that gives the theoretical option price based on these inputs.

The formula for the price of a European call option according to the Black-Scholes model is:

$$
 V = S \cdot N(d1) - K \cdot e^{(-r \ \cdot \ T)} \cdot N(d2)
$$

Where:

* V is the price of the call option,
* S is the current price of the underlying asset,
* N() represents the cumulative standard normal distribution,
* d1 and d2 are calculated as follows:
 $$ d1= \frac{\ln\left(\frac{S}{K}\right) + \left(r + \frac{\sigma^2}{2}\right) \cdot T}{\sigma \cdot \sqrt{T}} \\$$
 
$$ d2 = d_1 - \sigma \cdot \sqrt{T}  $$
* K is the strike price of the option,
* r is the risk-free interest rate,
* T is the time to expiration of the option (measured in years), and
* σ is the volatility of the underlying asset.

<h2>Implied volatility </h2>

Implied volatility is a measure of the market's expectation for the future volatility of an underlying asset's price. It is a critical component in options pricing models, such as the Black-Scholes model, as it reflects the market's consensus on the potential magnitude of future price fluctuations.
<br>Traders and investors often use implied volatility to assess the relative attractiveness of options. If the implied volatility is high, options may be relatively expensive, reflecting higher expected price fluctuations. Conversely, low implied volatility may make options comparatively cheaper, indicating lower expected price swings.
<br>Implied volatility is considered an important quantity in finance. Given an observed market option price V, the Black-Scholes implied volatility σ∗ can be determined by solving BS(σ∗ ; S, K, T, r) = V.


<div id="Heston Model">
    <h1>Heston Model</h1>
</div>

The Heston model is a mathematical model used to describe the dynamics of stock prices and is widely used in the field of quantitative finance. It was developed by Steven Heston in 1993 and is an extension of the Black-Scholes model, which is a basic option pricing model.

The Heston model introduces stochastic volatility, meaning that the volatility of the underlying asset is not constant but follows a random process. This addresses one of the limitations of the Black-Scholes model, which assumes a constant volatility.

The Heston model consists of two stochastic differential equations (SDEs) that describe the dynamics of the underlying asset price and its volatility. The first equation describes the stock price dynamics, and the second equation describes the volatility dynamics.

The main features of the Heston model include:
1. Stochastic volatility: The volatility of the underlying asset is modeled as a stochastic process.
2. Volatility mean-reversion: The volatility process tends to revert to a long-term average level.
3. Volatility smile: The model is able to capture the characteristic smile-shaped volatility curve observed in the market, where the implied volatility of options varies with the strike price.

<h4> Heston’s Stochastic Volatility Model under real world probability measure </h4>

$$ \large dS_t = r S_t dt + \sqrt{v_t} S_t dW^\mathbb{P}_{1,t} $$
$$\large dv_t = \kappa (\theta – v_t)dt + \sigma \sqrt{v_t} dW^\mathbb{P}_{2,t} $$
$$\large \rho dt = dW^\mathbb{P}_{2,t} dW^\mathbb{P}_{2,t}
$$


<h4> Using standard arbitrage arguments we arrive at Garman’s partial differential equation: </h4>

$\large \frac{\delta V}{\delta t} + \frac{S^2 v}{2}\frac{\delta^2 V}{\delta S^2} + rS\frac{\delta V}{\delta S} – rV + \kappa(\theta-v)\frac{\delta V}{\delta v} + \frac{\sigma^2 v}{2}\frac{\delta^2 V}{\delta v^2} + \rho \sigma Sv \frac{\delta^2 V}{\delta S \delta v} = 0$

Notation:

* $S_t$ Equity spot price, financial index
* $v_t$ Variance
* C  European call option price
* K  Strike price
* $W_{1,2}$ Standard Brownian movements.
* r  Interest rate
* $\kappa$ Mean reversion rate
* $\theta$ Long run variance
* $v_0$ Initial variance
* $\sigma$ Volatility of variance
* $\rho$ Correlation parameter
* t Current date
* T  Maturity date.
<h2>Fourier-Cosine method </h2>

$$
v\left(\mathbf{x}, t_0, u_0\right) \approx \mathbf{K} e^{-r \Delta t} \cdot \operatorname{Re}\left\{{\sum_{k=0}^{N-1} }^{\prime} \varphi_{h e s}\left(\frac{k \pi}{b-a} ; u_0\right) U_k \cdot e^{i k \pi \frac{\mathbf{x}-n}{b-a}}\right\} .
$$

The characteristic function of the log-asset price, φ_hes (ω;u_0 ), reads

$$
\begin{aligned}
\varphi_{h e s}\left(\omega ; u_0\right)= & \exp \left(i \omega \mu \Delta t+\frac{u_0}{\eta^2}\left(\frac{1-e^{-D \Delta t}}{1-G e^{-D \Delta t}}\right)(\lambda-i \rho \eta \omega-D)\right) \\
& \cdot \exp \left(\frac{\lambda \bar{u}}{\eta^2}\left(\Delta t(\lambda-i \rho \eta \omega-D)-2 \log \left(\frac{1-G e^{-D \Delta t}}{1-G}\right)\right)\right),
\end{aligned}
$$
with
$$
D=\sqrt{(\lambda-i \rho \eta \omega)^2+\left(\omega^2+i \omega\right) \eta^2} \text  { and }   G=\frac{\lambda-i \rho \eta \omega-D}{\lambda-i \rho \eta \omega+D} .
$$

$$
U_k= \begin{cases}\frac{2}{b-a}\left(\chi_k(0, b)-\psi_k(0, b)\right) & \text { for a call, } \\ \frac{2}{b-a}\left(-\chi_k(a, 0)+\psi_k(a, 0)\right) & \text { for a put. }\end{cases}
$$

$$
\begin{aligned}
& \chi_k(c, d):=\frac{1}{1+\left(\frac{k \pi}{b-a}\right)^2}\left[\cos \left(k \pi \frac{d-a}{b-a}\right) e^d-\cos \left(k \pi \frac{c-a}{b-a}\right) e^c\right. \\
& \left.+\frac{k \pi}{b-a} \sin \left(k \pi \frac{d-a}{b-a}\right) e^d-\frac{k \pi}{b-a} \sin \left(k \pi \frac{c-a}{b-a}\right) e^c\right] \\

& \psi_k(c, d):= \begin{cases}{\left[\sin \left(k \pi \frac{d-a}{b-a}\right)-\sin \left(k \pi \frac{c-a}{b-a}\right)\right] \frac{b-a}{k \pi},} & k \neq 0, \\
(d-c), & k=0 .\end{cases} \\
&
\end{aligned}
$$
How to determine the interval [a, b] is as follows,
$$ \begin{align*}
[a, b] = [ c_1 - L\sqrt{\left|c_2 \right|}, c_1 + L\sqrt{\left|c_2 \right|} ]
\end{align*}
$$
$$
\begin{align*}
 c_1 &= r\tau + (1 - \exp{(-\kappa  \tau)}) \frac{(\theta - v_0)}{2\kappa} - 0.5\theta\tau
\end{align*}
$$
$$
\begin{align*}
c_2&= \frac{1}{(8\kappa^3)}  \Bigg[ \sigma\tau\kappa\exp({-\kappa\tau})(v_0-\theta)(8\kappa\rho - 4\sigma)\
& +\kappa\rho\sigma(1-\exp({-\kappa\tau}))(16\theta - 8v_0)\
& +2\theta\kappa\tau(-4\kappa*\rho\sigma+\sigma^2 + 4\kappa^2)\
&+\sigma^2((\theta-2v_0)\exp({-2\kappa\tau})+\theta(6\exp({-\kappa\tau})-7)+2v_0)\
&+8\kappa^2(v_0-\theta)(1-\exp({-\kappa\tau})) \Bigg]
\end{align*}
$$

This characteristic function is uniquely specified, since we take $\sqrt{(x+y i)}$ such that its real part is nonnegative, and we restrict the complex logarithm to its principal branch. In this case the resulting characteristic function is the correct one for all complex $\omega$ in the strip of analycity of the characteristic function.
<br>where Σ' indicates that the first term in the summation is weighted by one-half.
<div id="Artificial Neural Network (ANN)">
    <h1>Artificial Neural Network (ANN)</h1>
</div>

<br>
ANNs generally constitute three levels of components, i.e., neurons, layers and the architecture from bottom to top. The architecture is determined by a combination of different layers, that are made up of numerous artificial neurons. A neuron, which involves learnable weights and biases, is the fundamental unit of ANNs. By connecting the neurons of adjacent layers, output signals of a previous layer enter a next layer as input signals. By stacking layers on top of each other, signals travel from the input layer through the hidden layers to the output layer potentially through cyclic or recurrent connections, and the ANN builds a mapping among input-output pairs.
<center>

<a><img src="https://i.stack.imgur.com/W6FuR.png" width="300">
<img src="https://www.mdpi.com/energies/energies-13-03913/article_deploy/html/images/energies-13-03913-g006.png" width="300"></a>

</center>

* <h4> Parameters of the ANN </h4>

<center>

| Parameters        | Options            |
|-------------------|--------------------|
| Hidden layers     | 4                  |
| Neurons(each layer)| 400                |
| Activation        | ReLU               |
| Dropout rate      | 0.0                |
| Batch-normalization| No                 |
| Initialization    | Glorot_uniform     |
| Optimizer         | Adam               |
| Batch size        | 1024               |

</center>

<div id="Evaluation Metrics">
    <h1>Evaluation Metrics</h1>
</div>

We compare the actual values and predicted values to calculate the accuracy of a regression model. Evaluation metrics provide a key role in the development of a model, as it provides insight to areas that require improvement.

There are different model evaluation metrics:

*   MAE: stands for Mean Absolute Error, which is another commonly used loss function in machine learning. It measures the average absolute difference between the predicted values and the true values.  The goal is to minimize the MAE in order to make better predictions. The MAE is more robust to outliers than the MSE, which tends to be sensitive to outliers.
$$ \begin{align}{MAE} = \frac{1}{n_{\text{samples}}} \sum_{i=0}^{n_{\text{samples}}-1} \left| y_i - \hat{y}_i \right|.\end{align}$$

*   MSE: stands for Mean Squared Error, which is a commonly used loss function for regression problems in machine learning. It measures the average squared difference between the predicted values and the true values. The goal is to minimize the MSE in order to make better predictions.
$$ \begin{align} {MSE} = \frac{1}{n_\text{samples}} \sum_{i=0}^{n_\text{samples} - 1} (y_i - \hat{y}_i)^2.\end{align} $$

*   R2-squared: is not an error, but rather a popular metric to measure the performance of your regression model. It represents how close the data points are to the fitted regression line. The higher the R-squared value, the better the model fits your data. The best possible score is 1.0 and it can be negative (because the model can be arbitrarily worse).
$$ \begin{align} R2&=1-\frac{\text{sum squared regression (SSR)}}{\text{total sum of squares (SST)}},\\ &=1-\frac{\sum({y_i}-\hat{y_i})^2}{\sum(y_i-\bar{y})^2}. \end{align} $$

<div id="Reference">
    <h1>Reference</h1>
</div>

* [ Liu, S., Oosterlee, C. W., & Bohte, S. M. (2019). Pricing Options and Computing Implied Volatilities using Neural Networks. Applied Sciences, 9(3), 486. doi: 10.3390/app9030486.]
(https://arxiv.org/abs/1901.08943)
* [ Fang, F., & Oosterlee, C. W. (2008). A novel pricing method for European options based on Fourier-cosine series expansions. SIAM Journal on Scientific Computing, 31(2), 826-848. doi: 10.1137/070679065]
(https://faculty.baruch.cuny.edu/lwu/890/FanOosterlee2008.pdf)
* [ The thesis is "The COS Method: An Efficient Fourier Method for Pricing Financial Derivatives." The author is Fang Fang, The thesis was defended on December 8, 2010, at the Delft University of Technology in the Netherlands.] (https://core.ac.uk/download/pdf/301638164.pdf#page=50&zoom=100,101,577.)
* [Dunn, R., Hauser, P., Seibold, T., & Gong, H. (2015). Estimating Option Prices with Heston’s Stochastic Volatility Model. Department of Mathematics and Statistics, Valparaiso University.]  (https://www.valpo.edu/mathematics-statistics/files/2015/07/Estimating-Option-Prices-with-Heston.)
* [S. Heston, A closed-form solution for options with stochastic volatility with applications to bond and currency options, Rev. Financ. Studies, 6 (1993), pp. 327–343.]
(https://faculty.baruch.cuny.edu/lwu/890/Heston93.pdf)
* [Shinde, A. S., & Takale, K. C. (2012). Study of Black-Scholes Model and its Applications. Procedia Engineering, 38, 270-279.]
(https://doi.org/10.1016/j.proeng.2012.06.035)