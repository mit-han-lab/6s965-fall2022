# Lecture 05: Quantization (Part I)

## Note Information

| Title       | Quantization (Part I)                                                |
|-------------|-----------------------------------------------------------------------------------------------------------------|
| Lecturer    | Song Han                                                                                                        |
| Date        | 09/22/2022                                                                                                      |
| Note Author | Pranav Krishna (pkrishna)                                                                                                 |
| Description | The first half ofn an introduction to Quantization, which is a common model compression technique |


## What is Quantization?

Quantization is the process of turning a continuous signal discrete, in a broad sense; it is common in signal processing (where you sample at discrete intervals) and image compression (where you reduce the space of possible colors for each pixel). This technique is *orthogonal* to pruning (from the last two lectures).


## Numeric Data Types

In Machine Learning, Quantization involves changing the data type of each weight to a more restrictive data type (i.e. can be represented in less bits). This section briefly describes the common data types used in Machine Learning models/

### Integers

Integers can be either signed or unsigned. If they are unsigned, then they are just $n$-bit numbers in the range $[0, 2^n-1]$. If they are signed, there are two ways they can be represented.
* *Signed-Magnitude*: represent the numbers $[-2^{n-1}-1, 2^{n-1}-1]$, where the first bit is a 'sign' bit, where $0$ is positive and $1$ is negative; then the rest of the $n-1$ bits represent the number. Its main drawback is that $1000\dots$ and $0000\dots$ represent the same number ($0$), among other quirks
* *Two's Complement*: represent the numbers $[-2^n, 2^{n-1}]$. Here, the idea is that $-x-1 = \sim x$ (bitwise NOT). Note that the negative numbers 'go the other way' compared to the previous representation type. It removes the redundancy of the previous method.

### Fixed Point Numbers

These represent decimals, but behave very similar to integers, in the sense that they are just integers but shifted by a power of $2$. They have a fixed number of bits to represent numbers before and after the decimal.

### Floating Point Numbers

This is a much more common method for representing real numbers. The data is split into three parts - a sign bit (usually the first), the exponent bits, and the mantissa/significant bits/fraction.

A number is then $(-1)^{\text{sign}} \times (1 + \text{mantissa}) + 2^{\text{exponent} - \text{exponent bias}}$, where the mantissa is read as a binary decimanl, and the exponent is read as an integer. If we let $k$ be the number of bits used to represent the exponent, then
