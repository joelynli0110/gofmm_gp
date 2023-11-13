## Implementation

[test_cnngp.py](https://github.com/joelynli0110/gofmm_gp/blob/dev/test_cnngp.py)

## Results

### Linear Solving of GOFMM with cnn_gp

#### CNN Architecture

* 3-layer CNN $n_1$
```
n1 = Sequential(
    Conv2d(kernel_size=3),
    ReLU(),
    Conv2d(kernel_size=3, stride=2),
    ReLU(),
    Conv2d(kernel_size=7, padding=0)
)
```

* 3-layer CNN $n_2$
```
n2 = Sequential(
    Conv2d(kernel_size=3),
    ReLU(),
    Conv2d(kernel_size=3, stride=2),
    ReLU(),
    Conv2d(kernel_size=3, padding=0)
)
```

* 2-layer CNN $n_3$
```
n3 = Sequential(
    Conv2d(kernel_size=2),
    ReLU(),
    Conv2d(kernel_size=2, padding=0)
)
```

<table>
  <tr>
    <th> CNN Architecture </th>
    <th> Input dimension <br> (C $\times$ H $\times$ W) </th>
    <th>Problem Size</th>
    <th>Norm Error</th>
    <th>Duration</th>
  </tr>
  <tr>
    <td rowspan="3">$n_1$</td>
    <td rowspan="3">$3 \times 14 \times 14$</td>
    <td>512</td>
    <td>$2.86 \times10^{-1}$</td>
    <td>00:00:38</td>
  </tr>
  <tr>
    <td>1024</td>
    <td>$5.31 \times 10 ^0$</td>
    <td>00:03:06</td>
  </tr>
  <tr>
    <td>2048</td>
    <td>$2.53 \times 10 ^3$</td>
    <td>00:19:21</td>
  </tr>
  <tr>
    <td rowspan="5">$n_2$</td>
    <td rowspan="5">$3 \times 6 \times 6$</td>
    <td>512</td>
    <td>$8.22 \times10^{-1}$</td>
    <td>00:00:24</td>
  </tr>
  <tr>
    <td>1024</td>
    <td>$1.04 \times10^{2}$</td>
    <td>00:02:52</td>
  </tr>
  <tr>
    <td>2048</td>
    <td>$2.3563 \times10^{4}$</td>
    <td>00:18:23</td>
  </tr>
  <tr>
    <td>4096</td>
    <td>$6.235 \times10^{3}$</td>
    <td>01:27:40</td>
  <tr>
  </tr>
    <td rowspan="6">$n_3$</td>
    <td rowspan="6">$3 \times 2 \times 2$</td>
    <td>512</td>
    <td>$9.6959 \times10^{4}$</td>
    <td> 01:28:30</td>
  </tr>
  </tr>
    <td>1024</td>
    <td>$1.52447 \times10^{5}$</td>
    <td> 01:28:30</td>
  </tr>
  </tr>
    <td>2048</td>
    <td>$6.4758 \times10^{4}$</td>
    <td> 00:18:21</td>
  </tr>
  </tr>
    <td>4096</td>
    <td>$6.4897 \times10^{4}$</td>
    <td> 01:27:18</td>
  </tr>
  <tr>
    <td>8192</td>
    <td>$2.9404 \times10^{5}$</td>
    <td>06:02:19</td>
  </tr>
  <tr>
    <td>16384</td>
    <td>$$</td>
    <td></td>
  </tr>
</table>

### Linear Solving of GOFMM with cnn_gp on MNIST
```var_bias = 7.86, var_weight = 2.79```
```
n5 = Sequential(
    Conv2d(kernel_size=3, padding="same", var_weight=var_weight * 7**2, var_bias=var_bias),
    ReLU(),
    Conv2d(kernel_size=28, padding=0, var_weight=var_weight,var_bias=var_bias)
)
```

```
n6 = Sequential(
    Conv2d(kernel_size=3, padding="same", var_weight=var_weight * 7**2, var_bias=var_bias),
    ReLU(),
    Conv2d(kernel_size=14, padding=0, var_weight=var_weight,var_bias=var_bias)
)
```

```
n7 = Sequential(
    Conv2d(kernel_size=3, padding="same", var_weight=var_weight * 7**2, var_bias=var_bias),
    ReLU(),
    Conv2d(kernel_size=7, padding=0, var_weight=var_weight,var_bias=var_bias)
)
```


<table>
  <tr>
    <td colspan="2", rowspan="2"> </td>
    <th colspan="6"> CNN Architecture</th>
  </tr>
  <tr>
    <td> $n_1$ </td>
    <td> $n_2$ </td>
    <td> $n_3$ </td>
    <td> $n_5$ </td>
    <td> $n_6$ </td>
    <td> $n_7$ </td>
  </tr>
  <tr>
    <th rowspan="8"> Problem size </th>
  </tr>
  <tr>
    <td>512</td>
    <td>$9.627 \times 10^{-8}$</td>
    <td>$7.508 \times 10^{-6}$</td>
    <td></td>
    <td>$3.105 \times 10^{-11}$</td>
    <td> </td>
    <td> </td>
  </tr>
  <tr>
    <td>1024</td>
    <td>$9.960 \times 10^{0}$</td>
    <td>$4.178 \times 10^{2}$</td>
    <td> singular </td>
    <td>$2.809 \times 10^{-3}$</td>
    <td> </td>
    <td> </td>
  </tr>
  <tr>
    <td>2048</td>
      <td> </td>
    <td> </td>
      <td> </td>
    <td> </td>
    <td> $5.089 \times 10^3$ </td>
    <td>$1.163 \times 10^4$ </td>
  </tr>
  <tr>
    <td>4096</td>
    <td> </td>
    <td> </td>
      <td> </td>
    <td> </td>
      <td> </td>
    <td> $3.278 \times 10^5$ </td>
  </tr>
  <tr>
    <td>8192</td>
  </tr>
  <tr>
    <td>16384</td>
  </tr>
