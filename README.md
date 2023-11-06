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
    <th>CNN Architecture</th>
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
    <td rowspan="4">$n_2$</td>
    <td rowspan="4">$3 \times 6 \times 6$</td>
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
  </tr>
  <tr>
    <td>8192</td>
    <td>$2.9404 \times10^{5}$</td>
    <td>06:02:19</td>
  </tr>
    
</table>
